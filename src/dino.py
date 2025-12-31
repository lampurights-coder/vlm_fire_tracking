import math
from functools import lru_cache
from typing import List, Optional
import os
import io
import math
from functools import lru_cache
from typing import Optional, List
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from transformers import AutoModel
from matplotlib import colormaps as cm
from scipy.ndimage import label
from src.detector import detector, center_of_biggest_bbox

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label
from torchvision import transforms
from transformers import AutoModel
import os
os.environ["HF_TOKEN"] = "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"

DEFAULT_TOP_K = 10
NORMALIZE_STATS = {'lvd1689m': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
 'sat493m': {'mean': [0.43, 0.411, 0.296], 'std': [0.213, 0.156, 0.143]}}


class DINOv3PatchSimilarity:
    def __init__(self, full_model_id: str, device_str: Optional[str] = None):
        self.full_model_id = full_model_id
        self.device_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model_cached(full_model_id, self.device_str)
        self.ps = self.infer_patch_size(self.model, 32)
        self.dataset_key = self.extract_dataset_key(full_model_id)
        self.src_state = None
        self.tgt_state = None

    def extract_dataset_key(self, model_id: str) -> str:
        return model_id.split("-pretrain-")[-1]

    @lru_cache(maxsize=3)
    def load_model_cached(self, full_model_id: str, device_str: str):
        model = AutoModel.from_pretrained(full_model_id).to(device_str)
        model.eval()
        return model

    def infer_patch_size(self, model, default: int):
        if hasattr(model.config, "patch_size"):
            ps = model.config.patch_size
            return ps[0] if isinstance(ps, (tuple, list)) else ps
        return default

    def pad_to_multiple(self, img: Image.Image, multiple: int):
        W, H = img.size
        Wp = int(math.ceil(W / multiple) * multiple)
        Hp = int(math.ceil(H / multiple) * multiple)
        canvas = Image.new("RGB", (Wp, Hp))
        canvas.paste(img, (0, 0))
        return canvas

    def preprocess(self, img: Image.Image):
        img = self.pad_to_multiple(img, self.ps)
        stats = NORMALIZE_STATS[self.dataset_key]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"])
        ])
        return transform(img).unsqueeze(0), np.array(img), img

    def create_patch_state(self, img: Image.Image):
        pv, disp, padded = self.preprocess(img)
        pv = pv.to(self.device_str)
        _, _, H, W = pv.shape
        rows, cols = H // self.ps, W // self.ps

        with torch.no_grad():
            hs = self.model(pixel_values=pv).last_hidden_state[0].cpu().numpy()

        n_patches = rows * cols
        patches = hs[-n_patches:].reshape(rows, cols, -1)
        X = patches.reshape(-1, patches.shape[-1])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        state = type("PatchState", (), {})()
        state.rows, state.cols = rows, cols
        state.ps = self.ps
        state.X = X
        state.Xn = Xn
        state.disp = disp
        state.padded = padded
        return state

    def process_images(self, src_img: Image.Image, tgt_img: Image.Image):
        self.src_state = self.create_patch_state(src_img)
        self.tgt_state = self.create_patch_state(tgt_img)

    def idx_to_rc(self, idx, cols):
        return idx // cols, idx % cols

    def coords_to_idx(self, x, y, state):
        x = max(0, min(x, state.cols * state.ps - 1))
        y = max(0, min(y, state.rows * state.ps - 1))
        return (y // state.ps) * state.cols + (x // state.ps)

    # ---------- BBOX FROM PATCH INDICES ----------
    def topk_bbox(self, state, indices: List[int]):
        rows, cols, ps = state.rows, state.cols, state.ps
        mask = np.zeros((rows, cols), dtype=bool)

        for idx in indices:
            r, c = self.idx_to_rc(idx, cols)
            mask[r, c] = True

        labels, num = label(mask)
        if num == 0:
            return None

        best_area, best_bbox = 0, None
        for i in range(1, num + 1):
            ys, xs = np.where(labels == i)
            if len(xs) > best_area:
                best_area = len(xs)
                best_bbox = (
                    xs.min() * ps,
                    ys.min() * ps,
                    (xs.max() + 1) * ps,
                    (ys.max() + 1) * ps,
                )
        return best_bbox

    # ---------- PATCH SELECTION WITH THRESHOLD ----------
    def select_patch(
        self,
        x: int,
        y: int,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: Optional[float] = 0.6,
        max_top_k: Optional[int] = None,
        allow_fallback: bool = False,
    ):
        """
        Returns:
            top_patches : List[PIL.Image]
            best_bbox   : Optional[Tuple[int, int, int, int]]
        """

        q_idx = self.coords_to_idx(x, y, self.src_state)
        qn = self.src_state.Xn[q_idx]

        cos = self.tgt_state.Xn @ qn
        cos = np.clip(cos, -1.0, 1.0)

        # ---- threshold filtering ----
        if similarity_threshold is not None:
            valid = np.where(cos >= similarity_threshold)[0]

            if len(valid) == 0:
                if not allow_fallback:
                    return [], None
                valid = np.argsort(cos)[::-1][:top_k]

            else:
                valid = valid[np.argsort(cos[valid])[::-1]]

        else:
            valid = np.argsort(cos)[::-1][:top_k]

        if max_top_k is not None:
            valid = valid[:max_top_k]

        indices = valid.tolist()
        if len(indices) == 0:
            return [], None

        best_bbox = self.topk_bbox(self.tgt_state, indices)

        top_patches = []
        for idx in indices:
            r, c = self.idx_to_rc(idx, self.tgt_state.cols)
            ps = self.tgt_state.ps
            top_patches.append(
                self.tgt_state.padded.crop(
                    (c * ps, r * ps, (c + 1) * ps, (r + 1) * ps)
                )
            )

        return top_patches, best_bbox

    
def draw_bbox_on_image(image_path, output_path, bbox, color='red', thickness=3):
    """
    Draw bounding box on image and save.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        bbox: Tuple (x1, y1, x2, y2) - top-left and bottom-right corners
        color: BBox color ('red', 'blue', etc. or RGB tuple)
        thickness: Line thickness in pixels
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create drawing context
    draw = ImageDraw.Draw(image)
    
    # Parse bbox (192, 160, 224, 224)
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle [PIL uses [(x1,y1), (x2,y2)] format]
    draw.rectangle([ (x1, y1), (x2, y2) ], outline=color, width=thickness)
    
    # Save with format handling
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        image.save(output_path, 'JPEG', quality=95)
    else:
        image.save(output_path)
    
    print(f"Saved with bbox: {output_path}")
    print(f"BBox: ({x1}, {y1}) -> ({x2}, {y2})")

def resize_and_save_image(input_path, output_path, ratio=1.5):
    """
    Load image, resize, and save with proper format handling.
    """
    # Load image
    image = Image.open(input_path)
    
    # Convert RGBA to RGB for JPEG (most common fix)
    if image.mode in ('RGBA', 'LA', 'P'):
        # Convert to RGB
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        image = background
    
    # Calculate new dimensions
    width, height = image.size
    new_width = int(width / ratio)
    new_height = int(height / ratio)
    
    # Resize
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save with proper format
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        resized_image = resized_image.convert('RGB')  # Ensure RGB for JPEG
        resized_image.save(output_path, 'JPEG', quality=95)
    else:
        resized_image.save(output_path)
    
    print(f"Saved: {output_path}")
    print(f"Original: {width}x{height} -> New: {new_width}x{new_height}")

if __name__ == "__main__":

    sim = DINOv3PatchSimilarity("./dinov3-convnext-tiny-pretrain-lvd1689m")

    input_path = "/home/ai/Desktop/PADDLE/fire_serv/frames/video_test_frame_003254.jpg"
    output_path = "input.jpg"  # Will be converted to RGB JPEG
    
    resize_and_save_image(input_path, output_path, ratio=1.5)
    PATH = "/home/ai/Desktop/PADDLE/fire_serv/frames"

    labels, boxes = detector(Path(output_path), "http://127.0.0.1:8033/v1/chat/completions")
    center = center_of_biggest_bbox(boxes)

    # Get center XY
    # center = yolo_center.get_center_xy(image_path)

    if center:
        x, y = center
        print(f"Center: ({x}, {y})")
    else:
        print("No detections found")

    for img in [PATH + "/" + name for name in os.listdir(PATH)]:
        img_show = img

        src_img = Image.open(output_path)
        tgt_img = Image.open(img_show)
        sim.process_images(src_img, tgt_img)

        patches, bbox = sim.select_patch(x=x, y=y)
        print("Best bbox:", bbox)

        # Usage - just change paths and bbox
        
        result_path = "fianl.jpg"
        try:
            draw_bbox_on_image(img_show, result_path, bbox, color='red', thickness=3)
        except:
            pass

# unset HTTP_PROXY
# unset HTTPS_PROXY
# unset ALL_PROXY
# unset http_proxy
# unset https_proxy
# unset all_proxy