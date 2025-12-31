import base64
import json
import re
import requests
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

def extract_labels_and_boxes(text: str, class_names):
    """
    Extract labels and bounding boxes from the model output string.
    Args:
        text (str): Input string containing labels and <locXXX> tokens
        class_names (dict): Dictionary mapping class IDs to names
    Returns:
        list[str]: List of labels (names)
        list[tuple[int, int, int, int]]: List of bounding boxes as (x1, y1, x2, y2) in 0-999 scale
    """
    # Reverse mapping: label -> id
    label_to_id = {v: k for k, v in class_names.items()}
    
    # Pattern to match each group: label<loc\d+><loc\d+><loc\d+><loc\d+>
    pattern = r'(\w+)<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>'  # Assumes exactly 4 locs per label
    matches = re.findall(pattern, text)
    
    labels = []
    locs_list = []
    for match in matches:
        label = match[0]
        if label in label_to_id:  # Only include known labels
            labels.append(label)
            locs = list(map(int, match[1:]))
            locs_list.append(locs)
        else:
            print(f"Ignoring unknown label: {label}")
    
    return labels, locs_list

def detector(input_image_path: Path, server_url: str, prompt: str = "detect fire bbox", model_name: str = "your-vlm-model-name"):
    """
    Detects objects (e.g., fire) in an image using a VLM server, parses bounding box location tokens,
    and returns the list of labels and bounding boxes in pixel coordinates.
   
    Args:
        input_image_path (Path): Path to the input image.
        server_url (str): URL of the VLM server (e.g., "http://127.0.0.1:8033/v1/chat/completions").
        prompt (str): Detection prompt for the VLM (default: "detect fire bbox").
        model_name (str): Name of the model to use on the server.
   
    Returns:
        list[str]: List of detected labels (names)
        list[tuple[int, int, int, int]]: List of bounding boxes as (x1, y1, x2, y2) in pixel coordinates.
    """
    LOC_SCALE = 1000
    CLASS_NAMES = {
        0: "fire",
        1: "smoke",
        2: "other"
    }
    
    def decode_image_to_b64(image_path: Path) -> str:
        with image_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def scale_locs_to_boxes(locs_list, width: int, height: int):
        boxes = []
        for locs in locs_list:
            if len(locs) != 4:
                raise ValueError(f"Invalid number of loc tokens for a box: {len(locs)} (must be 4)")
            x1_loc, y1_loc, x2_loc, y2_loc = locs
            x1 = int(x1_loc / LOC_SCALE * width)
            y1 = int(y1_loc / LOC_SCALE * height)
            x2 = int(x2_loc / LOC_SCALE * width)
            y2 = int(y2_loc / LOC_SCALE * height)
            boxes.append((x1, y1, x2, y2))
        return boxes
    
    # Load image
    image = Image.open(input_image_path).convert("RGB")
    W, H = image.size
    
    # Encode image
    b64_image = decode_image_to_b64(input_image_path)
    
    # Prepare payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 512,
    }
    
    # Send request
    resp = requests.post(server_url, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    
    # Extract text output
    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format: {json.dumps(result, indent=2)}") from e
    
    print("RAW MODEL OUTPUT:")
    print(text)
    
    # Extract labels and locs
    labels, locs_list = extract_labels_and_boxes(text, CLASS_NAMES)
    print("Extracted labels:", labels)
    print("Extracted locs (0-999):", locs_list)
    
    # Scale to pixel boxes
    boxes = []
    if locs_list:
        boxes = scale_locs_to_boxes(locs_list, W, H)
        print("Pixel boxes:", boxes)
    
    return labels, boxes


def draw(image_path, labels, boxes):
    # Load image for drawing (using OpenCV)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Draw bounding boxes and labels
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),  # Green color
            2  # Line thickness
        )
        # Draw label text above the box
        if i < len(labels):
            label = labels[i]
            text_position = (x1, y1 - 10)  # Slightly above the top-left corner
            cv2.putText(
                image,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # Font scale
                (0, 255, 0),  # Green color
                2  # Line thickness
            )

    # Save the output image
    output_path = "Bounding_Box_Visualization.jpg"
    cv2.imwrite(output_path, image)
    print(f"Output image saved to: {output_path}")

def center_of_biggest_bbox(bboxes):
    """
    bboxes: list of [x_min, y_min, x_max, y_max]
    returns: (x_center, y_center) of the largest bbox by area
    """

    if not bboxes:
        raise ValueError("bboxes list is empty")

    max_area = -1
    biggest_bbox = None

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        if width < 0 or height < 0:
            raise ValueError(f"Invalid bbox with negative size: {bbox}")

        area = width * height

        if area > max_area:
            max_area = area
            biggest_bbox = bbox

    x_min, y_min, x_max, y_max = biggest_bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    return int(x_center), int(y_center)


if __name__ == "__main__":
    # Example usage:
    image_path = "/home/ai/Downloads/image (1).webp"
    labels, boxes = detector(Path(image_path), "http://127.0.0.1:8033/v1/chat/completions")
    print(center_of_biggest_bbox(boxes))
    draw(image_path, labels, boxes)