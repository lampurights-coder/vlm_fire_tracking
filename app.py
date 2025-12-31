import base64
import re
import requests
from pathlib import Path
from PIL import Image
import cv2
import os
import tempfile
import uvicorn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse


# ---------------- CORE DETECTION UTILS ---------------- #


def extract_labels_and_boxes(text: str, class_names):
    """
    Parse model output like: 'fire<loc10><loc20><loc30><loc40> ...'
    and return labels + list of [x1,y1,x2,y2] in 0–999 coords.
    """
    label_to_id = {v: k for k, v in class_names.items()}
    pattern = r'(\w+)<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>'
    matches = re.findall(pattern, text)

    labels = []
    locs_list = []
    for match in matches:
        label = match[0]
        if label in label_to_id:
            labels.append(label)
            locs = list(map(int, match[1:]))
            locs_list.append(locs)
        else:
            # Optional: ignore unknown labels
            print(f"Ignoring unknown label: {label}")
    return labels, locs_list


def detector(
    input_image_path: Path,
    server_url: str,
    prompt: str = "detect fire bbox",
    model_name: str = "your-vlm-model-name",
):
    """
    Calls an OpenAI-compatible VLM server and returns labels + pixel bboxes.
    """
    LOC_SCALE = 1000
    CLASS_NAMES = {0: "fire", 1: "smoke", 2: "other"}

    def decode_image_to_b64(image_path: Path) -> str:
        with image_path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def scale_locs_to_boxes(locs_list, width: int, height: int):
        boxes = []
        for locs in locs_list:
            if len(locs) != 4:
                raise ValueError(
                    f"Invalid number of loc tokens for a box: {len(locs)} (must be 4)"
                )
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

    # OpenAI-compatible payload
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

    # Call VLM server
    resp = requests.post(server_url, json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()

    # Some servers return a list; normalise
    if isinstance(result, list):
        result = result[0]

    try:
        text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected response format:\n{result}") from e

    print("RAW MODEL OUTPUT:")
    print(text)

    # Extract labels + locs (0–999)
    labels, locs_list = extract_labels_and_boxes(text, CLASS_NAMES)
    print("Extracted labels:", labels)
    print("Extracted locs (0-999):", locs_list)

    # Scale to pixel boxes
    boxes = []
    if locs_list:
        boxes = scale_locs_to_boxes(locs_list, W, H)
        print("Pixel boxes:", boxes)

    return labels, boxes


def draw(image_path: Path, labels, boxes):
    """
    Draw bboxes + labels on image and save to a temp file.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(labels):
            cv2.putText(
                image,
                labels[i],
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    output_path = Path(tempfile.gettempdir()) / "Bounding_Box_Visualization.jpg"
    cv2.imwrite(str(output_path), image)
    return output_path


# ---------------- FASTAPI APP ---------------- #


app = FastAPI(title="Fire Detector API with Visualization")

# IMPORTANT: default to 127.0.0.1 instead of 'llama'
VLM_SERVER_URL = os.environ.get(
    "VLM_SERVER_URL",
    "http://127.0.0.1:8033/v1/chat/completions",
)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        labels, boxes = detector(tmp_path, VLM_SERVER_URL)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error contacting VLM server: {e}",
        )
    finally:
        # Cleanup temp file
        if tmp_path.exists():
            tmp_path.unlink()

    detections = [{"label": l, "bbox": b} for l, b in zip(labels, boxes)]
    return JSONResponse({"detections": detections})


@app.post("/detect_and_draw")
async def detect_and_draw(file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        labels, boxes = detector(tmp_path, VLM_SERVER_URL)
        output_path = draw(tmp_path, labels, boxes)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error contacting VLM server: {e}",
        )
    finally:
        # Keep tmp_path only if you want, otherwise cleanup:
        if tmp_path.exists():
            tmp_path.unlink()

    return FileResponse(
        output_path,
        media_type="image/jpeg",
        filename="result.jpg",
    )


# ---------------- ENTRYPOINT ---------------- #


if __name__ == "__main__":
    # Run with: python app.py
    # Or:      uvicorn app:app --host 0.0.0.0 --port 8765
    uvicorn.run(app, host="0.0.0.0", port=8765)
