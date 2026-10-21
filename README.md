![Pipeline overview](vl_din.png)

# Fire detection and tracking (Qwen3-VL + DINOv3 + BotSort)

Real-time fire tracking for a video file or an RTSP stream. A fine-tuned Qwen3-VL model proposes fire boxes. DINOv3 patch similarity and BotSort keep those boxes through the frames between VLM calls. A Gradio page shows the live view. A separate FastAPI app can score a single image.

![Detection pipeline](alogrithem.png)

Fine-tuned weights: [larshiakzemil/unsloth_finetune](https://huggingface.co/larshiakzemil/unsloth_finetune).

## What runs

| Piece | Role |
| --- | --- |
| llama.cpp server | OpenAI-compatible chat API for the GGUF VLM on port `8033` |
| `src/detector.py` | Sends a frame, parses `fire` / `smoke` / `other` plus four `<locN>` tokens, scales them to pixels |
| `src/dino.py` | `DINOv3PatchSimilarity` on `./dinov3-convnext-tiny-pretrain-lvd1689m` |
| BotSort (`boxmot`) | Track IDs. ReID is off (`with_reid=False`) |
| `src/frame_gather.py` | Reads `sources` from `configs/config.yaml` |
| `gradio_video.py` | Upload or RTSP, live preview, writes `./output/fire_tracking_[timestamp].mp4` |
| `app.py` | Image HTTP API on port `8765` |

`FireDINOTrackerGradio.vlm_interval` is `3.0` seconds. The Gradio page text still says 5 seconds; the class attribute is what the loop uses.

Overlay colors in the Gradio app:

- Red box: DINO patch tracking
- Blue box: BotSort track id
- Yellow text: still searching for fire

## Layout

```
app.py                  FastAPI image detector
gradio_video.py         Gradio video tracker (port 9007)
start.sh                llama-server, then Gradio
Dockerfile              llama.cpp CUDA image plus this app
requirements.txt
environment.yml
configs/config.yaml
src/detector.py
src/dino.py
src/frame_gather.py
utils/config.py
dinov3-convnext-tiny-pretrain-lvd1689m/   config only; weight file is not in git
notebook_finetuning/qwen3_VL_fire_fine_tuning_plus_dark.ipynb
```

`weights/` is not in the repository. `start.sh` expects these files before the container or local llama-server can start:

- `weights/larshiakzemil_unsloth_finetune_qwen3-vl-8b-instruct.Q8_0.gguf`
- `weights/larshiakzemil_unsloth_finetune_qwen3-vl-8b-thinking.BF16-mmproj.gguf`

The DINOv3 folder in git has `config.json` and the preprocessor config. The ConvNeXt weight file is not committed, so tracking cannot load until that file is placed in `dinov3-convnext-tiny-pretrain-lvd1689m/`.

## Requirements

- NVIDIA GPU for the llama.cpp server and for BotSort (`device="cuda"`)
- Docker with GPU support if you use the image
- Python packages in `requirements.txt` (`gradio`, `fastapi`, `uvicorn`, `boxmot`, `transformers`, `opencv`)

## Docker

`Dockerfile` copies `weights/*.gguf` at build time, so those files must exist first.

```bash
docker build -t fire-detection-system .
docker run --rm -it --gpus all -p 8033:8033 -p 9007:9007 fire-detection-system
```

`start.sh` starts `./llama-server` on `0.0.0.0:8033`, waits 5 seconds, then runs `python3 gradio_video.py`. The FastAPI process in `start.sh` is commented out.

- Gradio: `http://localhost:9007`
- VLM: `http://localhost:8033/v1/chat/completions`

## Run without Docker

1. Start a llama.cpp server that serves the same GGUF and mmproj on port `8033`.
2. Put the DINOv3 weights next to `config.json`.
3. Install dependencies and start Gradio from the repo root:

```bash
pip install -r requirements.txt
pip install gradio fastapi uvicorn
python3 gradio_video.py
```

Image API, pointed at the same VLM:

```bash
export VLM_SERVER_URL=http://127.0.0.1:8033/v1/chat/completions
python3 app.py
```

`app.py` listens on `0.0.0.0:8765`.

```bash
curl -F "file=@frame.jpg" http://127.0.0.1:8765/detect
curl -F "file=@frame.jpg" http://127.0.0.1:8765/detect_and_draw -o result.jpg
```

`/detect` returns `{"detections": [{"label": "fire", "bbox": [x1, y1, x2, y2]}]}`. `/detect_and_draw` returns a JPEG.

## Configuration

`configs/config.yaml` is what `FrameSynchronizer` reads. The Gradio app rewrites `sources[0].url` when you upload a file or set an RTSP URL.

Keys the gatherer uses:

- `sources`: list of `{id, url}` (file path or `rtsp://`)
- `max_failures`, `disconnect_timeout`, `buffer_size`, `log_level`
- `dashboard_alarm`: optional HTTP URL notified when a source drops
- `output_dir`: directory for tiled snapshots when it is a non-empty path. `false` turns saving off

`endpoint`, `access_key`, `secret_key`, `bucket`, Milvus, and the websocket fields are in the YAML and are not read by this tracker.

## Notes

- The default prompt sent to the VLM is `detect fire bbox`. Classes kept by the parser are `fire`, `smoke`, and `other`.
- Only `fire` boxes are passed to BotSort. The largest fire box seeds DINO.
- Processed file video is written under `./output/`. Uploads land in `./uploaded_videos/`.
- This is a research demo. Do not use it as the only fire alarm.
