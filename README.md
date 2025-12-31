🔥 Fire Detection & Tracking System
![Fire Detection Logo](

A real-time fire detection and tracking system combining Vision Language Model (VLM) for detection with DINOv3 patch similarity and BotSort for tracking. Processes video streams in real-time through a Gradio web interface.

🚀 Features
Multi-Model Detection: Fine-tuned Qwen3-VL for precise fire detection

Intelligent Tracking: DINOv3 patch similarity + BotSort multi-object tracking

Real-time Processing: Live frame-by-frame processing with visual feedback

Multiple Input Sources: Video files + RTSP streams

Web Interface: User-friendly Gradio UI

Dockerized: Easy GPU deployment

🏗️ Architecture
text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Gradio Web UI  │◄──►│  Tracking Engine │◄──►│ VLM Detection   │
│   (Port 9007)   │    │ (DINOv3+BotSort) │    │ Server (Port    │
└─────────────────┘    └──────────────────┘    │ 8033)           │
                                               └─────────────────┘
📋 Prerequisites
NVIDIA GPU with CUDA 12+

Docker & Docker Compose

8GB+ GPU memory

16GB+ system RAM

🛠️ Quick Start
1. Clone & Prepare
bash
git clone https://github.com/lampurights-coder/vlm_fire_tracking.git
cd vlm_fire_tracking

# Copy model weights to weights/
# - larshiakzemil_unsloth_finetune_qwen3-vl-8b-instruct.Q8_0.gguf
# - larshiakzemil_unsloth_finetune_qwen3-vl-8b-thinking.BF16-mmproj.gguf
2. Build & Run
bash
docker build -t fire-detection-system .
docker run --rm -it --gpus all \
  -p 8033:8033 -p 9007:9007 \
  -v $(pwd)/output:/app/output \
  fire-detection-system
3. Access Interfaces
🎥 Gradio UI: http://localhost:9007

🔍 VLM API: http://localhost:8033

🌐 Web Interface
Tab	Action	Example
📁 Upload File	Upload MP4/AVI/MOV	fire_test.mp4
🌐 RTSP URL	Enter stream URL	rtsp://192.168.1.100/stream
▶️ Start	Begin real-time processing	Live detection + tracking
Detection Legend:

🟥 Red Box: DINOv3 patch tracking

🔵 Blue Box + ID: BotSort tracker

🟢 Green Box: VLM fire detection

🟡 Yellow Text: "Searching for fire..."

![Demo Screenshot](Screenshot_from_2025-12-31 ⚙️ Configuration

config.yaml:

text
sources:
  - id: 11
    url: "rtsp://camera-ip/stream"  # or local video path
    fps: 30

vlm:
  server_url: "http://localhost:8033"
  confidence_threshold: 0.6
  detection_interval: 5  # seconds
🐛 Troubleshooting
Issue	Solution
GPU OOM	docker run ... -e MAX_BATCH_SIZE=1
VLM not responding	curl http://localhost:8033/health
No detections	Lower confidence_threshold to 0.4
Slow FPS	Use smaller model or reduce top_k_patches
View logs:

bash
docker logs <container_id> -f
📈 Performance Metrics
Metric	Value
FPS	10-20 (RTX 4090)
Accuracy	~85% on test sets
GPU Memory	~6GB
Latency	<200ms/frame
🔄 Model Updates
Convert new model: llama.cpp/convert.py

Place in weights/

Update start.sh: MODEL_NAME="new-model.gguf"

Rebuild: docker build -t fire-detection-system .

📁 Project Structure
text
vlm_fire_tracking/
├── weights/                 # GGUF model files
├── configs/config.yaml      # System config
├── src/
│   ├── detector.py         # VLM detection
│   ├── dino.py            # Patch similarity
│   └── botsort_tracker.py # Multi-object tracking
├── gradio_app.py           # Web interface
├── Dockerfile
├── start.sh
└── output/                 # Processed videos
🤝 Contributing & License
text
MIT License - Fork, modify, deploy!
See CONTRIBUTING.md for details.
Credits:

Qwen3-VL - Fire detection

DINOv3 - Patch tracking

BoxMOT - BotSort

⚠️ Production Note: Always use multiple detection systems and human verification for critical fire safety applications.
