import cv2
import time
import threading
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import gradio as gr
from src.frame_gather import FrameSynchronizer
from src.detector import detector, center_of_biggest_bbox
from src.dino import DINOv3PatchSimilarity
from boxmot import BotSort

class FireDINOTrackerGradio:
    def __init__(
        self,
        config_path="./configs/config.yaml",
        server_url="http://127.0.0.1:8033/v1/chat/completions",
        fps=20,
    ):
        self.config_path = config_path
        self.server_url = server_url
        self.fps = fps
        self.detect_future = None
        self.detect_frame = None
        self.executor = None  # Initialize as None, create when needed
        self.sim = DINOv3PatchSimilarity("./dinov3-convnext-tiny-pretrain-lvd1689m")
        self.ref_frame_path = None
        self.ref_center = None
        self.lock = threading.Lock()
        self.tracker = BotSort(
            reid_weights="",
            device="cuda",
            fp16=False,
            with_reid=False,
            half=True
        )
        self.last_submit_time = 0
        self.vlm_interval = 3.0
        self.stop_processing = False
        self.video_writer = None

    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def update_config_video_path(self, video_path):
        """Update the video path in config file"""
        config = self.load_config()
       
        # Update the first source with new video path
        if config.get('sources') and len(config['sources']) > 0:
            config['sources'][0]['url'] = video_path
        else:
            config['sources'] = [{'id': 11, 'url': video_path}]
       
        # Write back to config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
       
        return config

    def get_video_path_from_config(self):
        """Get video path from config file"""
        config = self.load_config()
        if config.get('sources') and len(config['sources']) > 0:
            return config['sources'][0].get('url')
        return None

    def _submit_detection(self, frame):
        ts = int(time.time() * 1000)
        img_path = f"/tmp/detect_{ts}.jpg"
        cv2.imwrite(img_path, frame)
        self.detect_frame = frame.copy()
        return self.executor.submit(detector, Path(img_path), self.server_url)

    def _handle_detection_result(self):
        if not (self.detect_future and self.detect_future.done()):
            return
       
        try:
            labels, boxes = self.detect_future.result()
            fire_boxes = [b for l, b in zip(labels, boxes) if l == "fire"]
           
            if not fire_boxes:
                print("[DETECT] no fire")
                return
           
            dets = np.array([[*b, 1.0, 0] for b in fire_boxes], dtype=np.float32)
            self.tracker.update(dets, self.detect_frame)
           
            center = center_of_biggest_bbox(fire_boxes)
            if not center:
                return
           
            x, y = center
            print(f"[DETECT] fire center: ({x},{y})")
           
            ts = int(time.time() * 1000)
            ref_path = f"/tmp/fire_ref_{ts}.jpg"
            cv2.imwrite(ref_path, self.detect_frame)
           
            with self.lock:
                self.ref_frame_path = ref_path
                self.ref_center = (x, y)
        finally:
            self.detect_future = None
            self.detect_frame = None

    def _init_video_writer(self, frame, output_path):
        if self.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (w, h)
            )
            print(f"[INFO] Recording to {output_path}")

    def process_frame(self, frame):
        """Process a single frame with fire detection and tracking"""
        current_time = time.time()
       
        # Submit detection periodically
        if self.detect_future is None and current_time - self.last_submit_time > self.vlm_interval:
            self.detect_future = self._submit_detection(frame)
            self.last_submit_time = current_time
       
        self._handle_detection_result()
       
        # DINO patch similarity tracking
        with self.lock:
            ref_path = self.ref_frame_path
            ref_center = self.ref_center
       
        if ref_path and ref_center:
            cur_path = "/tmp/current.jpg"
            cv2.imwrite(cur_path, frame)
           
            try:
                src_img = Image.open(ref_path)
                tgt_img = Image.open(cur_path)
                self.sim.process_images(src_img, tgt_img)
                x, y = ref_center
                patches, bbox = self.sim.select_patch(x=x, y=y)
               
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, "FIRE (DINO)", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    )
            except Exception as e:
                print("[DINO ERROR]", e)
       
        # BotSort tracking
        empty_dets = np.empty((0, 6), dtype=np.float32)
        tracks = self.tracker.update(empty_dets, frame)
        print(tracks)
       
        for trk in tracks:
            x1, y1, x2, y2, trk_id, conf, cls_, *_ = trk
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame, f"FIRE (BOT-SORT ID:{int(trk_id)})",
                (x1, max(20, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2,
            )
       
        # Status text
        if not ref_path and len(tracks) == 0:
            cv2.putText(
                frame, "DETECTING FIRE...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )
       
        return frame

    def process_video_stream(self, video_path=None):
        """Generator that yields processed frames in real-time using FrameSynchronizer"""
        # Create a new executor for this processing session
        self.executor = ThreadPoolExecutor(max_workers=1)
       
        # If video_path provided, update config
        if video_path:
            self.update_config_video_path(video_path)
            current_video = video_path
        else:
            # Load from config
            current_video = self.get_video_path_from_config()
       
        if not current_video:
            yield None, "❌ Video source not specified!"
            return
       
        # Check if it's a valid file or RTSP URL
        is_rtsp = current_video.startswith('rtsp://')
        if not is_rtsp and not os.path.exists(current_video):
            yield None, "❌ Video file not found!"
            return
       
        # Reset tracker state
        self.stop_processing = False
        self.ref_frame_path = None
        self.ref_center = None
        self.tracker = BotSort(
            reid_weights="",
            device="cuda",
            fp16=False,
            with_reid=False,
            half=True
        )
       
        # Create output video path
        output_path = f"./output/fire_tracking_{int(time.time())}.mp4"
        os.makedirs(Path(output_path).parent, exist_ok=True)
       
        # Temporarily open video to get metadata
        temp_cap = cv2.VideoCapture(current_video)
        if not temp_cap.isOpened():
            yield None, "❌ Failed to open video source!"
            return
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_rtsp else 0
        fps = temp_cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.fps = fps
        temp_cap.release()
       
        # Start FrameSynchronizer
        sync = FrameSynchronizer(self.config_path)
        sync.start()
       
        frame_count = 0
       
        try:
            while True:
                if self.stop_processing:
                    yield None, "⏹️ Processing stopped by user"
                    break
               
                data = sync.get_synchronized()
                if not data:
                    time.sleep(0.01)
                    continue
               
                frames, _ = data
                if not frames:
                    break
               
                frame, cam = frames[0]
               
                # Initialize video writer on first frame
                self._init_video_writer(frame, output_path)
               
                # Process frame
                processed_frame = self.process_frame(frame.copy())
               
                # Write to output video
                self.video_writer.write(processed_frame)
               
                # Convert BGR to RGB for Gradio
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
               
                frame_count += 1
                if total_frames > 0:
                    status = f"🔥 Processing: {frame_count}/{total_frames} frames ({(frame_count/total_frames*100):.1f}%)"
                else:
                    status = f"🔥 Processing live stream: {frame_count} frames"
               
                # Yield frame for real-time display
                yield processed_frame_rgb, status
               
                # Small delay to prevent overwhelming the interface
                time.sleep(0.01)
       
        finally:
            sync.stop()
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
           
            yield None, f"✅ Complete! Processed {frame_count} frames. Saved to: {output_path}"

    def stop(self):
        """Stop the processing"""
        self.stop_processing = True
        return "⏹️ Stopping..."

    def upload_video_to_config(self, video_file):
        """Handle video upload and update config"""
        if video_file is None:
            return "❌ No video selected", None
       
        # Copy uploaded file to a permanent location
        video_dir = "./uploaded_videos"
        os.makedirs(video_dir, exist_ok=True)
       
        file_name = Path(video_file.name).name
        dest_path = os.path.join(video_dir, f"{int(time.time())}_{file_name}")
        shutil.copy(video_file.name, dest_path)
       
        # Update config
        self.update_config_video_path(dest_path)
       
        return f"✅ Video uploaded and config updated!\nPath: {dest_path}", dest_path

    def set_rtsp_to_config(self, rtsp_url):
        """Handle RTSP URL input and update config"""
        if not rtsp_url or not rtsp_url.startswith('rtsp://'):
            return "❌ Invalid RTSP URL! Must start with 'rtsp://'", None
       
        # Update config
        self.update_config_video_path(rtsp_url)
       
        return f"✅ RTSP URL set and config updated!\nURL: {rtsp_url}", rtsp_url

def create_gradio_interface():
    tracker = FireDINOTrackerGradio(config_path="./configs/config.yaml")
   
    with gr.Blocks(title="Fire DINO Tracker - Real-time", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔥 Fire Detection & Tracking System (Real-time Processing)
       
        Upload a video file or enter an RTSP URL. Processing happens in **real-time** with live preview!
       
        ### Features:
        - **VLM Detection**: Vision Language Model detects fire every 5 seconds
        - **DINO Tracking**: DINOv3 patch similarity tracks fire across frames
        - **BotSort**: Multi-object tracking with consistent IDs
        """)
       
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Video Source")
               
                with gr.Tab("Upload File"):
                    video_upload = gr.File(
                        label="Upload Video File",
                        file_types=["video"],
                        type="filepath"
                    )
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        lines=2
                    )
                    upload_btn = gr.Button("📤 Upload & Update Config", variant="primary", size="lg")
               
                with gr.Tab("RTSP URL"):
                    rtsp_input = gr.Textbox(
                        label="Enter RTSP URL (e.g., rtsp://example.com/stream)",
                        placeholder="rtsp://..."
                    )
                    rtsp_status = gr.Textbox(
                        label="RTSP Status",
                        interactive=False,
                        lines=2
                    )
                    rtsp_btn = gr.Button("🔗 Set RTSP URL", variant="primary", size="lg")
               
                gr.Markdown("---")
               
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Processing", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")
               
                status_text = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    lines=2
                )
               
                gr.Markdown("""
                ### ⚙️ Current Config
                Config file: `./configs/config.yaml`
               
                Video path/URL is read from config and updated when you upload a file or set RTSP.
                """)
               
                gr.Markdown("""
                ### 🎨 Detection Legend
                - 🔴 **Red Box**: DINO patch tracking
                - 🔵 **Blue Box**: BotSort tracker with ID
                - 🟡 **Yellow Text**: Searching for fire
                """)
           
            with gr.Column(scale=2):
                gr.Markdown("### 🎥 Live Processing View")
               
                live_output = gr.Image(
                    label="Real-time Fire Detection",
                    type="numpy",
                    height=600,
                    show_label=True
                )
               
                gr.Markdown("""
                ### 💡 How it works:
                1. **Upload** a video file or **enter RTSP URL** (updates config automatically)
                2. Click **Start Processing** to begin real-time detection
                3. Watch live as each frame is processed
                4. Fire detection runs every 5 seconds
                5. Processed video is saved to `./output/` directory (for files; live streams are recorded until stopped)
                6. Use **Stop** button to cancel processing anytime
                """)
       
        # Hidden state for video path
        video_path_state = gr.State(value=None)
       
        # Event handlers
        def upload_and_update(video_file):
            status, path = tracker.upload_video_to_config(video_file)
            return status, path
       
        upload_btn.click(
            fn=upload_and_update,
            inputs=[video_upload],
            outputs=[upload_status, video_path_state]
        )
       
        def set_rtsp_and_update(rtsp_url):
            status, path = tracker.set_rtsp_to_config(rtsp_url)
            return status, path
       
        rtsp_btn.click(
            fn=set_rtsp_and_update,
            inputs=[rtsp_input],
            outputs=[rtsp_status, video_path_state]
        )
       
        # Start processing - uses generator for real-time updates
        start_btn.click(
            fn=tracker.process_video_stream,
            inputs=[video_path_state],
            outputs=[live_output, status_text]
        )
       
        stop_btn.click(
            fn=tracker.stop,
            outputs=[status_text]
        )
       
        gr.Markdown("""
        ---
        ### 📝 Notes:
        - Processing shows frames in **real-time** as they're processed (not after completion)
        - Original video path in config: will be used if no video is uploaded or RTSP set
        - Output videos are saved to: `./output/fire_tracking_[timestamp].mp4`
        - Detection interval: 5 seconds (configurable in code)
        - For RTSP live streams, processing continues until stopped, and total frames are not known in advance
        """)
   
    return demo

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./configs", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./uploaded_videos", exist_ok=True)
   
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=9007,
        share=False
    )

# ./llama.cpp/build/bin/llama-server   -hf larshiakzemil/unsloth_finetune:Q8_0   --n-gpu-layers 99   --ctx-size 65536   --batch-size 1024   --ubatch-size 256   --image-min-tokens 1024   --top-p 0.8   --top-k 20   --temp 0.7   --presence-penalty 1.2   --flash-attn off  --jinja   --host 127.0.0.1   --port 8033
#  docker run -d --gpus all   -p 8033:8033   -v /home/ai/.cache/llama.cpp:/root/.cache/llama.cpp   ghcr.io/ggml-org/llama.cpp:server-cuda   -hf larshiakzemil/unsloth_finetune:Q8_0   --n-gpu-layers 99   --ctx-size 65536   --batch-size 1024   --ubatch-size 256   --image-min-tokens 1024   --top-p 0.8   --top-k 20   --temp 0.7   --presence-penalty 1.2   --flash-attn on   --jinja   --host 0.0.0.0   --port 8033