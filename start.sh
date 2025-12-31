#!/bin/bash

# Start the llama-server in the background with your original command args
./llama-server \
  --model ./weights/larshiakzemil_unsloth_finetune_qwen3-vl-8b-instruct.Q8_0.gguf \
  --mmproj ./weights/larshiakzemil_unsloth_finetune_qwen3-vl-8b-thinking.BF16-mmproj.gguf  \
  --n-gpu-layers 99 \
  --jinja \
  --top-p 0.8 \
  --top-k 20 \
  --temp 0.7 \
  --min-p 0.0 \
  --flash-attn on \
  --presence-penalty 1.5 \
  --ctx-size 65536 \
  --host 0.0.0.0 \
  --port 8033 

# Wait briefly for the server to start (adjust if needed)
# sleep 5

# # Start your API
# exec uvicorn app:app --host 0.0.0.0 --port 8765

sleep 5
# Start your Gradio
python3 gradio_video.py