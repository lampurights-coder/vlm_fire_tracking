FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# Copy your application code
COPY . /app

# Create a directory for weights and copy them (adjust if needed)
RUN mkdir -p /app/weights
COPY weights/*.gguf /app/weights/

# Expose your API port
EXPOSE 8765 8033 9007 9007

# Internal communication
ENV VLM_SERVER_URL=http://localhost:8033/v1/chat/completions

RUN pip install uvicorn fastapi gradio

# Run the startup script
ENTRYPOINT []
CMD ["/bin/bash", "./start.sh"] 


# docker run --rm -it   --gpus all   -p 8033:8033   -p 9007:9007   llama-api:latest
