# HackAIthon Bảng C — Qwen3.5-9B MCQ pipeline
# Base: CUDA 12.4 runtime (khớp wheel cu124). Máy chấm cần GPU; có CPU fallback.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DATA_DIR=/data \
    OUTPUT_DIR=/output

# Python + build tools (llama-cpp-python build từ source với CUDA).
# ninja-build BẮT BUỘC: scikit-build-core dùng ninja làm generator, thiếu -> build fail.
# Dùng python3 mặc định (3.10 của ubuntu22.04) cho nhất quán build lẫn runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài deps. Build llama-cpp-python với CUDA (GGML_CUDA=on).
RUN pip3 install --no-cache-dir huggingface-hub hf-transfer numpy && \
    CMAKE_ARGS="-DGGML_CUDA=on" pip3 install --no-cache-dir llama-cpp-python==0.3.30

# Copy code (KHÔNG copy /data, /output — BTC mount lúc chạy)
COPY config.py main.py ./
COPY src/ ./src/

# NHÚNG model vào image (self-contained): chỉ Qwen3.5-9B-Q5 (~6.6GB). KHÔNG cần embedding (bỏ RAG).
# BTC pull về là chạy NGAY, không cần mạng, không cần mount model.
COPY models/ ./models/

# Entrypoint: đọc /data, ghi /output/pred.csv
ENTRYPOINT ["python3", "main.py"]
