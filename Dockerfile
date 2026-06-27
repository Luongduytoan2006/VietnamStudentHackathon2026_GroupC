# HackAIthon Bảng C — Qwen3.5-4B MCQ pipeline (theo Submission Guideline BTC)
# ------------------------------------------------------------
# BASE IMAGE — CUDA 12.8 (máy chấm BTC dùng RTX 5060Ti / Blackwell, cần CUDA >= 12.8).
# ------------------------------------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ------------------------------------------------------------
# SYSTEM DEPENDENCIES
# ninja-build BẮT BUỘC: scikit-build-core dùng ninja làm generator để build llama-cpp-python.
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential cmake ninja-build && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------
# PROJECT SETUP
# Code + model đặt tại /app (KHÔNG phải /code) để khi BTC mount data vào /code
# — dù mount RIÊNG file (/code/private_test.json) hay mount CẢ thư mục (/code) —
# cũng KHÔNG đè mất code/model. /code chỉ dùng cho input + output.
# ------------------------------------------------------------
WORKDIR /app
RUN mkdir -p /code

# ------------------------------------------------------------
# INSTALL LIBRARIES
# Build llama-cpp-python với CUDA. CMAKE_CUDA_ARCHITECTURES gồm 89 (RTX 40xx của dev)
# và 120 (RTX 50xx Blackwell của máy chấm BTC) để image chạy GPU được trên CẢ HAI.
# ------------------------------------------------------------
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir hf-transfer && \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89;120" \
        pip3 install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# COPY SOURCE + MODEL vào /app (nhúng model: self-contained, chạy không cần mạng)
# ------------------------------------------------------------
COPY config.py predict.py inference.sh ./
COPY src/ ./src/
COPY models/ ./models/

# ------------------------------------------------------------
# EXECUTION — đọc /code/private_test.json -> /code/submission.csv + submission_time.csv
# ------------------------------------------------------------
CMD ["bash", "/app/inference.sh"]
