# -*- coding: utf-8 -*-
"""Cấu hình CHUẨN cho bản nộp (đạt 90.7% public). 1 nguồn sự thật duy nhất.

Mọi tham số có thể override bằng biến môi trường (env) khi chạy Docker.
"""
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---- Model giải (Qwen3.5-9B GGUF, unsloth) ----
HF_REPO = "unsloth/Qwen3.5-9B-GGUF"
QUANT_FILE = {
    "Q4": "Qwen3.5-9B-Q4_K_M.gguf",
    "Q5": "Qwen3.5-9B-Q5_K_M.gguf",
    "Q6": "Qwen3.5-9B-Q6_K.gguf",
    "Q8": "Qwen3.5-9B-Q8_0.gguf",
}

# ============================================================================
# BỘ CẤU HÌNH CHUẨN — 90.7% + HARM-GUARD = 91.36% trên public 463 câu (xem PIPELINE.md)
# CHỈ 1 model: Qwen3.5-9B GGUF. KHÔNG dùng embedding/RAG (đã bỏ bge-m3).
# ============================================================================
DEFAULT_QUANT = os.environ.get("QUANT", "Q5")            # Q5_K_M ~6.6GB: cân bằng acc/tốc độ tốt nhất
DEFAULT_NCTX = int(os.environ.get("N_CTX", "6144"))      # đủ chứa câu hỏi dài + reasoning toán ~2000 token
CALC_MAXTOK = int(os.environ.get("CALC_MAXTOK", "2000"))  # CHÌA KHÓA: cho toán viết HẾT bài giải (cụt = sai)
CALC_VOTE = int(os.environ.get("CALC_VOTE", "1"))        # 1 = greedy (vote>1 phản tác dụng, đã chứng minh)
PIPELINE_MODE = os.environ.get("MODE", "hybrid")         # hybrid: calc+safety dùng solver, còn lại grammar đọc thẳng
HARM_GUARD = os.environ.get("HARM_GUARD", "1") == "1"    # GUARD câu xui phá hoại + option từ chối → refusal (+3, phá 0)

# ---- Mount theo thể lệ BTC (Docker đọc /data, ghi /output) ----
DATA_DIR = os.environ.get("DATA_DIR", "/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")


def model_path(quant=None):
    quant = quant or DEFAULT_QUANT
    return os.path.join(MODELS_DIR, QUANT_FILE[quant])


def ensure_model(quant=None):
    """Đảm bảo có file model; thiếu thì tải từ HF. Tìm cả ../final_code/models để khỏi tải lại."""
    quant = quant or DEFAULT_QUANT
    fname = QUANT_FILE[quant]
    p = os.path.join(MODELS_DIR, fname)
    if os.path.exists(p) and os.path.getsize(p) > 1_000_000_000:
        return p
    # thử mượn từ final_code/models (phòng thí nghiệm) để khỏi tải lại
    lab = os.path.join(os.path.dirname(HERE), "final_code", "models", fname)
    if os.path.exists(lab) and os.path.getsize(lab) > 1_000_000_000:
        return lab
    from huggingface_hub import hf_hub_download
    print(f"[config] Tải {fname} từ {HF_REPO} ...", flush=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=MODELS_DIR)


def has_gpu():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def gpu_free_mb():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        r = subprocess.run([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def auto_n_gpu_layers():
    """Auto-detect: có GPU đủ VRAM -> full offload (-1); ít VRAM -> offload 1 phần; không GPU -> CPU (0)."""
    if not has_gpu():
        return 0
    free = gpu_free_mb()
    if free >= 6500:
        return -1       # đủ cho Q5 full GPU
    if free >= 4500:
        return 28       # offload phần lớn, tràn CPU chút
    if free >= 2500:
        return 16
    return 0            # CPU-only fallback
