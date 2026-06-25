# -*- coding: utf-8 -*-
"""Preload các DLL CUDA + llama_cpp đúng thứ tự TRƯỚC khi import llama_cpp.

Windows: wheel cu124 KHÔNG kèm CUDA runtime, và Python không tự tìm thấy
ggml-*.dll/llama.dll cạnh nhau -> FileNotFoundError. Phải:
  1. add_dll_directory cho nvidia/*/bin + llama_cpp/lib
  2. ctypes.CDLL preload ggml-*.dll theo thứ tự phụ thuộc, rồi llama.dll
Trên Linux (Docker) hàm này là no-op: import llama_cpp chạy thẳng.
"""
import os
import sys
import glob
import ctypes


def preload():
    """Gọi 1 lần trước khi import llama_cpp. An toàn khi gọi lặp."""
    if sys.platform != "win32":
        return  # Linux/Docker: linker tự lo, bỏ qua

    # site-packages của interpreter đang chạy (venv nào cũng đúng)
    import sysconfig
    base = sysconfig.get_paths()["purelib"]
    libdir = os.path.join(base, "llama_cpp", "lib")

    for d in glob.glob(os.path.join(base, "nvidia", "*", "bin")) + [libdir]:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass

    # Thứ tự phụ thuộc: base -> cpu -> cuda -> ggml -> llama
    for dll in ["ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll"]:
        p = os.path.join(libdir, dll)
        if os.path.exists(p):
            try:
                ctypes.CDLL(p)
            except OSError:
                # ggml-cuda.dll có thể vắng nếu build CPU-only -> bỏ qua, llama.dll vẫn chạy CPU
                pass
