# -*- coding: utf-8 -*-
"""ENTRY-POINT bản nộp HackAIthon Bảng C (theo Submission Guideline của BTC).

Cơ chế End-to-End:
  1. Đọc /code/private_test.json (BTC mount vào khi chấm).
  2. Chạy pipeline giải từng câu, ĐO thời gian TỪNG câu (vòng for) theo yêu cầu BTC.
  3. Ghi 2 file ra /code:
       - submission.csv       : qid,answer
       - submission_time.csv  : qid,answer,time

Có thể override đường dẫn qua env (mặc định khớp máy chấm BTC):
  TEST_PATH (mặc định /code/private_test.json)
  OUT_DIR   (mặc định /code)
"""
import os
import sys
import csv
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from src.io_utils import load_test, LETTERS
from src.engine import Engine
from src import pipeline
from src.io_utils import make_labels

# ---- Đường dẫn theo guideline BTC (mount vào /code) ----
HERE = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.environ.get("TEST_PATH", "/code/private_test.json")
OUT_DIR = os.environ.get("OUT_DIR", "/code")


def find_test_file():
    """Ưu tiên /code/private_test.json (chuẩn BTC). Fallback: tìm json/csv quanh đó."""
    if os.path.exists(TEST_PATH):
        return TEST_PATH
    # phòng khi BTC mount tên/đường dẫn khác: quét /code rồi thư mục hiện tại
    for d in ("/code", HERE, os.getcwd()):
        if not os.path.isdir(d):
            continue
        for name in ("private_test.json", "private_test.csv",
                     "public_test.json", "public_test.csv"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".json", ".csv")) and "test" in fn.lower():
                return os.path.join(d, fn)
    return None


def main():
    test_path = find_test_file()
    if not test_path:
        print(f"[predict] KHÔNG tìm thấy file test (mong đợi {TEST_PATH})", file=sys.stderr)
        sys.exit(2)

    # Output ghi NGAY trong /code (nơi BTC mount data vào): data đi vào /code,
    # chạy xong 2 file CSV tự nằm trong /code (đồng nghĩa hiện ra thư mục host BTC mount).
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    sub_path = os.path.join(out_dir, "submission.csv")
    time_path = os.path.join(out_dir, "submission_time.csv")

    data = load_test(test_path)
    print(f"[predict] test={test_path} | {len(data)} câu | quant={config.DEFAULT_QUANT} "
          f"n_ctx={config.DEFAULT_NCTX} mode={config.PIPELINE_MODE}", flush=True)

    model_file = config.ensure_model()
    ngl = config.auto_n_gpu_layers()
    print(f"[predict] model={os.path.basename(model_file)} | n_gpu_layers={ngl} "
          f"(gpu={config.has_gpu()}, free={config.gpu_free_mb()}MiB)", flush=True)

    eng = Engine(model_file, n_gpu_layers=ngl, n_ctx=config.DEFAULT_NCTX)

    results = []  # (qid, answer, time_infer_sample)
    t_all = time.time()
    for i, item in enumerate(data, 1):
        # ĐO THỜI GIAN TỪNG CÂU theo guideline BTC (vòng for, end-start mỗi sample)
        start = time.time()
        res = pipeline.solve_one(eng, item, do_self_check=True, mode=config.PIPELINE_MODE)
        end = time.time()

        ans = res["answer"]
        # đảm bảo nhãn hợp lệ trong phạm vi A..n của câu này
        n = len(item["choices"])
        if ans not in make_labels(n):
            ans = "A"
        results.append((item["qid"], ans, round(end - start, 4)))

        if i % 25 == 0 or i == len(data):
            el = time.time() - t_all
            print(f"[predict] {i}/{len(data)} | {el:.0f}s ({el/i:.2f}s/câu)", flush=True)

    # ---- Ghi submission.csv (qid,answer) ----
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer"])
        for qid, ans, _ in results:
            w.writerow([qid, ans])

    # ---- Ghi submission_time.csv (qid,answer,time) ----
    with open(time_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "time"])
        for qid, ans, t in results:
            w.writerow([qid, ans, t])

    total = time.time() - t_all
    print(f"[predict] XONG ({len(results)} câu, tổng {total:.0f}s) -> "
          f"{sub_path} + {time_path}", flush=True)


if __name__ == "__main__":
    main()
