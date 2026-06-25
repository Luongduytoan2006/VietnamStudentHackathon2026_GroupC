# -*- coding: utf-8 -*-
"""ENTRYPOINT bản nộp — đạt 90.7% public (xem PIPELINE.md).

BTC mount /data (chứa private_test.csv|json) và /output. Chạy:
    python main.py
Local thử:
    DATA_DIR=. OUTPUT_DIR=. python main.py --test ../data/public-test_1780368312.json --out pred.csv
"""
import os
import sys
import json
import time
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import config
from src.io_utils import load_test, write_preds, LETTERS
from src.engine import Engine
from src import pipeline


def find_test_file():
    """Tìm file test ở DATA_DIR (ưu tiên private_test theo thể lệ; nhận cả .csv/.json)."""
    for name in ("private_test.csv", "public_test.csv", "private_test.json", "public_test.json"):
        p = os.path.join(config.DATA_DIR, name)
        if os.path.exists(p):
            return p
    if os.path.isdir(config.DATA_DIR):
        for fn in sorted(os.listdir(config.DATA_DIR)):
            if fn.lower().endswith((".json", ".csv")):
                return os.path.join(config.DATA_DIR, fn)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default=config.DEFAULT_QUANT)
    ap.add_argument("--mode", default=config.PIPELINE_MODE, choices=["hybrid", "full"])
    ap.add_argument("--test", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint resume (json)")
    args = ap.parse_args()

    test_path = args.test or find_test_file()
    if not test_path:
        print(f"[main] KHÔNG tìm thấy file test trong {config.DATA_DIR}", file=sys.stderr)
        sys.exit(2)
    out_path = args.out or os.path.join(config.OUTPUT_DIR, "pred.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    data = load_test(test_path)
    order = [x["qid"] for x in data]
    print(f"[main] test={test_path} | {len(data)} câu | mode={args.mode} quant={args.quant} "
          f"n_ctx={config.DEFAULT_NCTX} calc_maxtok={config.CALC_MAXTOK}", flush=True)

    model_file = config.ensure_model(args.quant)
    ngl = config.auto_n_gpu_layers()
    print(f"[main] model={os.path.basename(model_file)} | n_gpu_layers={ngl} "
          f"(gpu={config.has_gpu()}, free={config.gpu_free_mb()}MiB)", flush=True)

    t0 = time.time()
    eng = Engine(model_file, n_gpu_layers=ngl, n_ctx=config.DEFAULT_NCTX)
    print(f"[main] load {time.time()-t0:.1f}s", flush=True)

    ckpt = args.ckpt
    preds = json.load(open(ckpt, encoding="utf-8")) if (ckpt and os.path.exists(ckpt)) else {}
    todo = [x for x in data if x["qid"] not in preds]

    t0 = time.time()
    for i, item in enumerate(todo, 1):
        res = pipeline.solve_one(eng, item, do_self_check=True, mode=args.mode)
        preds[item["qid"]] = res["answer"]
        if ckpt and (i % 20 == 0 or i == len(todo)):
            json.dump(preds, open(ckpt, "w", encoding="utf-8"), ensure_ascii=False)
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[main] {len(preds)}/{len(data)} | {el:.0f}s ({el/i:.2f}s/câu)", flush=True)

    # validate: đủ câu + nhãn hợp lệ, lỗi -> ép A an toàn
    nopt = {x["qid"]: len(x["choices"]) for x in data}
    for q in order:
        if q not in preds:
            preds[q] = "A"
    bad = [q for q in order if preds[q] not in LETTERS or LETTERS.index(preds[q]) >= nopt[q]]
    for q in bad:
        preds[q] = "A"
    if bad:
        print(f"[main] đã ép A cho {len(bad)} nhãn ngoài phạm vi", file=sys.stderr)

    write_preds(out_path, order, preds)
    print(f"[main] XONG -> {out_path} ({len(order)} dòng, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
