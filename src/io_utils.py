# -*- coding: utf-8 -*-
"""Đọc test (.json hoặc .csv) + ghi pred.csv. Nhãn động A..K theo số lựa chọn."""
import os
import csv
import json
import string

LETTERS = string.ascii_uppercase  # đề dùng tới K (11 lựa chọn)


def make_labels(n):
    """['A','B',...] đúng n phần tử."""
    return [LETTERS[i] for i in range(n)]


def format_options(choices):
    """Đánh nhãn từng lựa chọn: 'A. ...\\nB. ...'."""
    return "\n".join(f"{l}. {c}" for l, c in zip(make_labels(len(choices)), choices))


def load_test(path):
    """Đọc bộ test. Hỗ trợ:
      - .json: list[{qid, question, choices}]
      - .csv : cột qid,question,choice_a..choice_k HOẶC qid,question,choices(json)
    Trả về list dict chuẩn hoá: {qid, question, choices:[...]}.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # đã đúng schema {qid, question, choices}
        return [{"qid": x["qid"], "question": x["question"], "choices": list(x["choices"])}
                for x in data]

    # CSV: thử vài layout phổ biến
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        choice_cols = [c for c in cols if c.lower().startswith("choice")
                       or c.lower() in [l.lower() for l in LETTERS]]
        for r in reader:
            qid = r.get("qid") or r.get("id")
            question = r.get("question") or r.get("q") or ""
            if "choices" in r and r["choices"]:
                try:
                    choices = json.loads(r["choices"])
                except (ValueError, TypeError):
                    choices = [c for c in r["choices"].split("|") if c]
            else:
                choices = [r[c] for c in choice_cols if r.get(c) not in (None, "")]
            rows.append({"qid": qid, "question": question, "choices": list(choices)})
    return rows


def write_preds(path, order, preds):
    """Ghi qid,answer theo đúng thứ tự order. Chỉ gọi khi đã đủ đáp án."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer"])
        for q in order:
            w.writerow([q, preds[q]])
