# -*- coding: utf-8 -*-
"""Trích chữ cái đáp án A..n từ output model. Ép thuộc phạm vi hợp lệ.

Fix bug notebook cũ: KHÔNG hardcode A-D — dùng nhãn động theo số lựa chọn n,
nếu không model gán E..K của câu 10-lc sẽ bị cắt mất.
"""
import re
from .io_utils import make_labels


def extract_answer(text, n, default="A"):
    """Lấy đáp án từ text. n = số lựa chọn (giới hạn A..LETTERS[n-1]).
    Ưu tiên: 'Đáp án: X' -> chữ cái đứng-một-mình cuối -> chữ cái bất kỳ cuối.
    """
    valid = set(make_labels(n))
    upper = chr(ord("A") + n - 1)  # chữ cái CUỐI của phạm vi, vd n=4 -> 'D'

    # bỏ phần <think>...</think> nếu model lỡ xả reasoning
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # nếu có </think> mở mà chưa đóng -> lấy phần sau cùng
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 1) 'Đáp án: X' / 'Đáp án là X' (mạnh nhất)
    m = re.findall(rf"[Đđ]áp\s*án[^A-{upper}]{{0,6}}([A-{upper}])", text)
    if m and m[-1] in valid:
        return m[-1]

    # 2) chữ cái hợp lệ đứng một mình (\b...\b) — lấy cái CUỐI
    for c in reversed(re.findall(rf"\b([A-{upper}])\b", text)):
        if c in valid:
            return c

    # 3) bất kỳ chữ cái hợp lệ nào — lấy cái CUỐI
    for c in reversed(re.findall(rf"([A-{upper}])", text)):
        if c in valid:
            return c

    return default
