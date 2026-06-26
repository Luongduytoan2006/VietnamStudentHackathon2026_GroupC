# -*- coding: utf-8 -*-
"""[5a..5e] 5 solver nhánh + LLM verifier + judge.

Triết lý: LLM không chọn đáp án ngay; nó verify từng option / loại trừ / judge cuối.
Mọi solver trả result dict: {answer, confidence, candidate_scores, route, evidence?}.
"""
import re
import json
from .io_utils import make_labels, format_options
from .extract import extract_answer
# LƯU Ý: KHÔNG import rag ở đây. Bản nộp MODE=hybrid không dùng RAG/embedding (reading đi grammar
# đọc-thẳng). solve_reading chỉ dùng khi MODE=full → import rag lazy bên trong để image khỏi cần bge-m3.

# ---------- tiện ích LLM ----------

def _grammar_choice(engine, item, sys_prompt, user_prompt):
    """Ép model trả 1 ký tự hợp lệ qua grammar."""
    n = len(item["choices"])
    raw = engine.chat(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_prompt}],
        max_tokens=8, temperature=0.0, grammar=engine.grammar_for(n))
    return extract_answer(raw, n)


def _json_verify(engine, evidence, question, option_text, max_tokens=160):
    """[5c-6] LLM verify 1 option dựa evidence → (label, confidence)."""
    sys = ("Bạn là bộ kiểm chứng đáp án trắc nghiệm. Chỉ dựa vào EVIDENCE, "
           "không dùng kiến thức ngoài. Nếu evidence không đủ, trả NOT_ENOUGH_INFO.")
    user = (f"EVIDENCE:\n{evidence}\n\nQUESTION:\n{question}\n\nOPTION:\n{option_text}\n\n"
            "Đánh giá OPTION: SUPPORTED (evidence ủng hộ rõ) | CONTRADICTED (bị phủ định) | "
            "NOT_ENOUGH_INFO. Trả JSON: {\"label\":\"...\",\"confidence\":0.0-1.0}")
    raw = engine.chat([{"role": "system", "content": sys},
                       {"role": "user", "content": user}],
                      max_tokens=max_tokens, temperature=0.0)
    return _parse_verdict(raw)


def _parse_verdict(raw):
    label, conf = "NOT_ENOUGH_INFO", 0.4
    m = re.search(r"SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO", raw, re.I)
    if m:
        label = m.group(0).upper()
    mc = re.search(r'"?confidence"?\s*[:=]\s*([01](?:\.\d+)?)', raw)
    if mc:
        try:
            conf = float(mc.group(1))
        except ValueError:
            pass
    return label, conf


_SUP = {"SUPPORTED": 1.0, "NOT_ENOUGH_INFO": 0.4, "CONTRADICTED": 0.0}


# ---------- [5a] safety ----------
_REFUSAL_KW = re.compile(
    r"không thể (?:cung cấp|hỗ trợ|giúp)|không hỗ trợ|từ chối|"
    r"tuân thủ pháp luật|liên hệ cơ quan|vi phạm pháp luật|"
    r"trái pháp luật|không hợp pháp|ngừng hoạt động|chấp hành|hợp pháp",
    re.IGNORECASE,
)


def solve_safety(engine, item, feat):
    labels = make_labels(len(item["choices"]))
    scores = {}
    for lab, ch in zip(labels, item["choices"]):
        scores[lab] = 1.0 if _REFUSAL_KW.search(str(ch)) else 0.0
    if max(scores.values()) > 0:
        best = max(scores, key=scores.get)
        return {"answer": best, "confidence": 0.85, "candidate_scores": scores, "route": "safety"}
    # không rõ → để LLM grammar
    ans = _grammar_choice(engine, item, _SYS_DIRECT, _user_direct(item))
    return {"answer": ans, "confidence": 0.5, "candidate_scores": {ans: 0.5}, "route": "safety"}


# ---------- [5b] calculation ----------
_SYS_CALC = (
    "Bạn là chuyên gia Toán–Lý–Hoá–Kinh tế. Giải bài trắc nghiệm theo bước: "
    "(1) tóm tắt dữ kiện và đại lượng; (2) chọn công thức đúng; (3) TÍNH ra kết quả số "
    "trước khi nhìn kỹ các lựa chọn; (4) so kết quả với các lựa chọn, chọn cái khớp nhất. "
    "Suy luận ngắn gọn, cẩn thận số học. Kết thúc bằng đúng một dòng: 'Đáp án: X'."
)


def solve_calculation(engine, item, feat, max_tokens=420, n_vote=1):
    """CoT tính toán. n_vote=1: greedy một lần. n_vote>1: self-consistency
    (chạy CoT n_vote lần temp>0, lấy đa số) — đắt hơn nhưng ổn định hơn cho toán."""
    n = len(item["choices"])
    user = (f"Bài toán:\n{item['question']}\n\nCác lựa chọn:\n{format_options(item['choices'])}\n\n"
            f"Trả lời bằng MỘT chữ cái trong [{', '.join(make_labels(n))}].")
    msgs = [{"role": "system", "content": _SYS_CALC}, {"role": "user", "content": user}]

    if n_vote <= 1:
        raw = engine.chat(msgs, max_tokens=max_tokens, temperature=0.0, stop=["<|im_end|>"])
        ans = extract_answer(raw, n, default=None)
        if ans is None:
            ans = _grammar_choice(engine, item, _SYS_DIRECT, _user_direct(item))
            return {"answer": ans, "confidence": 0.5, "candidate_scores": {ans: 0.5},
                    "route": "calculation", "truncated": True}
        return {"answer": ans, "confidence": 0.7, "candidate_scores": {ans: 0.7}, "route": "calculation"}

    # self-consistency: lần 1 greedy (mạnh nhất) + (n_vote-1) lần temp>0
    from collections import Counter
    votes = []
    for k in range(n_vote):
        temp = 0.0 if k == 0 else 0.7
        raw = engine.chat(msgs, max_tokens=max_tokens, temperature=temp, top_p=0.95,
                          stop=["<|im_end|>"], seed=(-1 if k else None))
        a = extract_answer(raw, n, default=None)
        if a is not None:
            votes.append(a)
    if not votes:
        ans = _grammar_choice(engine, item, _SYS_DIRECT, _user_direct(item))
        return {"answer": ans, "confidence": 0.5, "candidate_scores": {ans: 0.5},
                "route": "calculation", "truncated": True}
    win, cnt = Counter(votes).most_common(1)[0]
    conf = 0.6 + 0.35 * (cnt / len(votes))
    return {"answer": win, "confidence": round(conf, 2),
            "candidate_scores": {win: conf}, "route": "calculation", "votes": votes}


# ---------- [5c] reading_context (CHỈ dùng khi MODE=full; bản nộp hybrid KHÔNG gọi hàm này) ----------
def solve_reading(engine, item, feat):
    from . import rag  # lazy import: chỉ nạp khi thật sự chạy MODE=full (cần bge-m3)
    n = len(item["choices"])
    labels = make_labels(n)
    ctx = feat["context"] or item["question"]
    mainq = feat["main_question"]
    query = mainq + " " + " ".join(map(str, item["choices"]))
    evidence_chunks, ev_norm = rag.retrieve_evidence(ctx, query, top_k=5)
    evidence = "\n---\n".join(evidence_chunks)

    scores = {}
    for lab, ch in zip(labels, item["choices"]):
        label, conf = _json_verify(engine, evidence, mainq, str(ch))
        llm_s = _SUP[label] * conf
        # lexical overlap đơn giản
        lex = _lexical_overlap(str(ch), evidence)
        scores[lab] = 0.55 * llm_s + 0.25 * lex + 0.20 * _exact_bonus(str(ch), evidence)
    best = max(scores, key=scores.get)
    srt = sorted(scores.values(), reverse=True)
    margin = (srt[0] - srt[1]) if len(srt) > 1 else 1.0
    conf = min(0.95, 0.5 + scores[best] * 0.4 + margin * 0.2)
    return {"answer": best, "confidence": conf, "candidate_scores": scores,
            "route": "reading_context", "evidence": evidence_chunks}


def _lexical_overlap(a, b):
    ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _exact_bonus(choice, evidence):
    # số/ngày/tên riêng xuất hiện nguyên văn
    toks = re.findall(r"\d[\d.,/:-]*|\b[A-ZĐ][a-zà-ỹ]+\b", choice)
    if not toks:
        return 0.0
    hit = sum(1 for t in toks if t in evidence)
    return min(1.0, hit / len(toks))


# ---------- [5d] legal_admin ----------
_SYS_LEGAL = (
    "Bạn là chuyên gia pháp luật và thủ tục hành chính Việt Nam. Với câu hỏi: "
    "(1) xác định nguyên tắc/quy định liên quan; (2) loại trừ lựa chọn sai; (3) chọn đáp án đúng. "
    "Ưu tiên đáp án đúng thủ tục, hợp pháp. KHÔNG bịa số điều luật nếu không chắc. "
    "Suy luận ngắn. Kết thúc bằng đúng một dòng: 'Đáp án: X'."
)


def solve_legal(engine, item, feat, max_tokens=260):
    n = len(item["choices"])
    user = (f"Câu hỏi:\n{item['question']}\n\nCác lựa chọn:\n{format_options(item['choices'])}\n\n"
            f"Trả lời bằng MỘT chữ cái trong [{', '.join(make_labels(n))}].")
    raw = engine.chat([{"role": "system", "content": _SYS_LEGAL},
                       {"role": "user", "content": user}],
                      max_tokens=max_tokens, temperature=0.0, stop=["<|im_end|>"])
    ans = extract_answer(raw, n, default=None)
    if ans is None:
        ans = _grammar_choice(engine, item, _SYS_DIRECT, _user_direct(item))
        return {"answer": ans, "confidence": 0.5, "candidate_scores": {ans: 0.5},
                "route": "legal_admin", "truncated": True}
    return {"answer": ans, "confidence": 0.68, "candidate_scores": {ans: 0.68}, "route": "legal_admin"}


# ---------- [5e] general_knowledge ----------
_SYS_DIRECT = (
    "Bạn là chuyên gia giải trắc nghiệm tiếng Việt (kiến thức + suy luận). "
    "Đọc kỹ câu hỏi và TẤT CẢ lựa chọn, rồi trả về DUY NHẤT một chữ cái của đáp án đúng."
)


def _user_direct(item):
    n = len(item["choices"])
    return (f"Câu hỏi:\n{item['question']}\n\nCác lựa chọn:\n{format_options(item['choices'])}\n\n"
            f"Chỉ trả lời bằng MỘT chữ cái trong [{', '.join(make_labels(n))}].")


def solve_general(engine, item, feat, vote_shuffle=True):
    """[5e] direct lần 1 → shuffle → direct lần 2 → vote. Chống bias vị trí."""
    n = len(item["choices"])
    labels = make_labels(n)
    # lần 1: thứ tự gốc
    a1 = _grammar_choice(engine, item, _SYS_DIRECT, _user_direct(item))
    votes = [a1]
    if vote_shuffle and n <= 11:
        # [5e-2] shuffle bằng hoán vị xác định (đảo ngược) để tái lập được
        perm = list(range(n))[::-1]
        shuffled = {"qid": item["qid"], "question": item["question"],
                    "choices": [item["choices"][p] for p in perm]}
        a2s = _grammar_choice(engine, shuffled, _SYS_DIRECT, _user_direct(shuffled))
        # map nhãn shuffled -> gốc
        if a2s in labels:
            orig_idx = perm[labels.index(a2s)]
            votes.append(labels[orig_idx])
    from collections import Counter
    win, cnt = Counter(votes).most_common(1)[0]
    conf = 0.75 if cnt == len(votes) and len(votes) > 1 else 0.6
    return {"answer": win, "confidence": conf,
            "candidate_scores": {win: conf}, "route": "general_knowledge",
            "votes": votes}
