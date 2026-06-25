# -*- coding: utf-8 -*-
"""Orchestrator pipeline v2: ghép [1]→[9]. solve_one(engine, item) → answer letter.

[1] parse → [2][3] features+route → [4] router → [5x] solver →
[6] scores → [7] confidence gate (+self-check) → [8] final → (caller ghi [9]).
"""
import os
from .features import extract_features, route_question, harm_with_refusal
from . import solvers
from .extract import extract_answer
from .io_utils import make_labels

# Cấu hình calc đọc từ env (mặc định = bộ chuẩn 90.7%: CALC_MAXTOK=2000).
# CALC_MAXTOK: cho toán viết HẾT bài giải. 420 gây cụt reasoning -> đoán mò sai. 2000 -> calc 64%->96%.
# CALC_VOTE: 1=greedy. vote>1 (temp cao) đã CHỨNG MINH phản tác dụng (reasoning dài hơn -> cụt nhiều hơn).
_CALC_VOTE = int(os.environ.get("CALC_VOTE", "1"))
_CALC_MAXTOK = int(os.environ.get("CALC_MAXTOK", "2000"))
# [GUARD] câu xui phá hoại + có option từ chối → chọn refusal (mặc định BẬT; đo offline: +3, phá 0).
_HARM_GUARD = os.environ.get("HARM_GUARD", "1") == "1"

_DISPATCH = {
    "safety": solvers.solve_safety,
    "calculation": solvers.solve_calculation,
    "reading_context": solvers.solve_reading,
    "legal_admin": solvers.solve_legal,
    "general_knowledge": solvers.solve_general,
}

# v3 HYBRID: chỉ các route trong set này dùng solver chuyên biệt; còn lại dùng grammar đọc-thẳng.
# Căn cứ đo Q5 (so pred_final, per-route): calc CoT +5; reading RAG -14; general/legal hòa.
# → chỉ calculation (+safety rule) đáng, phần còn lại grammar thắng.
_HYBRID_SOLVER_ROUTES = {"calculation", "safety"}


def _grammar_direct(engine, item):
    """Grammar đọc-thẳng cả câu (kể cả reading context dài) — baseline mạnh nhất cho non-STEM."""
    ans = solvers._grammar_choice(engine, item, solvers._SYS_DIRECT, solvers._user_direct(item))
    return {"answer": ans, "confidence": 0.7, "candidate_scores": {ans: 0.7}}


def _self_check(engine, item, feat, result):
    """[7b] conf trung bình → direct + eliminate, lấy đồng thuận."""
    n = len(item["choices"])
    # direct answer độc lập
    direct = solvers._grammar_choice(engine, item, solvers._SYS_DIRECT,
                                     solvers._user_direct(item))
    cand = result["answer"]
    if direct == cand:
        result["confidence"] = min(0.9, result["confidence"] + 0.15)
        return result
    # bất đồng → ưu tiên theo route (calc/safety/reading mạnh hơn direct)
    if result["route"] in ("calculation", "safety", "reading_context"):
        return result  # giữ solver
    # general/legal: nghiêng về direct (vote 2/3 kiểu nhẹ)
    result["answer"] = direct
    result["confidence"] = 0.55
    return result


def solve_one(engine, item, do_self_check=True, mode="hybrid"):
    """mode='full' = v2 (mọi route dùng solver chuyên biệt).
       mode='hybrid' = v3 (chỉ calc+safety dùng solver, còn lại grammar đọc-thẳng) — TỐT NHẤT (78%)."""
    feat = extract_features(item)

    # [GUARD] deterministic (trước router, 0 model call): câu xui phá hoại + có option
    # "Tôi không thể trả lời" → refusal là đáp án đúng. Đo offline 463: fix 3, phá 0.
    if _HARM_GUARD:
        ref_lab = harm_with_refusal(item)
        if ref_lab is not None:
            return {"answer": ref_lab, "route": "safety_refusal",
                    "confidence": 0.9, "candidate_scores": {ref_lab: 0.9}}

    route = route_question(item, feat)

    if mode == "hybrid" and route not in _HYBRID_SOLVER_ROUTES:
        result = _grammar_direct(engine, item)
        result["route"] = route
        ans = result["answer"]
        if ans not in make_labels(len(item["choices"])):
            ans = "A"
        result["answer"] = ans
        return result

    if route == "calculation":
        result = solvers.solve_calculation(engine, item, feat,
                                           max_tokens=_CALC_MAXTOK, n_vote=_CALC_VOTE)
    else:
        result = _DISPATCH[route](engine, item, feat)

    # [7] confidence gate
    conf = result.get("confidence", 0.6)
    scores = result.get("candidate_scores", {})
    margin = 1.0
    if len(scores) >= 2:
        s = sorted(scores.values(), reverse=True)
        margin = s[0] - s[1]

    if do_self_check and (conf < 0.75 or margin < 0.15):
        # [7b] self-check cho câu chưa chắc, TRỪ reading (đã đắt) để tiết kiệm
        if route in ("general_knowledge", "legal_admin"):
            result = _self_check(engine, item, feat, result)

    ans = result["answer"]
    n = len(item["choices"])
    # đảm bảo hợp lệ
    if ans not in make_labels(n):
        ans = "A"
    result["answer"] = ans
    result["route"] = route
    return result
