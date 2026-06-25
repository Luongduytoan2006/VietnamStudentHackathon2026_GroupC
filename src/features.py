# -*- coding: utf-8 -*-
"""[2] Parse + [3] Feature extraction + detect loại câu. Theo temp.txt [2x][3x].

Tách context (đoạn dài) khỏi câu hỏi thật, trích feature để router [4] quyết định nhánh.
"""
import re

# [2b] pattern báo hiệu có context dài
_CTX_MARK = re.compile(
    r"Đoạn thông tin|Tiêu đề:|Nội dung:|Theo nội dung được cung cấp|"
    r"Theo ngữ cảnh|Theo đoạn (?:văn|thông tin)|Dựa vào đoạn|Đọc đoạn",
    re.IGNORECASE,
)
# câu hỏi thật thường bắt đầu bằng các cụm này (sau context)
_Q_MARK = re.compile(
    r"(Theo nội dung được cung cấp[,:]?|Theo đoạn[^,.]*[,:]|Câu hỏi[:：]|"
    r"Dựa vào (?:đoạn|nội dung|thông tin)[^,.]*[,:]?)",
    re.IGNORECASE,
)

_LATEX = re.compile(r"\$|\\frac|\\beta|\\alpha|\\sum|\\sqrt|\\epsilon|\\partial|\\int|e\^|\^\{|\\cdot|\\times")
_MATH_KW = re.compile(
    r"tính|bao nhiêu|giá trị|độ co giãn|tốc độ|xác suất|phương trình|đạo hàm|"
    r"tích phân|nồng độ|vận tốc|gia tốc|khối lượng|năng lượng|công suất|diện tích|"
    r"thể tích|bán kính|deflator|elasticity|gdp|lãi suất|phần trăm|đơn vị|"
    r"hàm phân phối|ma trận|nhiệt|mol|electron|bước sóng",
    re.IGNORECASE,
)
_LEGAL_KW = re.compile(
    r"luật|nghị định|thông tư|nghị quyết|điều\s*\d|khoản|pháp luật|hiến pháp|"
    r"bộ luật|văn bản|quyết định số|chính phủ|quốc hội|căn cước|thường trú|"
    r"hộ khẩu|hồ sơ|thẩm duyệt|phòng cháy|chữa cháy|đăng ký|giấy phép|"
    r"pháp nhân thương mại|cơ quan chức năng|thủ tục hành chính|nghĩa vụ",
    re.IGNORECASE,
)
# [3b-1] safety: ý đồ lách luật / tiếp tục hành vi bị cấm
_SAFETY_KW = re.compile(
    r"tiếp tục hoạt động.{0,30}(?:đình chỉ|bị cấm|thu hồi)|lách luật|"
    r"né tránh (?:kiểm tra|pháp luật|cơ quan)|trốn tránh trách nhiệm|"
    r"vẫn (?:hoạt động|kinh doanh).{0,20}(?:bị cấm|đình chỉ)|"
    r"làm thế nào để (?:vi phạm|gian lận|làm giả|trốn)",
    re.IGNORECASE,
)
# [GUARD] Ý ĐỒ XẤU: câu xui làm điều phá hoại/chống phá. Khi có kèm option "từ chối trả lời"
# thì refusal MỚI là đáp án đúng (đo offline trên 463: fix 3 câu, 0 phá → 90.7%→91.36%).
_HARM_INTENT = re.compile(
    r"phá hoại|xuyên tạc|kích động|bôi nhọ|xúc phạm|vu khống|làm suy yếu|"
    r"gây khó khăn cho việc|chống phá|lật đổ|làm giả|gian lận|"
    r"để (?:vi phạm|trốn|né tránh)",
    re.IGNORECASE,
)
# option kiểu TỪ CHỐI trả lời
_REFUSAL_OPT = re.compile(
    r"tôi không thể trả lời|không thể trả lời câu|nằm ngoài phạm vi|"
    r"ngoài phạm vi (?:trả lời|câu)|tôi không thể giúp",
    re.IGNORECASE,
)


def refusal_option_label(item):
    """Trả nhãn (A..) của option 'từ chối trả lời' nếu có, ngược lại None."""
    from .io_utils import make_labels
    labels = make_labels(len(item["choices"]))
    for lab, ch in zip(labels, item["choices"]):
        if _REFUSAL_OPT.search(str(ch)):
            return lab
    return None


def harm_with_refusal(item):
    """[GUARD] Trả nhãn option từ chối nếu câu có ý đồ xấu VÀ có sẵn option từ chối → đó là đáp án đúng.
    Ngược lại None. Rule deterministic, 0 model call. Đo offline: fix 3, phá 0."""
    if not _HARM_INTENT.search(item["question"]):
        return None
    return refusal_option_label(item)


def _has_numeric_choices(choices):
    hit = sum(1 for c in choices if re.search(r"\d", str(c)))
    return hit >= max(2, len(choices) // 2)


def split_context(question):
    """[2b] Tách (context, main_question). Nếu không có context dài → (None, question)."""
    if not _CTX_MARK.search(question):
        return None, question
    # tìm vị trí câu hỏi thật bắt đầu
    m = _Q_MARK.search(question)
    if m and m.start() > 80:  # context phải đủ dài mới tách
        return question[:m.start()].strip(), question[m.start():].strip()
    # fallback: câu cuối cùng kết thúc bằng '?' là câu hỏi
    qs = [s for s in re.split(r"(?<=[.?!])\s+", question) if s.strip()]
    if len(qs) >= 2 and len(question) > 400:
        return " ".join(qs[:-1]).strip(), qs[-1].strip()
    return question if len(question) > 400 else None, question


def extract_features(item):
    """[3a] Feature dict cho router."""
    q = item["question"]
    ch = item["choices"]
    blob = q + " " + " ".join(map(str, ch))
    ctx, main_q = split_context(q)
    return {
        "context": ctx,
        "main_question": main_q,
        "has_context": ctx is not None,
        "context_length": len(ctx) if ctx else 0,
        "has_latex": bool(_LATEX.search(blob)),
        "has_math_kw": bool(_MATH_KW.search(q)),
        "numeric_choices": _has_numeric_choices(ch),
        "has_legal": bool(_LEGAL_KW.search(q)),
        "has_safety": bool(_SAFETY_KW.search(blob)),
        "num_choices": len(ch),
    }


def route_question(item, feat):
    """[3b]+[4] Router rule-based. Thứ tự ưu tiên theo temp.txt."""
    if feat["has_safety"]:
        return "safety"
    if feat["has_latex"] or (feat["has_math_kw"] and feat["numeric_choices"]):
        return "calculation"
    if feat["has_context"]:
        return "reading_context"
    if feat["has_legal"]:
        return "legal_admin"
    return "general_knowledge"
