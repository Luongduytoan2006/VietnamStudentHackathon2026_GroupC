# PIPELINE — Chi tiết luồng xử lý & lý do thiết kế

Pipeline giải 1 câu trắc nghiệm tiếng Việt bằng **Qwen3.5-9B** (GGUF, gọi hàm in-process qua
`llama-cpp-python` — KHÔNG server/port). Đạt **91.36%** trên public 463 câu.

## Sơ đồ tổng quát

```
[1] INPUT {qid, question, choices[]}          (nhãn động A..K, không giả định 4 lựa chọn)
        │
        ▼
[2] HARM-GUARD  ◄── rule thuần, 0 gọi model
        │  Câu có ý đồ XẤU (phá hoại/xuyên tạc/kích động/bôi nhọ...)
        │  VÀ có option "Tôi không thể trả lời"?
        ├── CÓ ──► chọn ngay option TỪ CHỐI → trả về (0.0s)
        └── KHÔNG ─┐
                   ▼
[3] FEATURE (rule, 0 gọi model): has_context / has_latex / has_math_kw /
        numeric_choices / has_legal / has_safety / num_choices
                   ▼
[4] ROUTER (rule, chọn 1 trong 5 nhánh — theo thứ tự ưu tiên)
        has_safety ───────────────────► safety
        has_latex OR (math_kw + số) ───► calculation
        has_context ───────────────────► reading_context
        has_legal ─────────────────────► legal_admin
        else ──────────────────────────► general_knowledge
                   ▼
[5] SOLVER theo nhánh
        safety        → quét keyword (tuân thủ/từ chối/hợp pháp) → chọn; không rõ → grammar
        calculation   → CoT ĐẦY ĐỦ max_tokens=2000 → "Đáp án: X"     ◄ gọi model (nặng)
        reading       → grammar đọc-THẲNG cả đoạn → ép 1 ký tự        ◄ gọi model
        legal         → grammar đọc-thẳng → ép 1 ký tự                ◄ gọi model
        general       → grammar đọc-thẳng → ép 1 ký tự                ◄ gọi model
                   ▼
[6] VALIDATE: ép answer ∈ {A..n}; lỗi/thiếu → "A"   (pred.csv không nhãn rác)
                   ▼
[7] GHI pred.csv → qid,answer
```

## "Grammar ép 1 ký tự" (GBNF)
Thay vì để model viết tự do rồi đoán đáp án, ép model CHỈ xuất đúng 1 chữ cái hợp lệ:
```
root ::= [A-D]      # câu 4 lựa chọn → model buộc trả A/B/C/D, không gì khác
```
→ không sinh rác, nhanh (1 token), trích đáp án chính xác 100%. Nhãn co giãn theo số lựa chọn (A..K).

## Vì sao thiết kế vậy — lý do từng phần

| Thành phần | Lý do (có số đo) |
|---|---|
| **[2] Harm-guard đặt ĐẦU** | Rẻ nhất (rule, 0 token). Bắt sớm câu "bẫy đạo đức" (xui làm điều xấu + có option từ chối). Đo offline: **+3 câu, phá 0**. Vừa đúng đáp án vừa đúng hành vi an toàn. |
| **[4] Router bằng RULE** | Phân loại chỉ cần từ khóa → 0 token, ~0ms, dễ debug, đỡ 1 lần gọi model/câu. Sai số không nằm ở router. |
| **calc → CoT 2000 token** | Toán cần viết HẾT bài giải. Token cụt (420) khiến giải đúng vẫn bị ghi sai. Nâng 2000 → calc **59%→96%**. Đòn bẩy lớn nhất. |
| **reading/legal/general → đọc THẲNG, KHÔNG RAG** | Đã thử RAG: cắt nhỏ đoạn làm **mất câu chứa đáp án** → tụt 95%→77%. Model 9B đọc nguyên đoạn tốt hơn. |
| **Hybrid (chỉ calc+safety dùng solver riêng)** | Đo per-route: chỉ calc-CoT đáng dùng solver; reading/legal/general đọc-thẳng đã thắng → nhanh + chính xác nhất. |
| **1 model, in-process** | Đề cấm localhost → gọi hàm thẳng, không server/port. Self-contained, pull về chạy ngay. |

## Các hướng đã thử & BỎ (ablation trên 463 câu)
| Cải tiến | Kết quả đo | Quyết định |
|---|---|---|
| PAL (viết code Python giải toán) | calc 178→169 (**−9**, đè CoT đúng, fire cả câu không thuần số) | ❌ bỏ |
| RAG rerank (đoạn dài) | reading net **0** (fix 1 / phá 1, churn) | ❌ bỏ |
| Committee (đảo thứ tự + loại trừ) | **−1** (lật câu đang đúng) | ❌ bỏ |
| Self-consistency vote (temp>0) | chậm 40-145s/câu + phản tác dụng | ❌ bỏ |
| **Harm-guard** | **+3, phá 0** | ✅ **giữ** |

**Triết lý:** đốt compute ĐÚNG chỗ dễ sai nhất (toán → CoT dài), còn lại giữ đơn giản + nhanh
(đọc thẳng + grammar ép 1 ký tự). Cải tiến nào đo ra không tăng thì BỎ — với Qwen3.5-9B, pipeline
hybrid đã gần trần; thêm compute không tăng điểm.

## Cấu hình (1 nguồn sự thật: config.py + docker-compose.yml)
| Env | Default | Ý nghĩa |
|---|---|---|
| `QUANT` | Q5 | quant Qwen (Q4/Q5/Q6/Q8) |
| `N_CTX` | 6144 | context window |
| `CALC_MAXTOK` | 2000 | token reasoning toán — đừng hạ |
| `MODE` | hybrid | calc+safety dùng solver, còn lại grammar đọc-thẳng |
| `HARM_GUARD` | 1 | bật guard câu phá hoại + option từ chối → refusal |

Kết quả per-route (public 463): calculation 96% · reading 95% · general 85% · legal 79% · safety 100%.
