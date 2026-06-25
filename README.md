# HackAIthon 2026 — Bảng C — Giải trắc nghiệm tiếng Việt (Qwen3.5-9B)

Pipeline giải câu hỏi trắc nghiệm (MCQ) tiếng Việt bằng **Qwen3.5-9B** (GGUF Q5_K_M, chạy qua `llama-cpp-python` — gọi hàm **in-process**, KHÔNG qua server/port). Hợp lệ thể lệ (model ≤ 9B).

- **CHỈ 1 model duy nhất** (Qwen3.5-9B). KHÔNG cần embedding, KHÔNG cần Ollama, KHÔNG cần localhost, KHÔNG cần internet lúc chạy.
- Đọc test ở `/data`, ghi đáp án `pred.csv` (`qid,answer`) vào `/output`.
- **Accuracy: 91.36%** trên public 463 câu (so bộ tham chiếu).

---

# ⭐ HƯỚNG DẪN CHO BTC — PULL VỀ & CHẠY (3 bước)

> Cách nhanh nhất: pull image có sẵn từ Docker Hub (model đã nhúng trong image, không cần tải gì thêm, không cần build).

### ✅ ĐIỀU KIỆN BẮT BUỘC (phải đủ thì mới chạy ra kết quả)
1. **Máy có GPU NVIDIA** + đã cài **NVIDIA driver** + **NVIDIA Container Toolkit** (để Docker thấy GPU).
   - Không có GPU vẫn chạy được nhưng RẤT chậm (CPU fallback) — khuyến nghị dùng GPU.
2. **Docker** (hoặc Docker Desktop) đang chạy.
3. **VRAM trống ≥ 7GB** (model Qwen Q5 chiếm ~6.5GB). GPU 8GB là đủ.
4. Có **mạng để pull image lần đầu** (~12GB). Sau khi pull xong thì **chạy KHÔNG cần mạng**.

### 📦 BƯỚC 1 — Bỏ file test vào thư mục `data/`
```bash
# Đặt file đề của BTC vào ./data với MỘT trong các tên sau:
#   private_test.json   hoặc   private_test.csv
cp <file_test_cua_BTC>  ./data/private_test.json
```
> Định dạng JSON: list các `{"qid": "...", "question": "...", "choices": ["...", "..."]}`.
> Định dạng CSV cũng nhận (cột `qid,question,choice_a..` hoặc `qid,question,choices`).

### ▶️ BƯỚC 2 — Chạy
```bash
docker compose up        # tự pull image lần đầu rồi chạy; container chạy xong tự thoát
```
Hoặc không dùng compose:
```bash
docker pull karuizawa/hackaithon-bangc:guard
docker run --gpus all \
  -v "${PWD}/data:/data" \
  -v "${PWD}/output:/output" \
  karuizawa/hackaithon-bangc:guard
```

### 📄 BƯỚC 3 — Lấy kết quả
```
./output/pred.csv      (2 cột: qid,answer)
```

### ✔️ Dấu hiệu chạy ĐÚNG (kiểm tra để chắc có kết quả)
- Log in ra `[main] model=Qwen3.5-9B-Q5_K_M.gguf | n_gpu_layers=-1 (gpu=True, ...)` → đã thấy GPU
- Log chạy dần `[main] 25/.. | ..s/câu`, kết thúc bằng `[main] XONG -> /output/pred.csv (.. dòng)`.
- File `./output/pred.csv` có đủ số dòng = số câu đề + 1 dòng header, mọi `answer` là chữ cái A..K.
- Tốc độ tham chiếu ~5–6s/câu trên RTX 4060 8GB (toán chậm hơn, đọc/kiến thức nhanh).

### ⚠️ Lỗi thường gặp
| Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|
| `could not select device driver ... gpu` | thiếu NVIDIA Container Toolkit | cài toolkit, hoặc bỏ `--gpus all` (chạy CPU, chậm) |
| `libcuda.so.1 cannot open` | quên cờ `--gpus all` | thêm `--gpus all` vào `docker run` |
| `n_gpu_layers=0` dù có GPU | VRAM bị process khác chiếm | giải phóng VRAM (đóng app khác) rồi chạy lại |
| `KHÔNG tìm thấy file test` | chưa bỏ file vào `./data` | đặt `private_test.json`/`.csv` vào `./data` |

---

# 🔧 Build lại từ source (cho người dev, KHÔNG cần cho BTC)

Folder này cũng là source đầy đủ để build lại image:
```bash
git clone <repo_url> && cd submission
# Đặt model vào ./models trước khi build (model không lên GitHub vì >100MB)
# Tải tay: python -c "import config; config.ensure_model('Q5')"
docker build -t karuizawa/hackaithon-bangc:guard .
docker push karuizawa/hackaithon-bangc:guard
```

Hoặc chạy thẳng từ source bằng venv (Windows/Linux có GPU):
```bash
uv venv && uv sync
uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12  # Windows
python -c "import config; config.ensure_model('Q5')"      # tải Qwen Q5 (~6.6GB)
DATA_DIR=. OUTPUT_DIR=. python main.py --test data/public_test.json --out output/pred.csv
```

---

# 📊 PIPELINE — Sơ đồ xử lý 1 câu hỏi

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
[3] FEATURE (rule, 0 gọi model)
        has_context / context_length / has_latex / has_math_kw /
        numeric_choices / has_legal / has_safety / num_choices
                   │
                   ▼
[4] ROUTER (rule, chọn 1 trong 5 nhánh — theo thứ tự ưu tiên)
        has_safety ───────────────────► safety
        has_latex OR (math_kw + số) ───► calculation
        has_context ───────────────────► reading_context
        has_legal ─────────────────────► legal_admin
        else ──────────────────────────► general_knowledge
                   │
                   ▼
[5] SOLVER theo nhánh
        safety        → quét keyword (tuân thủ/từ chối/hợp pháp) → chọn; không rõ → grammar
        calculation   → CoT ĐẦY ĐỦ max_tokens=2000 → "Đáp án: X"     ◄ gọi model (nặng)
        reading       → grammar đọc-THẲNG cả đoạn → ép 1 ký tự        ◄ gọi model
        legal         → grammar đọc-thẳng → ép 1 ký tự                ◄ gọi model
        general       → grammar đọc-thẳng → ép 1 ký tự                ◄ gọi model
                   │
                   ▼
[6] VALIDATE: ép answer ∈ {A..n}; lỗi/thiếu → "A"   (pred.csv không nhãn rác)
                   │
                   ▼
[7] GHI pred.csv → qid,answer
```

### "Grammar ép 1 ký tự" (GBNF)
Thay vì để model viết tự do rồi đoán đáp án, ép model CHỈ xuất đúng 1 chữ cái hợp lệ:
```
root ::= [A-D]      # câu 4 lựa chọn → model buộc trả A/B/C/D, không gì khác
```
→ không sinh rác, nhanh (1 token), trích đáp án chính xác 100%. Nhãn co giãn theo số lựa chọn (A-K).

---

# 🧪 Quy trình phân tích & vì sao chọn vậy

### Lý do từng phần (có số đo)
| Thành phần | Lý do |
|---|---|
| **[2] Harm-guard đặt ĐẦU** | Rẻ nhất (rule, 0 token). Bắt sớm câu "bẫy đạo đức" (xui làm điều xấu). Baseline offline: **+3 câu, phá 0**. Vừa đúng đáp án vừa đúng hành vi an toàn. |
| **[4] Router bằng RULE** | Phân loại chỉ cần từ khóa → 0 token, ~0ms, dễ debug, đỡ 1 lần gọi model/câu. Sai số không nằm ở router. |
| **calc → CoT 2000 token** | Toán cần viết HẾT bài giải. Token cụt (420) khiến giải đúng vẫn bị ghi sai. Nâng 2000 → calc **59%→96%**. Đòn bẩy lớn nhất. |
| **reading/legal/general → đọc THẲNG, KHÔNG RAG** | Đã thử RAG: cắt nhỏ đoạn làm mất câu chứa keyword. 9B đọc nguyên đoạn tốt hơn. |
| **Hybrid (chỉ calc+safety dùng solver riêng)** | Đo per-route: chỉ calc-CoT đáng dùng solver; reading/legal/general đọc-thẳng đã thắng → nhanh + chính xác nhất. |
| **1 model, in-process** | Đề cấm localhost → gọi hàm thẳng, không server/port. Self-contained, pull về chạy ngay. |

### Các hướng đã thử & BỎ (ablation trên 463 câu)
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

### Kết quả per-route (public 463 câu)
- calculation: 96% (178/185)
- reading_context: 95% (73/77)
- general_knowledge: 85% (146/172)
- legal_admin: 79% (22/28)
- safety: 100% (1/1)
- **Tổng: 423/463 = 91.36%**

---

# ⚙️ Cấu hình (đã set sẵn mức CHUẨN, không cần đổi)

| Env | Default | Ý nghĩa |
|---|---|---|
| `QUANT` | `Q5` | Quant Qwen: Q4/Q5/Q6/Q8. Q5_K_M cân bằng acc/tốc độ |
| `N_CTX` | `6144` | Context window (đủ câu dài + reasoning toán) |
| `CALC_MAXTOK` | `2000` | Token tối đa reasoning toán — **đừng hạ** (cụt = sai) |
| `MODE` | `hybrid` | calc+safety dùng solver, còn lại grammar đọc-thẳng |
| `HARM_GUARD` | `1` | Câu xui phá hoại + có lựa chọn "Tôi không thể trả lời" → chọn từ chối (+3 câu, 0 rủi ro) |

### Model (đã nhúng sẵn trong image)
| File | Kích thước | Nguồn HF |
|---|---|---|
| `Qwen3.5-9B-Q5_K_M.gguf` | ~6.6GB | `unsloth/Qwen3.5-9B-GGUF` |

> Không cần model embedding. Pipeline chỉ dùng 1 model LLM.

---

# 📁 Cấu trúc thư mục

```
submission/
  main.py              # ENTRYPOINT: đọc /data → giải → ghi /output/pred.csv
  config.py            # cấu hình chuẩn + tải model Qwen
  src/
    bootstrap.py       # preload DLL CUDA (Windows; no-op trên Linux/Docker)
    io_utils.py        # đọc test .json/.csv, ghi pred.csv, nhãn A–K động
    features.py        # trích đặc trưng + router 5 nhánh + HARM-GUARD
    pipeline.py        # orchestrator solve_one (guard → router → solver → validate)
    solvers.py         # solver theo route: calc CoT / safety / reading / legal / general
    extract.py         # trích đáp án A–K từ output model
    engine.py          # wrapper llama.cpp (sinh + grammar ép 1 ký tự)
  models/              # Qwen3.5-9B-Q5_K_M.gguf  (nhúng vào image khi build; KHÔNG lên GitHub)
  data/                # BTC bỏ private_test.json/.csv vào đây
  output/              # pred.csv xuất ra đây
  Dockerfile           # build image (nhúng model vào)
  docker-compose.yml   # PULL-AND-RUN: kéo image từ Hub rồi chạy (không build)
  pyproject.toml       # deps cho dev (llama-cpp-python, hf-hub, numpy)
  README.md            # file này
  PIPELINE.md          # chi tiết pipeline (rút gọn từ phần "Pipeline" ở trên)
  .dockerignore        # loại data/output/md khỏi build context
  .gitignore           # chặn model + output + log khỏi GitHub
```

Chi tiết luồng xử lý từng bước và lý do thiết kế: xem [PIPELINE.md](PIPELINE.md) (rút gọn từ phần Pipeline ở trên).