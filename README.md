# HackAIthon 2026 — Bảng C — Giải trắc nghiệm tiếng Việt (Qwen3.5-9B)

Pipeline giải câu hỏi trắc nghiệm (MCQ) tiếng Việt bằng **Qwen3.5-9B** (GGUF Q5_K_M, chạy qua `llama-cpp-python` — gọi hàm **in-process**, KHÔNG qua server/port).

- Đọc test ở `/data`, ghi đáp án `pred.csv` (`qid,answer`) vào `/output`.

---

# 🚀 Cách chạy (3 bước)

Bọn em đã đóng gói sẵn toàn bộ pipeline (kèm model nhúng bên trong) thành một Docker image
và đẩy lên Docker Hub, nên BTC không phải build hay tải model — chỉ cần pull về là chạy được.

> Image dùng CUDA nên cần máy có **GPU NVIDIA** + Docker. Cần khoảng **7GB VRAM trống**
> (model Qwen Q5 chiếm ~6.5GB). Lần đầu cần mạng để pull image (~12GB); pull xong thì chạy không cần mạng.

### Bước 1 — Chuẩn bị thư mục làm việc
Lấy thư mục này về theo một trong hai cách:
- **Clone từ GitHub:** `git clone <repo_url> && cd submission`, hoặc
- **Tự tạo một thư mục** chứa file `docker-compose.yml` (bọn em đã để sẵn trong repo),
  bên trong có hai thư mục con `data/` và `output/`.

### Bước 2 — Bỏ đề thi vào `data/`
Đặt file đề của BTC vào `./data`, đặt tên `private_test.json` (hoặc `private_test.csv`):
```bash
cp <file_đề_của_BTC>  ./data/private_test.json
```
> JSON là list các `{"qid": "...", "question": "...", "choices": ["...", "..."]}`; định dạng CSV cũng nhận được.

### Bước 3 — Chạy
```bash
docker compose up
```
Lệnh này tự pull image từ Docker Hub (lần đầu) rồi chạy; container chạy xong sẽ tự thoát.
Kết quả nằm ở **`./output/pred.csv`** (2 cột `qid,answer`).

> Nếu không dùng compose, có thể chạy trực tiếp:
> ```bash
> docker run --gpus all -v "${PWD}/data:/data" -v "${PWD}/output:/output" karuizawa/hackaithon-bangc:guard
> ```

**Biết là đã chạy xong:** log kết thúc bằng `[main] XONG -> /output/pred.csv (.. dòng)`, và file
`./output/pred.csv` có số dòng = số câu đề + 1 dòng header, mỗi `answer` là một chữ cái A..K.
Tốc độ tham khảo khoảng ~5–6 giây/câu trên RTX 4060 8GB.

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
| **[2] Harm-guard đặt ĐẦU** | Nhóm em phỏng đoán câu "bẫy đạo đức" (xui làm điều xấu + có sẵn option từ chối) thì option từ chối vừa đúng đáp án vừa đúng hành vi an toàn. Bắt bằng rule là rẻ nhất (0 token) nên đặt ngay đầu pipeline. |
| **[4] Router bằng RULE** | Nhóm em cho rằng phân loại loại câu chỉ cần từ khóa → 0 token, ~0ms, dễ debug, đỡ 1 lần gọi model/câu; sai số thực tế không nằm ở khâu phân loại. |
| **calc → CoT 2000 token** | Toán cần viết HẾT bài giải mới ra số đúng; token cụt khiến model giải đúng nhưng bị ghi nhầm đáp án. Nhóm em nới rộng token cho riêng nhánh toán → đây là nhánh cải thiện rõ nhất. |
| **reading/legal/general → đọc THẲNG, KHÔNG RAG** | Nhóm em phỏng đoán (và quan sát thấy) việc cắt nhỏ đoạn để RAG dễ làm mất câu chứa đáp án; model 9B đọc nguyên đoạn ngắn vẫn nắm tốt hơn nên giữ đọc-thẳng. |
| **Hybrid (chỉ calc+safety dùng solver riêng)** | Nhóm em phỏng đoán chỉ toán (cần CoT) và safety (cần rule từ chối) là đáng dùng solver chuyên biệt; các nhánh còn lại đọc-thẳng cho kết quả nhanh và chính xác hơn. |
| **1 model, in-process** | Đề cấm localhost → gọi hàm thẳng, không server/port. Self-contained, pull về chạy ngay. |

**Triết lý:** nhóm em chủ trương đốt compute ĐÚNG chỗ dễ sai nhất (toán → CoT dài), còn các
nhánh khác giữ đơn giản + nhanh (đọc thẳng + grammar ép 1 ký tự). Với Qwen3.5-9B, đây là điểm
cân bằng mà nhóm phỏng đoán cho độ chính xác cao nhất trên tốc độ chấp nhận được.

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
| `HARM_GUARD` | `1` | Câu xui phá hoại + có lựa chọn "Tôi không thể trả lời" → chọn từ chối |

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
  README.md            # file này (sơ đồ pipeline + lý do thiết kế đầy đủ ở trên)
  .dockerignore        # loại data/output/md khỏi build context
  .gitignore           # chặn model + output + log khỏi GitHub
```