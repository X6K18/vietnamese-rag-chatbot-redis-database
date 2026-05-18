# Vietnamese Realtime RAG Chatbot

Hệ thống chatbot tiếng Việt thời gian thực sử dụng **Retrieval-Augmented Generation (RAG)** kết hợp **Redis**, **PhoBERT**, **FAISS**, và **Ollama**.

---

## Mục lục

- [Mục đích dự án](#mục-đích-dự-án)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Lý thuyết nền tảng](#lý-thuyết-nền-tảng)
  - [Retrieval-Augmented Generation (RAG)](#1-retrieval-augmented-generation-rag)
  - [Embedding và Vector Search](#2-embedding-và-vector-search)
  - [PhoBERT (Vietnamese BERT)](#3-phobert-vietnamese-bert)
  - [Maximum Marginal Relevance (MMR)](#4-maximum-marginal-relevance-mmr)
  - [FAISS (Facebook AI Similarity Search)](#5-faiss-facebook-ai-similarity-search)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Pipeline xử lý](#pipeline-xử-lý)
- [Các thành phần chi tiết](#các-thành-phần-chi-tiết)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [API Endpoints](#api-endpoints)

---

## Mục đích dự án

Xây dựng một **hệ thống chatbot thông minh** có khả năng:

- **Hiểu câu hỏi tiếng Việt** thông qua PhoBERT分类
- **Truy xuất thông tin chính xác** từ kho dữ liệu báo chí Việt Nam (hơn 36,000 đoạn văn bản đã được chunk)
- **Trả lời có trích dẫn nguồn** nhờ cơ chế RAG
- **Ghi nhớ hội thoại theo phiên** qua Redis
- **Streaming realtime** như ChatGPT
- **Đề xuất câu hỏi gợi ý** để tương tác tự nhiên hơn
- **Fallback thông minh**: nếu không tìm thấy tài liệu phù hợp, dùng kiến thức riêng của Ollama để trả lời

---

## Kiến trúc hệ thống

```
User
  |
  v
+------------------+       +------------------+
|   Streamlit App  | <---> |  FastAPI Server  |
|   (Frontend)     |       |  (Backend)       |
+------------------+       +--------+---------+
                                    |
                     +--------------+--------------+
                     |              |              |
                     v              v              v
              +-----------+  +----------+  +------------+
              |  PhoBERT  |  |  FAISS   |  |   Redis    |
              | Classifier|  |  Index   |  | Chat + Emb |
              +-----------+  +----------+  +------------+
                     |              |              |
                     +------+-------+              |
                            |                      |
                            v                      v
                     +-----------+          +------------+
                     |  Ollama   |          |  36,506    |
                     | qwen2.5   |          |  Documents |
                     +-----------+          +------------+
                            |
                            v
                    Streaming Answer
                    (NDJSON events)
```

### Luồng xử lý

1. **User** gửi câu hỏi qua Streamlit UI
2. **FastAPI** nhận request, gửi đến PhoBERT Classifier để phân loại chủ đề
3. **Query Expansion**: Mở rộng câu hỏi với chủ đề đã phân loại
4. **RAG Retrieval**: FAISS tìm kiếm vector tương đồng với câu hỏi đã mở rộng
5. **MMR Rerank**: Sắp xếp lại kết quả để đảm bảo đa dạng thông tin
6. **Kiểm tra độ liên quan**: Nếu tất cả documents có score < 0.3, dùng fallback Ollama
7. **Ollama LLM** sinh câu trả lời dựa trên context + lịch sử hội thoại
8. **Redis** lưu tin nhắn mới và tóm tắt hội thoại nếu cần
9. **Streaming response** trả về dưới dạng NDJSON (token, sources, follow_up, done)

---

## Lý thuyết nền tảng

### 1. Retrieval-Augmented Generation (RAG)

**RAG** là kiến trúc kết hợp giữa **truy xuất thông tin** (Retrieval) và **sinh văn bản** (Generation). Thay vì chỉ dùng kiến thức có sẵn trong mô hình ngôn ngữ, RAG cho phép mô hình truy xuất thông tin từ một kho dữ liệu bên ngoài trước khi sinh câu trả lời.

#### Công thức

Cho câu hỏi $q$, tìm tập tài liệu $D_{rel} = \{d_1, d_2, ..., d_k\}$ từ kho $D$:

$$D_{rel} = \text{top-}k \left( \text{sim}(q, d_i) \right), \forall d_i \in D$$

Sau đó, mô hình ngôn ngữ $M$ sinh câu trả lời $a$ dựa trên:

$$a = M(q, D_{rel}, h)$$

Trong đó $h$ là lịch sử hội thoại.

#### Lợi ích

- **Giảm ảo giác (hallucination)**: Mô hình dựa trên tài liệu thực tế
- **Cập nhật dễ dàng**: Chỉ cần cập nhập kho dữ liệu, không cần fine-tune model
- **Kiểm soát nguồn gốc**: Có thể trích dẫn chính xác tài liệu gốc

---

### 2. Embedding và Vector Search

#### Sentence Embedding

Chuyển văn bản thành vector số học có chiều cố định (768 chiều trong project này) bằng mô hình **SentenceTransformer**:

$$\text{emb}(t) = E(t) \in \mathbb{R}^{768}$$

Trong đó $E$ là mô hình `keepitreal/vietnamese-sbert`.

#### Similarity Search (L2 Distance)

FAISS index sử dụng **L2 distance (Euclidean)**:

$$d(q, d_i) = \sqrt{\sum_{j=1}^{768} (q_j - d_{i,j})^2}$$

Điểm tương đồng được chuyển thành **relevance score**:

$$\text{score}(q, d_i) = \frac{1}{1 + d(q, d_i)} \in (0, 1]$$

Score càng gần 1 càng liên quan.

---

### 3. PhoBERT (Vietnamese BERT)

**PhoBERT** ([Nguyen & Nguyen, 2020](https://github.com/VinAIResearch/PhoBERT)) là phiên bản BERT được tiền huấn luyện riêng cho tiếng Việt, dựa trên kiến trúc **RoBERTa**. PhoBERT vượt trội so với đa ngữ BERT trên các tác vụ NLP tiếng Việt.

Trong project này, PhoBERT được fine-tune để **phân loại chủ đề** cho câu hỏi đầu vào. Mô hình classification:

$$P(y=c | q) = \text{softmax}(W \cdot \text{PhoBERT}(q) + b)$$

Với 11 lớp chủ đề: `the_gioi`, `phap_luat`, `xa_hoi`, `kinh_te`, `y_te`, `thoi_su`, `the_thao`, `truyen_hinh`, `du_lich`, `van_hoa`, `thi_truong`.

---

### 4. Maximum Marginal Relevance (MMR)

**MMR** là kỹ thuật đa dạng hóa kết quả tìm kiếm, cân bằng giữa **độ liên quan** và **độ đa dạng**:

$$\text{MMR} = \arg\min_{d_i \in C \setminus S} \left[ \lambda \cdot \text{sim}(q, d_i) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j) \right]$$

Trong đó:
- $C$: tập ứng viên, $S$: tập đã chọn (ban đầu rỗng)
- $\text{sim}(q, d_i)$: độ tương đồng giữa câu hỏi và tài liệu
- $\text{sim}(d_i, d_j)$: độ tương đồng giữa hai tài liệu
- $\lambda \in [0, 1]$: tham số cân bằng (mặc định 0.5)

**Ý nghĩa**:
- $\lambda = 1$: Chỉ quan tâm độ liên quan (giống search thuần)
- $\lambda = 0$: Chỉ quan tâm độ đa dạng
- $\lambda = 0.5$: Cân bằng cả hai

---

### 5. FAISS (Facebook AI Similarity Search)

FAISS là thư viện tìm kiếm tương đồng vector hiệu năng cao của Meta. Project sử dụng **IndexFlatL2** - index brute-force chính xác tuyệt đối:

$$I, D = \text{FAISS.search}(q, k \cdot 5)$$

Trả về chỉ số $I$ và khoảng cách $D$ của $k \cdot 5$ kết quả gần nhất, sau đó lọc theo category và áp dụng MMR.

#### Quy mô index
- **36,506 vectors** (tương ứng 36,506 đoạn văn bản chunk từ 6,256 bài báo)
- **768 chiều** mỗi vector
- Khoảng cách L2 (Euclidean)

---

## Công nghệ sử dụng

| Công nghệ | Vai trò | Lý do chọn |
|-----------|---------|------------|
| **FastAPI** | Backend API framework | Async, performance, tự động document |
| **Streamlit** | Frontend UI | Python-native, nhanh, đẹp |
| **Redis** | Chat memory + Embedding cache | In-memory database siêu nhanh, realtime |
| **PhoBERT** | Text classification (chủ đề) | State-of-the-art cho tiếng Việt |
| **FAISS** | Vector similarity search | Index 36k vectors chỉ vài ms |
| **Sentence-Transformers** | Text embedding | `keepitreal/vietnamese-sbert` tối ưu cho tiếng Việt |
| **Ollama** | LLM inference (Qwen2.5:1.5b) | Chạy local, không cần GPU mạnh |
| **NumPy** | Numerical computation | Xử lý vector, ma trận |
| **scikit-learn** | Label encoding | Mã hóa nhãn chủ đề |
| **Torch** | Deep learning framework | Chạy PhoBERT |

---

## Cấu trúc thư mục

```
project/
│
├── backend/
│   ├── main.py              # FastAPI server, endpoints
│   ├── config.py            # Cấu hình (model path, Redis, Ollama)
│   ├── models.py            # (reserved)
│   ├── utils.py             # (reserved)
│   │
│   └── services/
│       ├── classifier.py    # PhoBERT text classification
│       ├── rag.py           # FAISS retrieval + MMR reranking
│       ├── llm.py           # Ollama prompt building + streaming
│       └── redis_client.py  # Redis chat history + embedding cache
│
├── frontend/
│   └── app.py               # Streamlit chat UI
│
├── models/
│   ├── phobert_model/       # Fine-tuned PhoBERT weights
│   ├── phobert_tokenizer/   # PhoBERT tokenizer
│   └── label_encoder.joblib # Label encoder (11 categories)
│
├── data/
│   ├── news_dataset.csv     # 6,256 bài báo gốc (32 MB)
│   ├── data.pkl             # 36,506 chunks (33 MB)
│   └── faiss.index          # 36,506 vectors 768-d (112 MB)
│
├── tests/
│   ├── test_redis.py        # Kiểm tra kết nối Redis
│   └── test_api.py          # Kiểm tra API endpoint
│
├── requirements.txt
└── README.md
```

---

## Pipeline xử lý

### 1. Classification

```
Câu hỏi: "Giá xăng dầu hôm nay thế nào?"
  |
  v
PhoBERT Tokenizer + Model
  |
  v
Chủ đề: "kinh_te" (confidence: 0.92)
```

### 2. Query Expansion

```
Câu hỏi gốc: "Giá xăng dầu hôm nay thế nào?"
Chủ đề: "kinh_te"
  |
  v
Câu hỏi mở rộng: "Giá xăng dầu hôm nay thế nào? (thuộc chủ đề kinh_te)"
```

### 3. Retrieval (FAISS + MMR)

```
Câu hỏi đã mở rộng
  |
  v
Sentence-Transformer Embedding (768-d)
  |
  v
FAISS IndexFlatL2 search → top 15 candidates
  |
  | (lọc theo category = "kinh_te")
  v
Category filtering → còn lại các document kinh_tế
  |
  | (nếu < kết quả, MMR đa dạng hóa)
  v
Top 3 documents với score và source
```

### 4. Relevance Check

```
Max score >= 0.3?
  |
  ├── YES → Prompt RAG (yêu cầu trích dẫn tài liệu)
  |
  └── NO → Prompt Fallback (dùng kiến thức Ollama)
```

### 5. Generation (Ollama)

```
Prompt (context + history + query)
  |
  v
Qwen2.5:1.5b (streaming)
  |
  v
Answer tokens → NDJSON events
```

### 6. Post-processing

```
Sau khi stream hoàn tất:
  |
  ├── Lưu câu trả lời vào Redis
  ├── Gửi sources event
  ├── Gửi follow_up event (sinh bằng Ollama)
  └── Gửi done event
```

---

## Các thành phần chi tiết

### `backend/services/classifier.py` - Phân loại chủ đề

Dùng PhoBERT fine-tuned với `AutoModelForSequenceClassification`. Đầu vào là câu hỏi tiếng Việt, đầu ra là một trong 11 chủ đề.

**Activation**: softmax trên logits cuối cùng

### `backend/services/rag.py` - Truy xuất và rerank

- **FAISS IndexFlatL2**: Tìm kiếm chính xác bằng L2 distance
- **MMR Reranking**: Công thức $\lambda \cdot \text{sim}(q,d_i) - (1-\lambda) \cdot \max \text{sim}(d_i, d_j)$
- **Category filtering**: Lọc tài liệu trùng chủ đề với câu hỏi

### `backend/services/llm.py` - Sinh câu trả lời

- **build_prompt()**: Xây dựng prompt với context và lịch sử
- **expand_query()**: Mở rộng câu hỏi với chủ đề
- **has_relevant_docs()**: Kiểm tra ngưỡng relevance score >= 0.3
- **generate_follow_up()**: Sinh câu hỏi gợi ý bằng Ollama
- **generate_stream()**: Stream câu trả lời từ Ollama, bắt lỗi ConnectionError

### `backend/services/redis_client.py` - Redis operations

- **Chat history**: Lưu và truy xuất lịch sử hội thoai với `RPUSH` / `LRANGE`
- **Conversation summarization**: Khi session > 20 messages, tóm tắt và trim list
- **Embedding cache**: Cache vector embedding với `SETEX` / `GET` (TTL: 1 giờ)

### `backend/config.py` - Cấu hình

```python
OLLAMA_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "keepitreal/vietnamese-sbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REDIS_TTL = 3600  # 1 giờ
```

### `frontend/app.py` - Giao diện người dùng

- Đăng nhập với hardcoded users
- Hiển thị lịch sử chat + nguồn tham khảo (expander)
- Nút câu hỏi gợi ý có thể click để gửi tiếp
- Admin panel quản lý session

---

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Health check |
| GET | `/health` | Health check + Redis ping |
| POST | `/chat/stream` | Chat streaming (NDJSON) |
| POST | `/chat/sync` | Chat đồng bộ (JSON) |
| GET | `/chat/history?session_id=...` | Lịch sử chat |
| DELETE | `/chat/history?session_id=...` | Xóa lịch sử |
| GET | `/admin/sessions` | List sessions (admin) |
| GET | `/admin/history/{session_id}` | Xem history (admin) |

### NDJSON Stream Format

```json
{"type": "token", "content": "Xin chào..."}
{"type": "token", "content": " đây là câu trả lời"}
{"type": "sources", "content": [{"title": "...", "url": "...", "score": 0.85}]}
{"type": "follow_up", "content": ["Câu hỏi 1?", "Câu hỏi 2?"]}
{"type": "done"}
```

---

## Cài đặt và chạy

### Yêu cầu

- Python 3.10+
- Redis server (chạy local hoặc Docker)
- Ollama (chạy local với `qwen2.5:1.5b`)

### Cài đặt

```bash
pip install -r requirements.txt
```

### Chạy backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### Chạy frontend

```bash
cd frontend
streamlit run app.py
```

### Tài khoản mặc định

| User | Password | Role |
|------|----------|------|
| `phantrongnguyen0618@gmail.com` | `123` | user |
| `smoking` | `456` | user |
| `admin` | `admin123` | admin |

---

## License

MIT
