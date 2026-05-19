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
  - [Transformer và Cơ chế Self-Attention](#6-transformer-và-cơ-chế-self-attention)
  - [Bi-Encoder vs Cross-Encoder](#7-bi-encoder-vs-cross-encoder)
  - [Chunking và Sliding Window](#8-chunking-và-sliding-window)
  - [Softmax và Temperature Scaling](#9-softmax-và-temperature-scaling)
  - [Prompt Engineering và Chain-of-Thought](#10-prompt-engineering-và-chain-of-thought)
  - [Redis Data Structures cho AI System](#11-redis-data-structures-cho-ai-system)
  - [NDJSON Streaming Protocol](#12-ndjson-streaming-protocol)
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

### 6. Transformer và Cơ chế Self-Attention

**Transformer** (Vaswani et al., 2017) là kiến trúc nền tảng của PhoBERT và toàn bộ mô hình ngôn ngữ hiện đại. Thành phần cốt lõi là cơ chế **Self-Attention**.

#### Scaled Dot-Product Attention

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Trong đó:
- $Q$ (Query): vector truy vấn
- $K$ (Key): vector khóa
- $V$ (Value): vector giá trị
- $d_k$: chiều của key vector
- $\sqrt{d_k}$: scaling factor, tránh gradient quá nhỏ khi $d_k$ lớn

#### Multi-Head Attention

Thay vì một attention duy nhất, Transformer dùng nhiều "đầu" song song:

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

Mỗi đầu học một khía cạnh khác nhau của mối quan hệ giữa các từ.

#### Positional Encoding

Vì Transformer không có cấu trúc tuần tự như RNN, cần thêm thông tin vị trí:

$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

**Ứng dụng trong project**: PhoBERT kế thừa kiến trúc Transformer với 12 layer, 768 hidden size, 12 attention heads. Qwen2.5:1.5b sử dụng Transformer decoder với 1.5B tham số.

---

### 7. Bi-Encoder vs Cross-Encoder

Hai kiến trúc khác nhau cho bài toán similarity/retrieval:

#### Bi-Encoder (dùng trong embedding & FAISS)

```
Câu A → BERT → Vector A (768-d)
Câu B → BERT → Vector B (768-d)
                   ↓
            Similarity(A, B)
```

- **Đặc điểm**: Mã hóa độc lập từng câu, similarity tính sau
- **Ưu điểm**: Có thể pre-compute và index hàng triệu vector, search siêu nhanh
- **Nhược điểm**: Mất tương tác chéo giữa hai câu
- **Dùng trong project**: `keepitreal/vietnamese-sbert` (Bi-Encoder) để tạo vector cho toàn bộ 36,506 chunks

#### Cross-Encoder (dùng trong reranking)

```
[Câu A; Câu B] → BERT → Similarity score
```

- **Đặc điểm**: Ghép hai câu làm đầu vào, tính attention chéo
- **Ưu điểm**: Chính xác hơn Bi-Encoder vì có tương tác chéo
- **Nhược điểm**: Chậm, không thể pre-compute, phải chạy cho từng cặp
- **Dùng trong project**: Dự kiến cho reranking nâng cao sau này

#### So sánh hiệu năng

| Tiêu chí | Bi-Encoder | Cross-Encoder |
|----------|------------|---------------|
| Tốc độ | Index 1M vector < 1s | 1 cặp ~ 10ms |
| Độ chính xác | ~85% | ~95% |
| Scale | ∞ (pre-compute) | Limited (runtime) |
| Use case | Retrieval | Reranking top-k |

---

### 8. Chunking và Sliding Window

**Chunking** là quá trình chia văn bản dài thành các đoạn nhỏ hơn để embedding và retrieval hiệu quả.

#### Vấn đề
- Mô hình embedding có **max sequence length** (thường 512 tokens)
- Bài báo dài có thể > 2000 tokens → phải chunk
- Câu hỏi ngắn cần so sánh với đoạn ngắn, không phải cả bài

#### Chiến lược chunking trong project

```
Bài báo gốc (1000 tokens)
  |
  +--- Chunk 1 (tokens 0-512)  → vector 1
  +--- Chunk 2 (tokens 256-768) → vector 2  (overlap 256)
  +--- Chunk 3 (tokens 512-1000) → vector 3
```

Thông số:
- **Chunk size**: ~512 tokens
- **Overlap**: ~128-256 tokens (tránh mất thông tin giữa các chunk)
- **Kết quả**: 6,256 bài báo → 36,506 chunks (~5.8 chunk/bài)

#### Vì sao có overlap?

$$ \text{Thông tin mất} \propto \frac{\text{chunk\_size} - \text{overlap}}{\text{chunk\_size}} $$

Với overlap, các câu ở biên giới giữa hai chunk được giữ lại, tăng khả năng retrieval thành công.

---

### 9. Softmax và Temperature Scaling

#### Softmax Function

Chuyển logits thành phân phối xác suất:

$$ P(y=c|x) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}} $$

Trong đó $z_i$ là logit (đầu ra raw của model) cho lớp $i$.

**Ứng dụng trong project**:
- **PhoBERT Classifier**: Softmax trên 11 logits để ra xác suất cho mỗi chủ đề
- **Ollama LLM**: Softmax trên toàn bộ từ vựng (~152,000 tokens của Qwen2.5)

#### Temperature Scaling

$$ P(y=c|x) = \frac{e^{z_c / T}}{\sum_{j=1}^{C} e^{z_j / T}} $$

| Temperature $T$ | Hiệu ứng | Khi nào dùng |
|:---:|---|---|
| $T \to 0$ | Deterministic, chọn từ có xác suất cao nhất | Trả lời chính xác, factual |
| $T = 1$ | Phân phối gốc | Mặc định |
| $T > 1$ | Phân phối "mềm" hơn, sáng tạo hơn | Sinh câu hỏi, brainstorm |

**Cấu hình trong project**:
- `temperature = 0.3`: Trả lời chính xác, ít sáng tạo cho factual QA
- `temperature = 0.1`: Cũ (cũ hơn), ít biến thể hơn
- `top_p = 0.9`: Nucleus sampling, chỉ chọn từ trong top 90% xác suất
- `top_k = 40`: Chỉ chọn từ trong top 40 tokens có xác suất cao nhất

---

### 10. Prompt Engineering và Chain-of-Thought

**Prompt Engineering** là kỹ thuật thiết kế câu lệnh đầu vào để tối ưu chất lượng đầu ra của LLM.

#### Zero-shot Prompting

```
Câu hỏi: {query}
Trả lời:
```

Dùng trong project cho trường hợp không có tài liệu (fallback).

#### Few-shot Prompting

```
Câu hỏi: "Thủ đô Việt Nam là gì?"
Trả lời: "Hà Nội"

Câu hỏi: "Thủ đô Lào là gì?"
Trả lời: "Viêng Chăn"

Câu hỏi: {query}
Trả lời:
```

#### Chain-of-Thought (CoT) Prompting

Kỹ thuật yêu cầu mô hình "suy nghĩ từng bước" trước khi trả lời:

```
QUY TẮC:
1. PHÂN TÍCH câu hỏi và tài liệu
2. SUY LUẬN từng bước
3. KIỂM TRA thông tin có trong tài liệu không
4. TRẢ LỜI có trích dẫn
```

**Lợi ích của CoT**:
- Giảm hallucination (mô hình phải kiểm tra từng bước)
- Tăng khả năng truy xuất thông tin chính xác
- Có thể trace được lý do tại sao mô hình trả lời như vậy

#### Prompt Template trong project

```
[Hệ thống] - Vai trò, quy tắc
[Lịch sử] - Hội thoại trước đó (nếu có)
[Tài liệu] - Context RAG (nếu có)
[Câu hỏi] - Query hiện tại
[Trả lời] - Đầu ra (cấu trúc với trích dẫn và gợi ý)
```

---

### 11. Redis Data Structures cho AI System

Redis là in-memory database được dùng với 3 cấu trúc dữ liệu chính:

#### List (Chat History)

```
Key: "chat:{session_id}"
Value: ["msg1", "msg2", "msg3", ...]

Operations:
  RPUSH key msg    # Thêm tin nhắn vào cuối
  LRANGE key -N -1 # Lấy N tin nhắn gần nhất
  LTRIM key -N -1  # Giữ lại N tin nhắn cuối
  LLEN key         # Đếm số tin nhắn
```

**Độ phức tạp**: $O(1)$ cho RPUSH, $O(N)$ cho LRANGE (với N là kết quả trả về).

#### String + TTL (Embedding Cache)

```
Key: "emb:{query_text}"
Value: "[0.12, -0.45, 0.78, ...]" (JSON array)
TTL: 3600 giây (1 giờ)

Operations:
  SETEX key TTL value  # Set + expire
  GET key              # Get nếu còn hạn
```

**Cache hit ratio** dự kiến: Các câu hỏi tương tự thường lặp lại, cache giảm 50-70% số lần gọi embedding model.

#### String + TTL (Conversation Summary)

```
Key: "chat_summary:{session_id}"
Value: "User hỏi về giá xăng | Bot trả lời giá dầu thế giới..."
TTL: 3600 giây
```

Kích hoạt khi session có > 20 messages, lưu tóm tắt và trim list để quản lý context window hiệu quả.

---

### 12. NDJSON Streaming Protocol

**NDJSON** (Newline-Delimited JSON) là giao thức streaming được dùng trong endpoint `/chat/stream`.

#### So với các giao thức khác

| Giao thức | Ưu điểm | Nhược điểm |
|-----------|---------|------------|
| **SSE** (`text/event-stream`) | Chuẩn web, EventSource API | Không phức tạp, mỗi event chỉ 1 dòng |
| **NDJSON** (`application/x-ndjson`) | Đơn giản, parse bằng `readline()` | Cần client tự parse |
| **WebSocket** | 2 chiều, realtime | Phức tạp hơn HTTP |
| **Chunked transfer** | Chuẩn HTTP/1.1 | Không cấu trúc |

#### Format

Mỗi dòng là một JSON object với trường `type` và `content`:

```
{"type":"token","content":"Xin chào bạn"}          ← Token stream
{"type":"token","content":", tôi có thể giúp gì?"}  ← Token tiếp theo
{"type":"sources","content":[{"title":"...","url":"..."}]}  ← Sources
{"type":"follow_up","content":["Hỏi A?","Hỏi B?"]}  ← Gợi ý
{"type":"done"}                                      ← Kết thúc
```

#### Xử lý phía client

```python
for line in response.iter_lines(decode_unicode=True):
    if line and line.strip():
        event = json.loads(line)
        if event["type"] == "token":
            answer += event["content"]
        elif event["type"] == "sources":
            sources = event["content"]
        elif event["type"] == "done":
            break
```

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
