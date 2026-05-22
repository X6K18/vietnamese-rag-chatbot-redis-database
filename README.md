<div align="center">

# Vietnamese Realtime RAG Chatbot 🇻🇳

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-00a393?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Redis](https://img.shields.io/badge/Redis-7.0+-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5:1.5b-000?logo=ollama&logoColor=white)](https://ollama.ai/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-2596BE?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![PhoBERT](https://img.shields.io/badge/PhoBERT-fine--tuned-FF6F00?logo=huggingface&logoColor=white)](https://github.com/VinAIResearch/PhoBERT)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**A real-time Vietnamese question-answering chatbot powered by Retrieval-Augmented Generation (RAG), combining PhoBERT topic classification, FAISS vector search, Redis conversational memory, and local LLM inference via Ollama.**

[Features](#features) • [Architecture](#architecture) • [Tech Stack](#tech-stack) • [Getting Started](#getting-started) • [API Reference](#api-reference) • [Project Structure](#project-structure) • [Testing](#testing) • [License](#license)

</div>

---

## Features

- **🇻🇳 Vietnamese-First** — Fine-tuned PhoBERT for 11-category topic classification; Sentence-Transformer embeddings (`keepitreal/vietnamese-sbert`) optimized for Vietnamese text
- **📡 Real-Time Streaming** — Token-by-token answers via NDJSON protocol, similar to ChatGPT
- **🔍 RAG with Source Citations** — Retrieves relevant context from 36,506 document chunks (6,256 news articles) and cites inline sources with titles, URLs, and relevance scores
- **🧠 Intelligent Retrieval Pipeline** — Query expansion with predicted category → FAISS L2 search → category filtering → MMR reranking (λ=0.5) → relevance threshold check (≥0.3)
- **💬 Conversational Memory** — Redis-backed per-session chat history with auto-summarization after 20+ messages
- **⚡ Embedding Cache** — Redis-based query embedding cache (1-hour TTL) reduces embedding calls by 50–70%
- **💡 Follow-up Suggestions** — LLM generates 2–3 contextual follow-up questions after each answer
- **🔐 User Authentication** — Redis-based registration/login with hashed passwords and admin role
- **📊 Operations Dashboard** — Standalone HTML/CSS/JS dashboard for pipeline visualization, document stats, system health, and model insight demos
- **🎯 Smart Fallback** — When no relevant documents found (max score < 0.3), falls back to Ollama's own knowledge

---

## Architecture

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

### Pipeline Flow

1. **User** sends a Vietnamese question via the Streamlit UI
2. **FastAPI** receives the request and sends it to the PhoBERT classifier for topic prediction
3. **Query Expansion** — The question is augmented with the predicted category
4. **FAISS Retrieval** — Sentence-Transformer generates a 768-dim embedding; FAISS searches top-15 (k×5) nearest vectors
5. **Category Filtering** — Results are filtered by the predicted topic
6. **MMR Reranking** — Maximum Marginal Relevance (λ=0.5) selects top-3 with diversity
7. **Relevance Check** — If max score < 0.3, fallback to Ollama's own knowledge (no RAG context)
8. **Ollama Generation** — Qwen2.5:1.5b generates a streaming answer with RAG context or fallback prompt
9. **Redis Persistence** — Messages saved to Redis List; auto-summarized if session exceeds 20 messages
10. **NDJSON Streaming** — Response streamed as events: `token` → `sources` → `follow_up` → `done`

---

## Tech Stack

| Technology | Role | Rationale |
|------------|------|-----------|
| **FastAPI** | Backend API framework | Async, high-performance, auto-generated docs |
| **Streamlit** | Frontend UI | Python-native, rapid development, clean UI |
| **Redis** | Chat memory & embedding cache | Ultra-fast in-memory database, ideal for real-time |
| **PhoBERT** | Topic classification | SOTA Vietnamese NLP model (VinAI Research) |
| **FAISS** | Vector similarity search | ~3ms search over 36K 768-dim vectors |
| **Sentence-Transformers** | Text embedding | `keepitreal/vietnamese-sbert` for Vietnamese |
| **Ollama** | LLM inference | Local Qwen2.5:1.5b, no GPU required |
| **NumPy** | Numerical computation | Vector and matrix operations |
| **scikit-learn** | Label encoding | Topic label encoding/decoding |
| **PyTorch** | Deep learning framework | PhoBERT model inference |

### LLM Generation Parameters

| Parameter | Value |
|-----------|-------|
| Model | `qwen2.5:1.5b` |
| Temperature | 0.3 |
| top_p | 0.9 |
| top_k | 40 |
| Context window | 1024 tokens |
| Repeat penalty | 1.1 |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Redis server (local or Docker)
- [Ollama](https://ollama.ai/) with `qwen2.5:1.5b` pulled

### Installation

```bash
pip install -r requirements.txt

# Pull the Ollama model
ollama pull qwen2.5:1.5b
```

### Running the System

The system requires three terminals:

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start the FastAPI backend
uvicorn backend.main:app --reload --port 8000

# Terminal 3: Start the Streamlit frontend
cd frontend
streamlit run app.py
```

The Streamlit UI will be available at `http://localhost:8501`.

> **Operations Dashboard** — Open `dashboard/index.html` in a browser (static HTML/CSS/JS, no server required).

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check with Redis ping |
| POST | `/chat/stream` | Streaming chat (NDJSON) |
| POST | `/chat/sync` | Synchronous chat (JSON) |
| GET | `/chat/history?session_id=...` | Get chat history |
| DELETE | `/chat/history?session_id=...` | Delete chat history |
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login (returns session_id) |
| GET | `/admin/sessions` | List all sessions (admin) |
| GET | `/admin/history/{session_id}` | View session history (admin) |

### NDJSON Stream Format

```
{"type":"token","content":"Xin chào bạn"}
{"type":"token","content":", tôi có thể giúp gì?"}
{"type":"sources","content":[{"title":"...","url":"...","score":0.85}]}
{"type":"follow_up","content":["Câu hỏi 1?","Câu hỏi 2?"]}
{"type":"done"}
```

### Default Accounts

| Username | Password | Role |
|----------|----------|------|
| `phantrongnguyen0618@gmail.com` | `123` | user |
| `smoking` | `456` | user |
| `admin` | `admin123` | admin |

---

## Project Structure

```
├── backend/
│   ├── main.py                  # FastAPI server, endpoints, CORS
│   ├── config.py                # Configuration (paths, Redis, models)
│   ├── models.py                # (reserved)
│   ├── utils.py                 # Text preprocessing utilities
│   └── services/
│       ├── classifier.py        # PhoBERT topic classification
│       ├── rag.py               # FAISS retrieval + MMR reranking
│       ├── llm.py               # Ollama prompt building + streaming
│       └── redis_client.py      # Redis chat history + embedding cache
├── frontend/
│   └── app.py                   # Streamlit chat UI
├── dashboard/                   # Operations dashboard (HTML/CSS/JS)
├── models/
│   ├── phobert_model/           # Fine-tuned PhoBERT weights
│   ├── phobert_tokenizer/       # PhoBERT tokenizer (BPE)
│   └── label_encoder.joblib     # Label encoder (11 categories)
├── data/
│   ├── news_dataset.csv         # 6,256 original news articles (~32 MB)
│   ├── data.pkl                 # 36,506 pre-chunked documents (~33 MB)
│   └── faiss.index              # FAISS index, 36,506×768-dim (~112 MB)
├── tests/
│   ├── test_redis.py            # Redis connection test
│   └── test_api.py              # API endpoint tests
├── docs/                        # Documentation and reference materials
├── reports/                     # Project reports and presentations
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── run.md                       # Quick-start instructions
├── LICENSE                      # Apache License 2.0
└── .gitignore
```

---

## Configuration

Key configuration in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `REDIS_HOST` | `127.0.0.1` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_TTL` | `3600` | Embedding cache TTL (1 hour) |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model name |
| `EMBED_MODEL` | `keepitreal/vietnamese-sbert` | Embedding model |
| `DEVICE` | auto (cuda/cpu) | Inference device |
| `RELEVANCE_THRESHOLD` | `0.3` | Minimum relevance score for RAG |

---

## Testing

```bash
# Test Redis connection
python tests/test_redis.py

# Test API (requires running backend)
python tests/test_api.py
```

---

## Data Scale

| Metric | Value |
|--------|-------|
| News articles | 6,256 |
| Document chunks | 36,506 |
| Embedding dimension | 768 |
| FAISS index type | IndexFlatL2 (brute-force) |
| Classification categories | 11 |
| Chunk size | ~512 tokens |
| Chunk overlap | ~128–256 tokens |

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ for the Vietnamese NLP community
</div>
