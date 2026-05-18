# 🚀 Vietnamese Realtime RAG Chatbot with Redis + PhoBERT + Ollama

## 📌 Tổng Quan Dự Án

Đây là phiên bản nâng cấp của hệ thống Vietnamese RAG Chatbot hiện tại.

Hệ thống mới được cải tiến theo hướng:

* Realtime AI System
* Redis Realtime Database
* Semantic Retrieval
* Session Memory
* Embedding Cache
* Docker Deployment
* Production Architecture

Công nghệ sử dụng:

* PhoBERT
* Sentence Transformers
* Redis
* FAISS
* Ollama
* FastAPI
* Streamlit
* Docker

---

# 🧠 Kiến Trúc Mới

```text
                ┌────────────────────┐
                │     Frontend       │
                │    Streamlit UI    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │      FastAPI       │
                │    Backend API     │
                └─────────┬──────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌───────────────────┐          ┌────────────────────┐
│ Redis Chat Memory │          │ Redis Embedding    │
│ Session History   │          │ Cache              │
└───────────────────┘          └────────────────────┘
         │                                 │
         └────────────────┬────────────────┘
                          ▼
                ┌────────────────────┐
                │ PhoBERT Classifier │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │  FAISS Retrieval   │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │    Ollama LLM      │
                │   qwen2.5:1.5b     │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │  Streaming Answer  │
                └────────────────────┘
```

---

# 📂 Cấu Trúc Thư Mục Mới

```text
project/
│
├── app/
│   ├── frontend/
│   │   └── streamlit_app.py
│   │
│   ├── backend/
│   │   ├── api.py
│   │   ├── rag.py
│   │   ├── llm.py
│   │   ├── utils.py
│   │   ├── redis_manager.py
│   │   ├── embedding_cache.py
│   │   └── auth.py
│   │
│   ├── models/
│   │   ├── phobert_model/
│   │   ├── phobert_tokenizer/
│   │   └── label_encoder.joblib
│   │
│   ├── data/
│   │   ├── faiss.index
│   │   ├── data.pkl
│   │   └── news_dataset.csv
│   │
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── requirements.txt
│
└── README.md
```

---

# 🔥 Redis Integration

## Redis được dùng cho:

| Feature         | Vai trò                |
| --------------- | ---------------------- |
| Chat Memory     | Lưu lịch sử hội thoại  |
| Embedding Cache | Cache vector embedding |
| Session Store   | Quản lý user session   |
| Realtime Layer  | Tăng tốc inference     |
| Queue System    | Background processing  |

---

# 🧩 redis_manager.py

```python
import redis
import json


class RedisManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host="redis",
            port=6379,
            decode_responses=True
        )

    # =========================
    # CHAT HISTORY
    # =========================

    def save_message(self, session_id, role, content):
        key = f"chat:{session_id}"

        message = {
            "role": role,
            "content": content
        }

        self.redis_client.rpush(key, json.dumps(message))

    def get_history(self, session_id):
        key = f"chat:{session_id}"

        messages = self.redis_client.lrange(key, 0, -1)

        return [json.loads(m) for m in messages]

    def clear_history(self, session_id):
        key = f"chat:{session_id}"
        self.redis_client.delete(key)

    # =========================
    # EMBEDDING CACHE
    # =========================

    def get_embedding(self, text):
        key = f"embedding:{text}"

        value = self.redis_client.get(key)

        if value:
            return json.loads(value)

        return None

    def save_embedding(self, text, embedding):
        key = f"embedding:{text}"

        self.redis_client.set(
            key,
            json.dumps(embedding)
        )
```

---

# 🧠 embedding_cache.py

```python
from sentence_transformers import SentenceTransformer
from redis_manager import RedisManager


class EmbeddingCache:
    def __init__(self):
        self.model = SentenceTransformer(
            "keepitreal/vietnamese-sbert"
        )

        self.redis = RedisManager()

    def encode(self, text):

        cached = self.redis.get_embedding(text)

        if cached:
            return cached

        embedding = self.model.encode([text])[0].tolist()

        self.redis.save_embedding(text, embedding)

        return embedding
```

---

# 🔎 rag.py (Realtime Version)

```python
import faiss
import pickle
import numpy as np

from embedding_cache import EmbeddingCache


class RAG:
    def __init__(self, index_path, data_path):

        self.index = faiss.read_index(index_path)

        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.encoder = EmbeddingCache()

    def search(self, query, category=None, k=3):

        q_emb = self.encoder.encode(query)

        q_emb = np.array([q_emb]).astype("float32")

        D, I = self.index.search(q_emb, k * 5)

        results = []

        for idx in I[0]:

            item = self.data[idx]

            if category:
                if item["category"] != category:
                    continue

            results.append(item)

            if len(results) == k:
                break

        return results
```

---

# 🤖 llm.py (Improved Prompt)

```python
import ollama


def build_prompt(query, docs, history=None):

    context = "\n\n".join([
        f"[{i+1}] {doc['text']}"
        for i, doc in enumerate(docs)
    ])

    history_text = ""

    if history:
        history_text += "## CHAT HISTORY\n"

        for msg in history[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    return f"""
Bạn là trợ lý AI tiếng Việt.

QUY TẮC:
- Chỉ sử dụng context.
- Không được bịa thông tin.
- Nếu không có dữ liệu thì nói không tìm thấy.

{history_text}

## CONTEXT
{context}

## QUESTION
{query}

## ANSWER
"""


def generate_answer(query, docs, history=None):

    prompt = build_prompt(
        query=query,
        docs=docs,
        history=history
    )

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    return response["message"]["content"]
```

---

# ⚡ api.py (FastAPI Backend)

```python
from fastapi import FastAPI
from pydantic import BaseModel

from rag import RAG
from llm import generate_answer
from redis_manager import RedisManager

app = FastAPI()

redis_manager = RedisManager()

rag = RAG(
    index_path="data/faiss.index",
    data_path="data/data.pkl"
)


class ChatRequest(BaseModel):
    session_id: str
    query: str


@app.post("/chat")
def chat(request: ChatRequest):

    history = redis_manager.get_history(
        request.session_id
    )

    docs = rag.search(request.query)

    answer = generate_answer(
        query=request.query,
        docs=docs,
        history=history
    )

    redis_manager.save_message(
        request.session_id,
        "user",
        request.query
    )

    redis_manager.save_message(
        request.session_id,
        "assistant",
        answer
    )

    return {
        "answer": answer,
        "sources": docs
    }
```

---

# 🎨 streamlit_app.py

```python
import streamlit as st
import requests
import uuid


API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="Vietnamese AI Assistant",
    page_icon="🧠",
    layout="wide"
)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("🧠 Vietnamese Realtime RAG Chatbot")


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if query := st.chat_input("Nhập câu hỏi..."):

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    response = requests.post(
        API_URL,
        json={
            "session_id": st.session_state.session_id,
            "query": query
        }
    )

    data = response.json()

    answer = data["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
```

---

# 🐳 Dockerfile

```dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# 🐳 docker-compose.yml

```yaml
version: '3.9'

services:

  redis:
    image: redis:latest
    container_name: redis_server
    ports:
      - "6379:6379"

  backend:
    build: .
    container_name: rag_backend
    ports:
      - "8000:8000"
    depends_on:
      - redis

  frontend:
    build: .
    command: streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - backend
```

---

# 📦 requirements.txt

```text
streamlit
fastapi
uvicorn
redis
faiss-cpu
numpy
pandas
torch
transformers
sentence-transformers
scikit-learn
joblib
ollama
requests
python-dotenv
```

---

# 🔥 Những nâng cấp lớn so với phiên bản cũ

| Phiên bản cũ        | Phiên bản mới           |
| ------------------- | ----------------------- |
| Streamlit only      | FastAPI + Streamlit     |
| Session local       | Redis realtime session  |
| Không cache         | Redis embedding cache   |
| Single architecture | Production architecture |
| Chậm khi encode     | Realtime retrieval      |
| Không scale được    | Docker scalable         |
| Basic chatbot       | Realtime AI System      |

---

# 📈 Hướng phát triển tiếp theo

## 1. Redis Vector Search

Thay FAISS bằng:

* Redis Stack
* RediSearch
* Vector Similarity

---

## 2. Celery + Redis Queue

Background processing:

* upload PDF
* chunking
* embedding
* indexing

---

## 3. Streaming Response

Cho chatbot trả lời realtime như ChatGPT.

---

## 4. Authentication Database

Hiện tại đang hardcode.

Nên nâng cấp:

* PostgreSQL
* JWT Authentication
* OAuth2

---

# 🎯 Giá trị CV/GitHub

Dự án lúc này sẽ có các keyword rất mạnh:

* NLP
* RAG
* Redis
* Realtime AI
* Semantic Search
* PhoBERT
* Ollama
* Vector Database
* Docker
* FastAPI
* Streamlit
* AI Backend
* MLOps
* Distributed Systems

---

# 🏁 Kết luận

Đây không còn là chatbot NLP cơ bản nữa.

Sau khi thêm Redis + FastAPI + Docker:

Hệ thống đã trở thành:

✅ Realtime Vietnamese AI Assistant

✅ Production-style RAG System

✅ Mini AI Infrastructure Project

✅ Phù hợp portfolio AI Engineer / Data Engineer / MLOps
