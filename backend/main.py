# backend/main.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json

from backend.config import REDIS_HOST, REDIS_PORT, REDIS_DB
from backend.services.classifier import TextClassifier
from backend.services.rag import RAG
from backend.services.llm import generate_stream
from backend.services.redis_client import (
    redis_client,
    add_chat_message,
    get_chat_history,
    clear_history,
)

# ---------- Khởi tạo FastAPI và các service ----------
app = FastAPI(title="RAG Chatbot API", description="API cho chatbot RAG với Redis và streaming")

# Khởi tạo các thành phần ML (chỉ một lần)
classifier = TextClassifier()
retriever = RAG()

# ---------- Pydantic models ----------
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

# ---------- Cấu hình admin (token tĩnh, có thể thay bằng JWT) ----------
ADMIN_TOKEN = "admin-secret-123"   # Nên đặt trong biến môi trường

def verify_admin(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return True

# ---------- API Endpoints ----------

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Nhận câu hỏi, trả về câu trả lời dạng streaming text/plain.
    Lưu lịch sử hội thoại vào Redis sau khi có đủ câu trả lời.
    """
    # 1. Phân loại chủ đề
    label = classifier.predict(req.message)
    
    # 2. Truy xuất tài liệu liên quan (RAG)
    docs = retriever.search(req.message, category=label, k=3)
    
    # 3. Lưu tin nhắn người dùng vào Redis
    add_chat_message(req.session_id, "user", req.message)
    
    # 4. Tạo generator stream và lưu câu trả lời
    async def generate():
        full_answer = ""
        async for token in generate_stream(req.message, docs, req.session_id):
            full_answer += token
            yield token
        # Lưu toàn bộ câu trả lời vào Redis
        add_chat_message(req.session_id, "assistant", full_answer)
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/chat/sync")
async def chat_sync(req: ChatRequest):
    """Phiên bản đồng bộ (không stream), trả về JSON"""
    label = classifier.predict(req.message)
    docs = retriever.search(req.message, category=label, k=3)
    add_chat_message(req.session_id, "user", req.message)
    
    # Thu thập toàn bộ câu trả lời (không stream)
    full_answer = ""
    async for token in generate_stream(req.message, docs, req.session_id):
        full_answer += token
    add_chat_message(req.session_id, "assistant", full_answer)
    
    return {"answer": full_answer, "category": label, "sources": docs}

@app.get("/chat/history")
async def get_history(session_id: str):
    """Lấy lịch sử chat của một session (dành cho user)"""
    history = get_chat_history(session_id, max_messages=50)
    return {"session_id": session_id, "history": history}

@app.delete("/chat/history")
async def delete_history(session_id: str):
    """Xóa toàn bộ lịch sử chat của session"""
    clear_history(session_id)
    return {"status": "ok", "session_id": session_id}

# ---------- API dành cho Admin ----------
@app.get("/admin/sessions", dependencies=[Depends(verify_admin)])
async def admin_list_sessions():
    """Liệt kê tất cả các session_id có lịch sử chat trong Redis"""
    keys = redis_client.keys("chat:*")
    sessions = [key.decode().replace("chat:", "") for key in keys]
    return {"sessions": sessions}

@app.get("/admin/history/{session_id}", dependencies=[Depends(verify_admin)])
async def admin_get_history(session_id: str):
    """Admin xem toàn bộ lịch sử của một session (không giới hạn số tin nhắn)"""
    history = get_chat_history(session_id, max_messages=1000)
    return {"session_id": session_id, "history": history}

# ---------- Health check ----------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "redis": redis_client.ping()}