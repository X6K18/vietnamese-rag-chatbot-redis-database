from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json

from backend.config import REDIS_HOST, REDIS_PORT, REDIS_DB
from backend.services.classifier import TextClassifier
from backend.services.rag import RAG
from backend.services.llm import generate_stream, expand_query, generate_follow_up, has_relevant_docs
from backend.services.redis_client import (
    redis_client,
    add_chat_message,
    get_chat_history,
    clear_history,
)

app = FastAPI(title="RAG Chatbot API", description="API cho chatbot RAG với Redis và streaming")

classifier = TextClassifier()
retriever = RAG()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

ADMIN_TOKEN = "admin-secret-123"

def verify_admin(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return True

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    label = classifier.predict(req.message)
    expanded_query = expand_query(req.message, label)
    docs = retriever.search(expanded_query, category=label, k=3)

    relevant = has_relevant_docs(docs)
    llm_docs = docs if relevant else []

    add_chat_message(req.session_id, "user", req.message)

    async def generate():
        full_answer = ""
        async for token in generate_stream(req.message, llm_docs, req.session_id, category=label):
            full_answer += token
            yield json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"

        add_chat_message(req.session_id, "assistant", full_answer)

        sources = [
            {
                "title": d["title"],
                "source": d["source"],
                "url": d.get("url", ""),
                "score": round(d["score"], 3),
            }
            for d in docs
        ]
        yield json.dumps({"type": "sources", "content": sources}, ensure_ascii=False) + "\n"

        follow_ups = await generate_follow_up(req.message, full_answer, docs, req.session_id)
        if follow_ups:
            yield json.dumps({"type": "follow_up", "content": follow_ups}, ensure_ascii=False) + "\n"

        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.post("/chat/sync")
async def chat_sync(req: ChatRequest):
    label = classifier.predict(req.message)
    expanded_query = expand_query(req.message, label)
    docs = retriever.search(expanded_query, category=label, k=3)

    relevant = has_relevant_docs(docs)
    llm_docs = docs if relevant else []

    add_chat_message(req.session_id, "user", req.message)

    full_answer = ""
    async for token in generate_stream(req.message, llm_docs, req.session_id, category=label):
        full_answer += token
    add_chat_message(req.session_id, "assistant", full_answer)

    sources = [
        {
            "title": d["title"],
            "source": d["source"],
            "url": d.get("url", ""),
            "score": round(d["score"], 3),
        }
        for d in docs
    ]
    follow_ups = await generate_follow_up(req.message, full_answer, docs, req.session_id)

    return {
        "answer": full_answer,
        "category": label,
        "sources": sources,
        "follow_up": follow_ups,
        "from_knowledge": not relevant,
    }

@app.get("/chat/history")
async def get_history(session_id: str):
    history = get_chat_history(session_id, max_messages=50)
    return {"session_id": session_id, "history": history}

@app.delete("/chat/history")
async def delete_history(session_id: str):
    clear_history(session_id)
    return {"status": "ok", "session_id": session_id}

@app.get("/admin/sessions", dependencies=[Depends(verify_admin)])
async def admin_list_sessions():
    keys = redis_client.keys("chat:*")
    sessions = [key.decode().replace("chat:", "") for key in keys]
    return {"sessions": sessions}

@app.get("/admin/history/{session_id}", dependencies=[Depends(verify_admin)])
async def admin_get_history(session_id: str):
    history = get_chat_history(session_id, max_messages=1000)
    return {"session_id": session_id, "history": history}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "redis": redis_client.ping()}
