import ollama
import asyncio
from backend.config import OLLAMA_MODEL
from backend.services.redis_client import get_chat_history

def build_prompt(query: str, docs: list, history=None):
    context = "\n\n".join([
        f"[{i+1}] {doc['text']}\n(Nguồn: {doc['source']} - {doc['title']})"
        for i, doc in enumerate(docs)
    ])
    history_text = ""
    if history:
        history_text = "## LỊCH SỬ HỘI THOẠI\n"
        for msg in history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"
    return f"""
Bạn là chatbot tiếng Việt thân thiện.

## QUY TẮC
- CHỈ dùng CONTEXT và LỊCH SỬ HỘI THOẠI để trả lời.
- KHÔNG bịa thông tin.
- Nếu không đủ thông tin, nói "Tôi không tìm thấy thông tin phù hợp."

{history_text}
## CONTEXT
{context}

## CÂU HỎI HIỆN TẠI
{query}

## TRẢ LỜI (ngắn gọn, có thể trích dẫn [1], [2]):
"""

async def generate_stream(query: str, docs: list, session_id: str):
    history = get_chat_history(session_id, max_messages=10)
    prompt = build_prompt(query, docs, history)
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.1}
        )
        for chunk in stream:
            yield chunk["message"]["content"]
    except Exception as e:
        yield f"⚠️ Lỗi: {str(e)}"