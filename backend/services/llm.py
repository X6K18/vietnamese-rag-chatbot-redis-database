import ollama
import asyncio
import json
import re
from backend.config import OLLAMA_MODEL
from backend.services.redis_client import get_chat_history

RELEVANCE_THRESHOLD = 0.3

def _max_score(docs: list) -> float:
    return max((d.get("score", 0) for d in docs), default=0.0)

def has_relevant_docs(docs: list) -> bool:
    return len(docs) > 0 and _max_score(docs) >= RELEVANCE_THRESHOLD

def build_prompt(query: str, docs: list, history=None, category: str = None):
    relevant = has_relevant_docs(docs)

    history_text = ""
    if history:
        history_text = "## LỊCH SỬ HỘI THOẠI\n"
        for msg in history[-6:]:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    category_hint = f"\nChủ đề phát hiện: {category}" if category else ""

    if relevant:
        context = "\n\n".join([
            f"[{i+1}] {doc['text']}\n(Nguồn: {doc['source']} - {doc['title']}{' - ' + doc['url'] if doc.get('url') else ''})"
            for i, doc in enumerate(docs)
        ])
        return f"""Bạn là trợ lý AI tiếng Việt, trả lời dựa trên tài liệu tham khảo.

## QUY TẮC
1. CHỈ dùng thông tin trong TÀI LIỆU THAM KHẢO để trả lời.
2. TRÍCH DẪN nguồn với [1], [2] ngay trong câu trả lời.
3. Nếu tài liệu không đủ, nói rõ "Tài liệu không đề cập đến..." và KHÔNG tự suy diễn.
4. Cuối câu trả lời, đề xuất 2-3 câu hỏi gợi ý ngắn.
{category_hint}

{history_text}
## TÀI LIỆU THAM KHẢO
{context}

## CÂU HỎI
{query}

## TRẢ LỜI (trích dẫn [1], [2] và kèm gợi ý):
"""
    else:
        return f"""Bạn là trợ lý AI tiếng Việt thông minh, có kiến thức sâu rộng.

## QUY TẮC
1. KHÔNG có tài liệu tham khảo cho câu hỏi này. Hãy dùng KIẾN THỨC của bạn để trả lời.
2. Nói rõ "Tôi không tìm thấy tài liệu cụ thể trong cơ sở dữ liệu về [chủ đề này], nhưng dựa trên kiến thức của tôi:" trước khi trả lời.
3. Trả lời chính xác, hữu ích, có cấu trúc.
4. Cuối câu trả lời, đề xuất 2-3 câu hỏi gợi ý ngắn.
{category_hint}

{history_text}
## CÂU HỎI
{query}

## TRẢ LỜI (dùng kiến thức riêng, mở đầu bằng lưu ý và kèm gợi ý):
"""


def expand_query(query: str, category: str = None) -> str:
    if category and category != "general":
        return f"{query} (thuộc chủ đề {category})"
    return query


async def generate_follow_up(query: str, answer: str, docs: list, session_id: str) -> list:
    prompt = f"""
Dựa vào câu hỏi và câu trả lời sau, hãy đề xuất 2-3 câu hỏi gợi ý ngắn gọn (tiếng Việt) mà người dùng có thể hỏi tiếp.

Câu hỏi: {query}
Câu trả lời: {answer[:500]}

Trả lời dưới dạng JSON array: ["câu hỏi 1?", "câu hỏi 2?", "câu hỏi 3?"]
"""
    try:
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.3, "num_ctx": 1024, "repeat_penalty": 1.1}
        )
        content = stream["message"]["content"].strip()
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


async def generate_stream(query: str, docs: list, session_id: str, category: str = None, temperature: float = 0.3):
    history = get_chat_history(session_id, max_messages=10)
    prompt = build_prompt(query, docs, history, category)

    generation_args = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 1024,
            "repeat_penalty": 1.1,
        }
    }

    try:
        stream = ollama.chat(**generation_args)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
    except Exception as e:
        err = str(e).lower()
        is_conn_err = any(kw in err for kw in ["connection", "refused", "econnrefused", "cannot connect", "winerror 10061", "actively refused"])
        is_mem_err = any(kw in err for kw in ["unable to allocate", "out of memory", "cuda out of memory", "cuda_error"])
        if is_conn_err:
            yield f"\n\n⚠️ **Lỗi kết nối**: Không thể kết nối đến Ollama. Hãy đảm bảo Ollama đang chạy (`ollama serve`)."
        elif is_mem_err:
            yield f"\n\n⚠️ **Lỗi bộ nhớ**: Máy không đủ RAM/VRAM để chạy model. Hãy thử:\n1. `ollama pull qwen2.5:0.5b` (model nhẹ hơn)\n2. Đóng bớt ứng dụng khác\n3. Khởi động lại Ollama"
        else:
            yield f"\n\n⚠️ **Lỗi**: {e}"
