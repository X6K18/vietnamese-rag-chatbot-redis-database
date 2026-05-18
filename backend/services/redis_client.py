import redis
import json
from backend.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def get_chat_history(session_id: str, max_messages=10):
    key = f"chat:{session_id}"
    summary_key = f"chat_summary:{session_id}"

    summary = redis_client.get(summary_key)
    summary_text = json.loads(summary) if summary else None

    history = redis_client.lrange(key, -max_messages, -1)
    messages = [json.loads(m) for m in history]

    if summary_text and messages:
        messages.insert(0, {
            "role": "system",
            "content": f"[Tóm tắt hội thoại trước đó: {summary_text}]"
        })

    return messages

def add_chat_message(session_id: str, role: str, content: str):
    key = f"chat:{session_id}"
    msg = json.dumps({"role": role, "content": content})
    redis_client.rpush(key, msg)
    redis_client.expire(key, REDIS_TTL)

    length = redis_client.llen(key)
    if length > 20:
        summary_key = f"chat_summary:{session_id}"
        existing = redis_client.get(summary_key)
        old_summary = json.loads(existing) if existing else ""

        old_messages = redis_client.lrange(key, 0, -(11))
        trimmed = []
        for m in old_messages:
            try:
                trimmed.append(json.loads(m))
            except json.JSONDecodeError:
                continue

        if len(trimmed) > 2:
            brief = "; ".join([
                f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content'][:100]}"
                for m in trimmed[-6:]
            ])
            new_summary = f"{old_summary} | {brief}" if old_summary else brief
            if len(new_summary) > 1000:
                new_summary = new_summary[-1000:]
            redis_client.setex(summary_key, REDIS_TTL, json.dumps(new_summary))

            redis_client.ltrim(key, -(11), -1)

def clear_history(session_id: str):
    redis_client.delete(f"chat:{session_id}")
    redis_client.delete(f"chat_summary:{session_id}")

def cache_embedding(query: str, embedding: list):
    key = f"emb:{query}"
    redis_client.setex(key, REDIS_TTL, json.dumps(embedding))

def get_cached_embedding(query: str):
    data = redis_client.get(f"emb:{query}")
    return json.loads(data) if data else None
