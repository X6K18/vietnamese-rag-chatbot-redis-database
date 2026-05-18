import redis
import json
from backend.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def get_chat_history(session_id: str, max_messages=10):
    key = f"chat:{session_id}"
    history = redis_client.lrange(key, -max_messages, -1)
    return [json.loads(m) for m in history]

def add_chat_message(session_id: str, role: str, content: str):
    key = f"chat:{session_id}"
    msg = json.dumps({"role": role, "content": content})
    redis_client.rpush(key, msg)
    redis_client.expire(key, REDIS_TTL)

def clear_history(session_id: str):
    redis_client.delete(f"chat:{session_id}")

def cache_embedding(query: str, embedding: list):
    key = f"emb:{query}"
    redis_client.setex(key, REDIS_TTL, json.dumps(embedding))

def get_cached_embedding(query: str):
    data = redis_client.get(f"emb:{query}")
    return json.loads(data) if data else None