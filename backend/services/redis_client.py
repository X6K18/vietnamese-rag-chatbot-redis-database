import redis
import json
import hashlib
import time as time_module
from backend.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL


def create_redis_connection(max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                socket_connect_timeout=5, socket_timeout=5,
                retry_on_timeout=True, decode_responses=False
            )
            client.ping()
            return client
        except redis.ConnectionError as e:
            if attempt < max_retries - 1:
                time_module.sleep(retry_delay)
            else:
                print(f"⚠️ Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT} — {e}")
                return None
        except Exception as e:
            print(f"⚠️ Unexpected error connecting to Redis: {e}")
            return None
    return None


redis_client = create_redis_connection()

def _is_redis_available():
    if redis_client is None:
        return False
    try:
        redis_client.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError, AttributeError):
        return False

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str, role: str = "user") -> dict:
    if not _is_redis_available():
        return {"success": False, "message": f"Redis không khả dụng tại {REDIS_HOST}:{REDIS_PORT}"}
    key = f"user:{username}"
    if redis_client.exists(key):
        return {"success": False, "message": "Tên đăng nhập đã tồn tại"}
    user_data = {
        "username": username,
        "password": hash_password(password),
        "role": role,
        "created_at": json.dumps({"time": time_module.time()})
    }
    redis_client.hset(key, mapping=user_data)
    return {"success": True, "message": "Đăng ký thành công"}

def authenticate_user(username: str, password: str):
    if not _is_redis_available():
        return None
    key = f"user:{username}"
    if not redis_client.exists(key):
        return None
    stored = redis_client.hgetall(key)
    stored_password = stored.get(b"password", b"").decode()
    if stored_password == hash_password(password):
        return stored.get(b"role", b"user").decode()
    return None

def user_exists(username: str) -> bool:
    if not _is_redis_available():
        return False
    return redis_client.exists(f"user:{username}")

def get_user_role(username: str) -> str:
    if not _is_redis_available():
        return None
    key = f"user:{username}"
    if redis_client.exists(key):
        return redis_client.hget(key, "role").decode()
    return None

def get_chat_history(session_id: str, max_messages=10):
    if not _is_redis_available():
        return []
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
    if not _is_redis_available():
        return
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
    if not _is_redis_available():
        return
    redis_client.delete(f"chat:{session_id}")
    redis_client.delete(f"chat_summary:{session_id}")

def cache_embedding(query: str, embedding: list):
    if not _is_redis_available():
        return
    key = f"emb:{query}"
    redis_client.setex(key, REDIS_TTL, json.dumps(embedding))

def get_cached_embedding(query: str):
    if not _is_redis_available():
        return None
    data = redis_client.get(f"emb:{query}")
    return json.loads(data) if data else None
