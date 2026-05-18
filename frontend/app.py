import streamlit as st
import requests
import json
from typing import List, Dict, Generator, Optional

API_BASE = "http://localhost:8000"
ADMIN_TOKEN = "admin-secret-123"

VALID_USERS = {
    "phantrongnguyen0618@gmail.com": {"password": "123", "role": "user"},
    "smoking": {"password": "456", "role": "user"},
    "admin": {"password": "admin123", "role": "admin"}
}

def authenticate(username: str, password: str):
    if username in VALID_USERS and VALID_USERS[username]["password"] == password:
        return VALID_USERS[username]["role"]
    return None

def get_chat_history(session_id: str) -> List[Dict]:
    try:
        resp = requests.get(f"{API_BASE}/chat/history", params={"session_id": session_id})
        if resp.status_code == 200:
            return resp.json().get("history", [])
    except Exception as e:
        st.error(f"Không thể tải lịch sử: {e}")
    return []

def delete_chat_history(session_id: str) -> bool:
    try:
        resp = requests.delete(f"{API_BASE}/chat/history", params={"session_id": session_id})
        return resp.status_code == 200
    except Exception as e:
        st.error(f"Lỗi xoá lịch sử: {e}")
        return False

def ask_question_stream(session_id: str, message: str) -> Generator[dict, None, None]:
    try:
        resp = requests.post(
            f"{API_BASE}/chat/stream",
            json={"session_id": session_id, "message": message},
            stream=True,
            timeout=120
        )
        if resp.status_code != 200:
            yield {"type": "error", "content": f"⚠️ Lỗi từ server: {resp.status_code}"}
            return
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield {"type": "error", "content": f"⚠️ Lỗi kết nối: {e}"}

def admin_list_sessions() -> List[str]:
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    try:
        resp = requests.get(f"{API_BASE}/admin/sessions", headers=headers)
        if resp.status_code == 200:
            return resp.json().get("sessions", [])
    except Exception as e:
        st.error(f"Lỗi lấy danh sách session: {e}")
    return []

def admin_get_history(session_id: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    try:
        resp = requests.get(f"{API_BASE}/admin/history/{session_id}", headers=headers)
        if resp.status_code == 200:
            return resp.json().get("history", [])
    except Exception as e:
        st.error(f"Lỗi lấy lịch sử admin: {e}")
    return []

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.session_id = ""
    st.session_state.messages = []

def login_page():
    st.title("Đăng nhập")
    with st.form("login_form"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập")
        if submitted:
            role = authenticate(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                st.session_state.session_id = f"user_{hash(username)}"
                if role == "user":
                    history = get_chat_history(st.session_state.session_id)
                    st.session_state.messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in history
                    ]
                else:
                    st.session_state.messages = []
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu")

def render_message(msg: dict):
    role = msg["role"]
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    follow_up = msg.get("follow_up", [])

    with st.chat_message(role):
        st.markdown(content)

        if sources:
            with st.expander("Nguồn tham khảo"):
                for i, s in enumerate(sources):
                    url = s.get("url", "")
                    title = s.get("title", "Không có tiêu đề")
                    source = s.get("source", "")
                    score = s.get("score", 0)
                    st.markdown(
                        f"**{i+1}. [{title}]({url})**  \n"
                        f"📂 {source} • Độ liên quan: {score:.0%}"
                        if url else
                        f"**{i+1}. {title}**  \n"
                        f"📂 {source} • Độ liên quan: {score:.0%}"
                    )

        if follow_up and role == "assistant":
            st.markdown("---")
            st.markdown("**Câu hỏi gợi ý:**")
            cols = st.columns(len(follow_up))
            for i, q in enumerate(follow_up):
                if cols[i].button(q, key=f"fu_{hash(q)}_{len(st.session_state.messages)}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

def chat_interface():
    with st.sidebar:
        st.markdown(f"Người dùng: `{st.session_state.username}`")
        if st.button("Đăng xuất"):
            logout()
        st.markdown("---")
        st.markdown("Tuỳ chỉnh")
        if st.button("Xoá lịch sử chat"):
            if delete_chat_history(st.session_state.session_id):
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Không thể xoá lịch sử")
        st.markdown("---")
        st.info("Bot phân tích chủ đề, truy xuất tài liệu liên quan và trả lời có trích dẫn nguồn.")

    for msg in st.session_state.messages:
        render_message(msg)

    if st.session_state.get("pending_query"):
        query = st.session_state.pop("pending_query")
    else:
        query = st.chat_input("Nhập câu hỏi của bạn...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            sources = []
            follow_up = []

            for event in ask_question_stream(st.session_state.session_id, query):
                if event["type"] == "token":
                    full_response += event["content"]
                    placeholder.markdown(full_response + "▌")
                elif event["type"] == "sources":
                    sources = event["content"]
                elif event["type"] == "follow_up":
                    follow_up = event["content"]
                elif event["type"] == "error":
                    full_response = event["content"]
                    placeholder.markdown(full_response)
                elif event["type"] == "done":
                    break

            msg = {"role": "assistant", "content": full_response}
            if sources:
                msg["sources"] = sources
            if follow_up:
                msg["follow_up"] = follow_up

            placeholder.markdown(full_response)

            if sources:
                with st.expander("Nguồn tham khảo"):
                    for i, s in enumerate(sources):
                        url = s.get("url", "")
                        title = s.get("title", "Không có tiêu đề")
                        source = s.get("source", "")
                        score = s.get("score", 0)
                        st.markdown(
                            f"**{i+1}. [{title}]({url})**  \n"
                            f"📂 {source} • Độ liên quan: {score:.0%}"
                            if url else
                            f"**{i+1}. {title}**  \n"
                            f"📂 {source} • Độ liên quan: {score:.0%}"
                        )

            if follow_up:
                st.markdown("---")
                st.markdown("**Câu hỏi gợi ý:**")
                cols = st.columns(len(follow_up))
                for i, q in enumerate(follow_up):
                    key = f"fu_{hash(q)}_{len(st.session_state.messages)}"
                    if cols[i].button(q, key=key, use_container_width=True):
                        st.session_state.pending_query = q
                        st.rerun()

        st.session_state.messages.append(msg)

def admin_panel():
    st.markdown("## Quản trị hệ thống")
    st.markdown(f"Đang đăng nhập với quyền **Admin** – `{st.session_state.username}`")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Danh sách session")
        sessions = admin_list_sessions()
        if not sessions:
            st.info("Chưa có session nào")
        selected = st.selectbox("Chọn session để xem", sessions) if sessions else None

    with col2:
        st.markdown("### Lịch sử hội thoại")
        if selected:
            history = admin_get_history(selected)
            if history:
                for msg in history:
                    role = "**User**" if msg["role"] == "user" else "**Assistant**"
                    st.markdown(f"{role}: {msg['content']}")
                    st.markdown("---")
            else:
                st.info("Session này không có tin nhắn hoặc đã hết hạn")

    if st.button("Làm mới danh sách"):
        st.rerun()

    if st.button("Đăng xuất"):
        logout()

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.session_id = ""
    st.session_state.messages = []
    st.rerun()

def main():
    st.set_page_config(page_title="Vietnamese RAG Chatbot", page_icon="🧠", layout="wide")
    st.markdown("""
        <style>
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-top: 8px;
        }
        div[data-testid="column"] button {
            font-size: 0.85rem;
            padding: 4px 8px;
            height: auto;
            white-space: normal;
            word-break: break-word;
        }
        </style>
    """, unsafe_allow_html=True)

    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.role == "admin":
            admin_panel()
        else:
            chat_interface()

if __name__ == "__main__":
    main()
