# frontend/app.py
import streamlit as st
import requests
import json
from typing import List, Dict, Generator

# ---------- CẤU HÌNH ----------
API_BASE = "http://localhost:8000"          # Địa chỉ backend FastAPI
ADMIN_TOKEN = "admin-secret-123"            # Token admin (phải trùng với backend)

# ---------- XÁC THỰC NGƯỜI DÙNG ----------
# Trong thực tế, nên dùng database hoặc gọi API đăng nhập.
# Ở đây giữ nguyên cơ chế đơn giản để minh họa.
VALID_USERS = {
    "phantrongnguyen0618@gmail.com": {"password": "123", "role": "user"},
    "smoking": {"password": "456", "role": "user"},
    "admin": {"password": "admin123", "role": "admin"}
}

def authenticate(username: str, password: str):
    if username in VALID_USERS and VALID_USERS[username]["password"] == password:
        return VALID_USERS[username]["role"]
    return None

# ---------- HÀM GỌI API (USER) ----------
def get_chat_history(session_id: str) -> List[Dict]:
    """Lấy lịch sử chat của session hiện tại"""
    try:
        resp = requests.get(f"{API_BASE}/chat/history", params={"session_id": session_id})
        if resp.status_code == 200:
            return resp.json().get("history", [])
    except Exception as e:
        st.error(f"Không thể tải lịch sử: {e}")
    return []

def delete_chat_history(session_id: str) -> bool:
    """Xoá lịch sử chat của session hiện tại"""
    try:
        resp = requests.delete(f"{API_BASE}/chat/history", params={"session_id": session_id})
        return resp.status_code == 200
    except Exception as e:
        st.error(f"Lỗi xoá lịch sử: {e}")
        return False

def ask_question_stream(session_id: str, message: str) -> Generator[str, None, None]:
    """Gửi câu hỏi và nhận câu trả lời streaming"""
    try:
        resp = requests.post(
            f"{API_BASE}/chat/stream",
            json={"session_id": session_id, "message": message},
            stream=True,
            timeout=60
        )
        if resp.status_code != 200:
            yield f"⚠️ Lỗi từ server: {resp.status_code}"
            return
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk
    except Exception as e:
        yield f"⚠️ Lỗi kết nối: {e}"

# ---------- HÀM GỌI API (ADMIN) ----------
def admin_list_sessions() -> List[str]:
    """Admin: lấy danh sách các session_id có lịch sử"""
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    try:
        resp = requests.get(f"{API_BASE}/admin/sessions", headers=headers)
        if resp.status_code == 200:
            return resp.json().get("sessions", [])
    except Exception as e:
        st.error(f"Lỗi lấy danh sách session: {e}")
    return []

def admin_get_history(session_id: str) -> List[Dict]:
    """Admin: lấy lịch sử của một session bất kỳ"""
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    try:
        resp = requests.get(f"{API_BASE}/admin/history/{session_id}", headers=headers)
        if resp.status_code == 200:
            return resp.json().get("history", [])
    except Exception as e:
        st.error(f"Lỗi lấy lịch sử admin: {e}")
    return []

# ---------- KHỞI TẠO SESSION STATE ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.session_id = ""
    st.session_state.messages = []          # Tin nhắn hiển thị trên UI

# ---------- MÀN HÌNH ĐĂNG NHẬP ----------
def login_page():
    st.title("🔐 Đăng nhập")
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
                # Nếu là user, tải lịch sử chat từ backend
                if role == "user":
                    history = get_chat_history(st.session_state.session_id)
                    st.session_state.messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in history
                    ]
                else:
                    st.session_state.messages = []  # Admin không cần lịch sử chat
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu")

# ---------- GIAO DIỆN ADMIN ----------
def admin_panel():
    st.markdown("## 🛠️ Quản trị hệ thống")
    st.markdown(f"👑 Đang đăng nhập với quyền **Admin** – `{st.session_state.username}`")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Danh sách session")
        sessions = admin_list_sessions()
        if not sessions:
            st.info("Chưa có session nào (chưa có người dùng chat)")
        selected = st.selectbox("Chọn session để xem", sessions) if sessions else None
    
    with col2:
        st.markdown("### Lịch sử hội thoại")
        if selected:
            history = admin_get_history(selected)
            if history:
                for msg in history:
                    role = "👤 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
                    st.markdown(f"{role}: {msg['content']}")
                    st.markdown("---")
            else:
                st.info("Session này không có tin nhắn hoặc đã hết hạn")
    
    if st.button("🔁 Làm mới danh sách"):
        st.rerun()
    
    # Nút đăng xuất
    if st.button("🚪 Đăng xuất"):
        logout()

# ---------- GIAO DIỆN CHAT CHO USER ----------
def chat_interface():
    # Sidebar
    with st.sidebar:
        st.markdown(f"👤 **Người dùng:** `{st.session_state.username}`")
        if st.button("🚪 Đăng xuất"):
            logout()
        st.markdown("---")
        st.title("💬 Tuỳ chỉnh")
        if st.button("🗑️ Xoá lịch sử chat"):
            if delete_chat_history(st.session_state.session_id):
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Không thể xoá lịch sử")
        st.markdown("---")
        st.info("Bot sẽ phân loại chủ đề, truy xuất tài liệu liên quan và trả lời dựa trên ngữ cảnh hội thoại.")
    
    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Ô nhập tin nhắn
    if query := st.chat_input("Nhập câu hỏi của bạn..."):
        # Thêm tin nhắn user vào UI
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Nhận streaming từ bot
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for token in ask_question_stream(st.session_state.session_id, query):
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        
        # Lưu tin nhắn assistant vào UI
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def logout():
    """Đăng xuất, xoá tất cả session state"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.session_id = ""
    st.session_state.messages = []
    st.rerun()

# ---------- ĐIỀU HƯỚNG CHÍNH ----------
def main():
    st.set_page_config(page_title="Vietnamese RAG Chatbot", page_icon="🧠", layout="wide")
    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.role == "admin":
            admin_panel()
        else:
            chat_interface()

if __name__ == "__main__":
    main()