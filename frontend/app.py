import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"

def api_register(username, password):
    try:
        resp = requests.post(f"{API_BASE}/auth/register", json={"username": username, "password": password}, timeout=10)
        if resp.status_code == 200:
            return {"success": True, "message": "Đăng ký thành công!"}
        return {"success": False, "message": resp.json().get("detail", "Lỗi đăng ký")}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối: {e}"}

def api_login(username, password):
    try:
        resp = requests.post(f"{API_BASE}/auth/login", json={"username": username, "password": password}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {"success": True, "username": data["username"], "role": data["role"], "session_id": data["session_id"]}
        return {"success": False, "message": resp.json().get("detail", "Sai tên đăng nhập hoặc mật khẩu")}
    except Exception as e:
        return {"success": False, "message": f"Lỗi kết nối: {e}"}

def get_chat_history(session_id):
    try:
        resp = requests.get(f"{API_BASE}/chat/history", params={"session_id": session_id})
        if resp.status_code == 200:
            return resp.json().get("history", [])
    except Exception:
        pass
    return []

def delete_chat_history(session_id):
    try:
        resp = requests.delete(f"{API_BASE}/chat/history", params={"session_id": session_id})
        return resp.status_code == 200
    except Exception:
        return False

def ask_question_stream(session_id, message):
    try:
        resp = requests.post(
            f"{API_BASE}/chat/stream",
            json={"session_id": session_id, "message": message},
            stream=True,
            timeout=120
        )
        if resp.status_code != 200:
            yield {"type": "error", "content": f"Lỗi từ server: {resp.status_code}"}
            return
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield {"type": "error", "content": f"Lỗi kết nối: {e}"}

def admin_list_sessions():
    headers = {"Authorization": "Bearer admin-secret-123"}
    try:
        resp = requests.get(f"{API_BASE}/admin/sessions", headers=headers)
        if resp.status_code == 200:
            return resp.json().get("sessions", [])
    except Exception:
        pass
    return []

def admin_get_history(session_id):
    headers = {"Authorization": "Bearer admin-secret-123"}
    try:
        resp = requests.get(f"{API_BASE}/admin/history/{session_id}", headers=headers)
        if resp.status_code == 200:
            return resp.json().get("history", [])
    except Exception:
        pass
    return []

MODERN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }

    h1, h2, h3, .stMarkdown {
        color: #e0e0ff !important;
    }

    .auth-container {
        max-width: 420px;
        margin: 60px auto;
        padding: 40px 35px;
        background: rgba(255,255,255,0.06);
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
    }

    .auth-title {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .auth-subtitle {
        text-align: center;
        color: #8888aa;
        font-size: 14px;
        margin-bottom: 30px;
    }

    .auth-tabs {
        display: flex;
        gap: 0;
        margin-bottom: 28px;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
    }

    .auth-tab {
        flex: 1;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        cursor: pointer;
        font-weight: 500;
        font-size: 14px;
        color: #8888aa;
        transition: all 0.3s ease;
        border: none;
        background: transparent;
    }

    .auth-tab.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #e0e0ff !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #6666aa !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.5);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .chat-message {
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 12px;
        animation: fadeIn 0.3s ease;
    }

    .stChatMessage {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 8px !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="chatMessageContent"] {
        color: #e0e0ff !important;
    }

    div[data-testid="chatMessageContent"] p {
        color: #e0e0ff !important;
        line-height: 1.6;
    }

    .stChatInputContainer {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
        padding: 4px !important;
    }

    .stChatInputContainer input {
        color: #e0e0ff !important;
    }

    .stExpander {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
    }

    .stExpander summary {
        color: #8888cc !important;
        font-weight: 500;
    }

    .stInfo, .stSuccess, .stError {
        border-radius: 12px !important;
    }

    .stSidebar {
        background: rgba(15,12,41,0.8) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    .stSidebar .stMarkdown {
        color: #c0c0e0 !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid rgba(102,126,234,0.3);
        color: #8888cc;
    }

    .typing-dots::after {
        content: '';
        animation: dots 1.5s steps(4, end) infinite;
    }

    @keyframes dots {
        0% { content: ''; }
        25% { content: '.'; }
        50% { content: '..'; }
        75% { content: '...'; }
        100% { content: ''; }
    }

    .stColumn button {
        background: rgba(102,126,234,0.15) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 10px !important;
        color: #a0a0e0 !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
        transition: all 0.3s ease;
    }

    .stColumn button:hover {
        background: rgba(102,126,234,0.3) !important;
        border-color: #667eea !important;
        transform: translateY(-1px);
    }

    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(102,126,234,0.3);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102,126,234,0.5);
    }

    div[data-testid="stNotification"] {
        border-radius: 12px !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stMarkdown"] p {
        color: #c0c0e0 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
</style>
"""

def auth_page():
    st.markdown(MODERN_CSS, unsafe_allow_html=True)

    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">🧠 RAG Chatbot</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Trợ lý AI thông minh - Tiếng Việt</div>', unsafe_allow_html=True)

        tab = st.session_state.get("auth_tab", "login")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Đăng nhập", key="tab_login", use_container_width=True,
                         type="secondary" if tab != "login" else "primary"):
                st.session_state.auth_tab = "login"
                st.rerun()
        with col2:
            if st.button("Đăng ký", key="tab_register", use_container_width=True,
                         type="secondary" if tab != "register" else "primary"):
                st.session_state.auth_tab = "register"
                st.rerun()

        if tab == "login":
            with st.form("login_form"):
                username = st.text_input("Tên đăng nhập", placeholder="Nhập tên đăng nhập...")
                password = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu...")
                submitted = st.form_submit_button("Đăng nhập", use_container_width=True)
                if submitted:
                    if not username or not password:
                        st.error("Vui lòng nhập đầy đủ thông tin")
                    else:
                        result = api_login(username, password)
                        if result["success"]:
                            st.session_state.authenticated = True
                            st.session_state.username = result["username"]
                            st.session_state.role = result["role"]
                            st.session_state.session_id = result["session_id"]
                            if result["role"] == "user":
                                history = get_chat_history(result["session_id"])
                                st.session_state.messages = [
                                    {"role": msg["role"], "content": msg["content"]}
                                    for msg in history
                                ]
                            st.rerun()
                        else:
                            st.error(result["message"])

        else:
            with st.form("register_form"):
                new_username = st.text_input("Tên đăng nhập", placeholder="Chọn tên đăng nhập...")
                new_password = st.text_input("Mật khẩu", type="password", placeholder="Chọn mật khẩu...")
                confirm_password = st.text_input("Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu...")
                submitted = st.form_submit_button("Đăng ký", use_container_width=True)
                if submitted:
                    if not new_username or not new_password:
                        st.error("Vui lòng nhập đầy đủ thông tin")
                    elif len(new_username) < 3:
                        st.error("Tên đăng nhập phải có ít nhất 3 ký tự")
                    elif len(new_password) < 3:
                        st.error("Mật khẩu phải có ít nhất 3 ký tự")
                    elif new_password != confirm_password:
                        st.error("Mật khẩu xác nhận không khớp")
                    else:
                        result = api_register(new_username, new_password)
                        if result["success"]:
                            st.success(result["message"])
                            st.session_state.auth_tab = "login"
                            st.rerun()
                        else:
                            st.error(result["message"])

        st.markdown('</div>', unsafe_allow_html=True)

def render_message(msg):
    role = msg["role"]
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    follow_up = msg.get("follow_up", [])

    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander(f"📚 Nguồn tham khảo ({len(sources)})"):
                for i, s in enumerate(sources):
                    url = s.get("url", "")
                    title = s.get("title", "Không có tiêu đề")
                    source = s.get("source", "")
                    score = s.get("score", 0)
                    if url:
                        st.markdown(
                            f"**{i+1}. [{title}]({url})**  \n"
                            f"📂 `{source}` • Độ liên quan: `{score:.0%}`"
                        )
                    else:
                        st.markdown(
                            f"**{i+1}. {title}**  \n"
                            f"📂 `{source}` • Độ liên quan: `{score:.0%}`"
                        )
                        if i < len(sources) - 1:
                            st.markdown("---")

        if follow_up and role == "assistant":
            st.markdown("---")
            st.markdown("💡 **Câu hỏi gợi ý:**")
            cols = st.columns(min(len(follow_up), 3))
            for i, q in enumerate(follow_up):
                col_idx = i % 3
                if cols[col_idx].button(q, key=f"fu_{hash(q)}_{len(st.session_state.messages)}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

def chat_interface():
    with st.sidebar:
        st.markdown("### 🙋 Người dùng")
        st.markdown(f'<span class="status-badge">👤 {st.session_state.username}</span>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### ⚙️ Tuỳ chỉnh")
        if st.button("🗑️ Xoá lịch sử", use_container_width=True):
            if delete_chat_history(st.session_state.session_id):
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Không thể xoá lịch sử")
        st.markdown("---")
        st.markdown("### ℹ️ Giới thiệu")
        st.info(
            "Bot sử dụng **PhoBERT** để phân tích chủ đề, "
            "**FAISS** để truy xuất tài liệu và **Qwen2.5** "
            "để sinh câu trả lời có trích dẫn nguồn."
        )
        st.markdown("---")
        if st.button("🚪 Đăng xuất", use_container_width=True):
            logout()

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
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)})"):
                    for i, s in enumerate(sources):
                        url = s.get("url", "")
                        title = s.get("title", "Không có tiêu đề")
                        source = s.get("source", "")
                        score = s.get("score", 0)
                        if url:
                            st.markdown(
                                f"**{i+1}. [{title}]({url})**  \n"
                                f"📂 `{source}` • Độ liên quan: `{score:.0%}`"
                            )
                        else:
                            st.markdown(
                                f"**{i+1}. {title}**  \n"
                                f"📂 `{source}` • Độ liên quan: `{score:.0%}`"
                            )
                        if i < len(sources) - 1:
                            st.markdown("---")

            if follow_up:
                st.markdown("---")
                st.markdown("💡 **Câu hỏi gợi ý:**")
                cols = st.columns(min(len(follow_up), 3))
                for i, q in enumerate(follow_up):
                    key = f"fu_{hash(q)}_{len(st.session_state.messages)}"
                    col_idx = i % 3
                    if cols[col_idx].button(q, key=key, use_container_width=True):
                        st.session_state.pending_query = q
                        st.rerun()

        st.session_state.messages.append(msg)

def admin_panel():
    st.markdown("## 🛡️ Quản trị hệ thống")
    st.markdown(f'<span class="status-badge">👤 Admin: {st.session_state.username}</span>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 💬 Phiên hoạt động")
        sessions = admin_list_sessions()
        if not sessions:
            st.info("Chưa có phiên nào")
        selected = st.selectbox("Chọn phiên", sessions or ["(trống)"], label_visibility="collapsed") if sessions else None

    with col2:
        st.markdown("### 📜 Lịch sử hội thoại")
        if selected:
            history = admin_get_history(selected)
            if history:
                for msg in history:
                    role_icon = "🙋" if msg["role"] == "user" else "🤖"
                    st.markdown(f"**{role_icon} {msg['role'].title()}**: {msg['content']}")
                    st.markdown("---")
            else:
                st.info("Phiên này không có tin nhắn hoặc đã hết hạn")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🚪 Đăng xuất", use_container_width=True):
            logout()

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.session_id = ""
    st.session_state.messages = []
    st.rerun()

def main():
    st.set_page_config(page_title="RAG Chatbot - Trợ lý AI", page_icon="🧠", layout="wide")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.session_state.session_id = ""
        st.session_state.messages = []
        st.session_state.auth_tab = "login"

    if not st.session_state.authenticated:
        auth_page()
    else:
        st.markdown(MODERN_CSS, unsafe_allow_html=True)
        if st.session_state.role == "admin":
            admin_panel()
        else:
            chat_interface()

if __name__ == "__main__":
    main()
