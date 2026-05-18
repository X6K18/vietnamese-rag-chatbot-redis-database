import requests

def test_chat():
    url = "http://127.0.0.1:8000/chat"
    payload = {
        "session_id": "test_user_123",
        "query": "Chào bạn, bạn có thể giúp gì cho tôi?"
    }
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Lỗi kết nối:", e)

if __name__ == "__main__":
    test_chat()