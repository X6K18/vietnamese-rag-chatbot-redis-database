import redis

# 1. Khởi tạo kết nối
# Dùng 127.0.0.1 thay localhost để tránh lỗi IPv6 trên Windows
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 2. Kiểm tra kết nối bằng lệnh ping
try:
    if r.ping():
        print("Kết nối Redis thành công!")
except Exception as e:
    print(f"Lỗi kết nối: {e}")