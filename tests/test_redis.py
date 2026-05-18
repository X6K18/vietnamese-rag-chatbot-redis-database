import redis

# 1. Khởi tạo kết nối
# Mặc định: host='localhost', port=6379, db=0
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# 2. Kiểm tra kết nối bằng lệnh ping
try:
    if r.ping():
        print("Kết nối Redis thành công!")
except Exception as e:
    print(f"Lỗi kết nối: {e}")