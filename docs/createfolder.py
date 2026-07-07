import os

# Danh sách các folder bạn muốn tạo bên trong docs/
folder_list = [
    "gaussian",
    "semantic",
    "sparse_view",
    "depth_guided"
]

# Thư mục cha
parent_dir = "models"

print("--- Bắt đầu khởi tạo các folder ---")

for folder_name in folder_list:
    # Kết hợp đường dẫn: docs/ten_folder
    folder_path = os.path.join(parent_dir, folder_name)
    
    # Kiểm tra nếu folder chưa tồn tại thì mới tạo
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Đã tạo thành công: {folder_path}/")
    else:
        print(f"⚠️ Folder đã tồn tại (Bỏ qua): {folder_path}/")

print("\n--- Hoàn thành! ---")