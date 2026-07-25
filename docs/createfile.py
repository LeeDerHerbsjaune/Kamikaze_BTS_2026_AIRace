import os

# Danh sách các file cần tạo
file_list = [
    "logger.py",
    "seed.py",
    "config.py",
    "timer.py",
    "io.py"
]

# Thư mục đích
target_dir = "utils"

def create_bulk_files(directory, files):
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Tạo thành công thư mục: `{directory}/` \n')
    
    print("--- Bắt đầu khởi tạo các file ---")
    for file_name in files:
        # Kết hợp đường dẫn: docs/file_name.md
        file_path = os.path.join(directory, file_name)
        
        # Kiểm tra xem file đã tồn tại chưa để tránh ghi đè dữ liệu cũ
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                # Gợi ý: Tự động điền tiêu đề Markdown bằng tên file cho đẹp
                title = file_name.replace('.md', '').replace('_', ' ').title()
                f.write(f"# {title}\n\n**Tài liệu cho {title.lower()}**\n")
            print(f"✅ Đã tạo: {file_path}")
        else:
            print(f"⚠️ File đã tồn tại (Bỏ qua): {file_path}")
            
    print("\n--- Hoàn thành! ---")

if __name__ == "__main__":
    create_bulk_files(target_dir, file_list)