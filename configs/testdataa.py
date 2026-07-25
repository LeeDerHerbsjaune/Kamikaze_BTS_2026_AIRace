from huggingface_hub import hf_hub_download
import os

# Đường dẫn thư mục bạn muốn lưu file (thay đổi theo ý bạn)
thu_muc_luu = r"E:\dataset\huggingface\ThuyChi"

# Tải file và ép lưu vào thư mục trên
file_path = hf_hub_download(
    repo_id="YuanhaoXD/Mip-NeRF360", 
    filename="bicycle/poses_bounds.npy", 
    repo_type="dataset",
    local_dir=thu_muc_luu,       # Chỉ định thư mục lưu tại đây
    local_dir_use_symlinks=False # Tắt symlink để tránh lỗi phân mảnh trên Windows
)

print(f"File đã được lưu cấu trúc chính xác tại: {file_path}")