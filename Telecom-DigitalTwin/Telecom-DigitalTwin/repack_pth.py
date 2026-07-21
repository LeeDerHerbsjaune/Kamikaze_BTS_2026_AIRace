"""
repack_pth.py - Đóng gói lại 1 folder đã bị lỡ giải nén (từ file .pth PyTorch
gốc, vốn tự nó là 1 container ZIP) trở lại thành file .pth có thể
torch.load() được bình thường.

Dấu hiệu bạn cần script này: mở checkpoint tải về thấy 1 FOLDER (thay vì 1
file .pth) chứa bên trong các file/thư mục: byteorder, data.pkl, version,
.format_version, .storage_alignment, data/, .data/ - đây chính là nội dung
thật bên trong 1 file .pth (PyTorch lưu checkpoint dưới dạng ZIP container
từ bản >=1.6), đã bị công cụ tải/giải nén nào đó tự bung ra.

Dùng:
    python repack_pth.py "D:\\...\\iter_5000\\iter_5000" "D:\\...\\iter_5000_fixed.pth"

Sau khi chạy xong, PHẢI kiểm tra lại bằng:
    python -c "import torch; ckpt = torch.load('iter_5000_fixed.pth', map_location='cpu'); print(ckpt['iteration'], ckpt['gaussians']['xyz'].shape)"
trước khi dùng cho inference - nếu lệnh trên không lỗi và in ra số iteration
+ shape hợp lý, checkpoint đã được khôi phục đúng.
"""
import argparse
import os
import zipfile


def repack_pth(src_folder: str, out_pth: str):
    src_folder = os.path.abspath(src_folder)
    if not os.path.isdir(src_folder):
        raise NotADirectoryError(f"'{src_folder}' không phải là 1 thư mục.")

    # Thư mục gốc bên trong zip = tên thư mục nguồn (KHÔNG bắt buộc trùng tên
    # file .pth cuối cùng - PyTorchStreamReader tự dò theo entry đầu tiên nó
    # đọc được, miễn mọi file dùng chung 1 prefix nhất quán).
    parent_dir = os.path.dirname(src_folder.rstrip(os.sep))
    n_files = 0

    with zipfile.ZipFile(out_pth, "w", zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(src_folder):
            for fn in files:
                full_path = os.path.join(root, fn)
                # arcname PHẢI dùng dấu '/' (chuẩn ZIP), kể cả khi chạy trêncd
                # Windows (os.sep là '\\') - nếu để nguyên '\\' file zip vẫn
                # tạo được nhưng PyTorchStreamReader (dựa trên miniz, đọc
                # theo chuẩn ZIP POSIX path) sẽ không nhận diện đúng thư mục.
                rel_path = os.path.relpath(full_path, parent_dir)
                arcname = rel_path.replace(os.sep, "/")
                zf.write(full_path, arcname=arcname)
                n_files += 1

    print(f"Đã đóng gói {n_files} file từ '{src_folder}' -> '{out_pth}'")
    print("Kiểm tra lại bằng:")
    print(f'  python -c "import torch; ckpt = torch.load(r\'{out_pth}\', map_location=\'cpu\'); '
          f'print(ckpt[\'iteration\'], ckpt[\'gaussians\'][\'xyz\'].shape)"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_folder", help="Thư mục đã bị giải nén (chứa data.pkl, byteorder, version, ...)")
    parser.add_argument("out_pth", help="Đường dẫn file .pth đóng gói lại")
    args = parser.parse_args()
    repack_pth(args.src_folder, args.out_pth)