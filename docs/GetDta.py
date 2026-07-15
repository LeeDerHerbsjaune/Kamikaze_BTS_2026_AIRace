# Cách này để truy cập dữ liệu từ thư mục sparse của COLMAP
import pycolmap

reconstruction = pycolmap.Reconstruction(
    r"E:\project\AIRace2026\VAI_NVS_DATA\phase1\public_set\hcm0031\train\sparse\0"
)

output_file = "colmap_info.txt"

with open(output_file, "w", encoding="utf-8") as f:

    # Thống kê
    f.write("========== Reconstruction ==========\n")
    f.write(f"Number of images   : {reconstruction.num_images()}\n")
    f.write(f"Number of cameras  : {reconstruction.num_cameras()}\n")
    f.write(f"Number of points3D : {reconstruction.num_points3D()}\n\n")

    # Cameras
    f.write("========== Cameras ==========\n")
    for camera_id, camera in reconstruction.cameras.items():
        f.write(f"Camera ID : {camera_id}\n")
        f.write(f"Model     : {camera.model}\n")
        f.write(f"Size      : {camera.width} x {camera.height}\n")
        f.write(f"Params    : {camera.params}\n")
        f.write("-" * 60 + "\n")

    # Images
    f.write("\n========== Images ==========\n")
    for image_id, image in reconstruction.images.items():
        f.write(f"Image ID       : {image_id}\n")
        f.write(f"Name           : {image.name}\n")
        f.write(f"Camera ID      : {image.camera_id}\n")
        f.write(f"Cam from World :\n{image.cam_from_world}\n")
        f.write("-" * 60 + "\n")

    # Point Cloud
    f.write("\n========== Points3D ==========\n")
    for point3D_id, point3D in reconstruction.points3D.items():
        f.write(f"Point ID : {point3D_id}\n")
        f.write(f"XYZ      : {point3D.xyz}\n")
        f.write(f"Color    : {point3D.color}\n")
        f.write("-" * 60 + "\n")

print(f"Đã lưu thông tin vào: {output_file}")