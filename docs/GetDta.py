#cách này để truy cập dữ liệu từ thư mục sparse của COLMAP
import pycolmap

reconstruction = pycolmap.Reconstruction("E:\\project\\AIRace2026\\VAI_NVS_DATA\\phase1\\private_set1\\HCM0249\\train\\sparse\\0")

print(reconstruction.num_images())
print(reconstruction.num_points3D())
print(reconstruction.num_cameras()) 

#lấy cam
for camera_id, camera in reconstruction.cameras.items():
    print(camera_id)
    print(camera.model)
    print(camera.width, camera.height)
    print(camera.params)

#lấy img
for image_id, image in reconstruction.images.items():
    print(image.name)
    print(image.cam_from_world)

#lấy pointcloud
for point3D_id, point3D in reconstruction.points3D.items():
    print(point3D_id)
    print(point3D.xyz)
    print(point3D.color)