# # Load Images.Py

# from dataloader.dataset.entities import Image

# class LoadImages:
#     def load_images(self) -> dict[int, Image]:
#         images = {}
#         for image_id, colmap_image in self.reconstruction.images.items():
#             if hasattr(colmap_image, "cam_from_world"):
#                 pose = colmap_image.cam_from_world()
#                 R, t = pose.rotation.matrix(), pose.translation
#             else:
#                 R, t = colmap_image.R, colmap_image.t

#             images[image_id] = Image(
#                 id=image_id, name=colmap_image.name,
#                 camera_id=colmap_image.camera_id, R=R, t=t
#             )
#         return images
