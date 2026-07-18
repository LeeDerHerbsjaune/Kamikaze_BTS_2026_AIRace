# # Load Intrinsics.Py
# from dataloader.dataset.entities import Camera

# _CAMERA_PARAM_LAYOUT = {
#     "SIMPLE_PINHOLE": ("f", "cx", "cy"),
#     "SIMPLE_RADIAL": ("f", "cx", "cy", "k"),
#     "RADIAL": ("f", "cx", "cy", "k1", "k2"),
#     "PINHOLE": ("fx", "fy", "cx", "cy"),
#     "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
#     "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")
# }
# class LoadIntrinstic:
#     def __init__(self, reconstruction):
#         self.reconstruction = reconstruction

#     def load_cameras(self) -> dict[int, Camera]:
#             cameras = {}
#             for camera_id, camera in self.reconstruction.cameras.items():
#                 model_name = camera.model.name if hasattr(camera.model, "name") else str(camera.model)
                
#                 params = camera.params
#                 layout = _CAMERA_PARAM_LAYOUT.get(model_name)
#                 if layout is None:
#                     raise NotImplementedError(f"Model '{model_name}' chưa được hỗ trợ.")
                
#                 fx = float(params[layout.index("fx")]) if "fx" in layout else float(params[layout.index("f")])
#                 fy = float(params[layout.index("fy")]) if "fy" in layout else float(params[layout.index("f")])
#                 cx, cy = float(params[layout.index("cx")]), float(params[layout.index("cy")])

#                 cameras[camera_id] = Camera(
#                     id=camera_id, model=model_name,
#                     width=camera.width, height=camera.height,
#                     fx=fx, fy=fy, cx=cx, cy=cy,
#                 )
#             return cameras
        