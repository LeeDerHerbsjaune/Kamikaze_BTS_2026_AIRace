# # Load Pose.Py

# from typing import Optional

# from dataloader.dataset.entities import Point3D

# class LoadPose:
#     def load_points3D(self, min_track_length: int = 3, max_error: Optional[float] = 2.0) -> dict[int, Point3D]:
#         points3D = {}
#         n_skipped = 0
#         for point3D_id, point3D in self.reconstruction.points3D.items():
#             track_length = len(point3D.track.elements) if hasattr(point3D, "track") else None
#             if min_track_length and track_length is not None and track_length < min_track_length:
#                 n_skipped += 1
#                 continue
#             if max_error is not None and getattr(point3D, "error", 0.0) > max_error:
#                 n_skipped += 1
#                 continue
#             points3D[point3D_id] = Point3D(id=point3D_id, xyz=point3D.xyz, color=point3D.color)
        
#         if n_skipped > 0:
#             print(f"[ColmapLoader] Đã lọc bỏ {n_skipped} điểm nhiễu.")
#         return points3D