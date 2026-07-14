import torch


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2 + r[:, 3] ** 2)
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R


def build_scaling_rotation(scaling, rotation):
    L = torch.zeros((scaling.shape[0], 3, 3), device=scaling.device)
    R = build_rotation(rotation)
    L[:, 0, 0] = scaling[:, 0]
    L[:, 1, 1] = scaling[:, 1]
    L[:, 2, 2] = scaling[:, 2]
    L = R @ L
    return L
