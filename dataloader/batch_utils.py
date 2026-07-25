import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any

def bts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hàm đóng gói (Custom Collate Function) chuyển đổi danh sách các mẫu dữ liệu
    đơn lẻ từ BTSDataset thành một Batch thống nhất để nạp vào GPU.
    
    Giải quyết bài toán gom cụm các kiểu dữ liệu hỗn hợp (Tensor + String).
    """
    if not batch:
        return {}

    # Khởi tạo từ điển chứa batch dữ liệu đầu ra
    batched_data = {}
    
    # 1. Thu thập và xếp chồng (stack) các dữ liệu Tensor tính toán hình học
    # Các Tensor như image [3, H, W], R [3, 3], t [3] sẽ được xếp chồng thêm một trục Batch_Size ở đầu
    batched_data["image"] = torch.stack([sample["image"] for sample in batch], dim=0)
    batched_data["R"] = torch.stack([sample["R"] for sample in batch], dim=0)
    batched_data["t"] = torch.stack([sample["t"] for sample in batch], dim=0)
    
    # 2. Thu thập các metadata dạng chuỗi (String) hoặc ID không thể tính toán số học
    # Thay vì báo lỗi, chúng ta giữ chúng dưới dạng Danh sách (List) thông thường
    batched_data["image_name"] = [sample["image_name"] for sample in batch]
    batched_data["camera_id"] = [sample["camera_id"] for sample in batch]
    
    return batched_data


def create_dataloader(dataset: torch.utils.data.Dataset, 
                      cfg: Dict[str, Any], 
                      is_training: bool = True) -> DataLoader:
    """
    Hàm tiện ích giúp khởi tạo nhanh bộ nạp DataLoader của PyTorch, 
    tự động cấu hình bộ nhớ đệm và gán hàm đóng gói bts_collate_fn.
    """
    # Đọc các thông số thiết lập từ file cấu hình config
    batch_size = cfg["training"]["batch_size"] if is_training else cfg.get("eval", {}).get("batch_size", 1)
    shuffle = is_training  # Chỉ xáo trộn dữ liệu khi huấn luyện
    num_workers = cfg["training"].get("num_workers", 4)
    
    # Khởi tạo DataLoader tiêu chuẩn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=bts_collate_fn,
        pin_memory=True,          # Tối ưu hóa tốc độ đẩy dữ liệu từ RAM máy tính lên VRAM của GPU
        drop_last=is_training     # Loại bỏ batch cuối cùng nếu bị lẻ thành viên, giữ kích thước batch ổn định khi train
    )
    
    return dataloader