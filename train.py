import hydra
from omegaconf import DictConfig, OmegaConf

# config_path: Đường dẫn tương đối từ file main.py đến thư mục chứa file yaml
# config_name: Tên file yaml (không cần đuôi .yaml)
@hydra.main(version_base=None, config_path="config", config_name="default")
def my_app(cfg: DictConfig) -> None:
    print("--- Cấu hình nhận được từ Hydra ---")
    
    # 1. Truy cập trực tiếp bằng dấu chấm (Dot-notation)
    print(f"Server Host: {cfg.server.host}")
    print(f"Dataset Path: {cfg.dataset.path}")
    print(f"Learning Rate: {cfg.training.lr}")
    
    print("\n--- In toàn bộ cấu hình dưới dạng YAML đẹp ---")
    # Sử dụng OmegaConf (đi kèm với Hydra) để xem toàn bộ config
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()