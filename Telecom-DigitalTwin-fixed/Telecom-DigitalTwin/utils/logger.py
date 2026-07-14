"""Logger đơn giản: in console + ghi ra file + (tuỳ chọn) TensorBoard."""
import os
import datetime


class Logger:
    def __init__(self, log_dir, use_tensorboard=True):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, "train.log"), "a")
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir)
            except ImportError:
                self.tb = None

    def _timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_text(self, msg):
        line = f"[{self._timestamp()}] {msg}"
        print(line)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def log_scalar(self, tag, value, step):
        if self.tb:
            self.tb.add_scalar(tag, value, step)

    def log_dict(self, prefix, d, step):
        for k, v in d.items():
            self.log_scalar(f"{prefix}/{k}", v, step)
        self.log_text(f"[{prefix} @ iter {step}] " + ", ".join(f"{k}={v:.4f}" for k, v in d.items()))

    def close(self):
        if self.tb:
            self.tb.close()
        self.log_file.close()
