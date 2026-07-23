"""Logger: prints to console + writes to a file (train.log) + (optionally)
TensorBoard, and also maintains "notes.md" - a table summarizing every
result file (checkpoint, point cloud .ply, novel-view images, video, camera
trajectory plot...) produced during train/inference, along with the
timestamp and relevant metrics at that point in time."""
import os
import datetime


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


class Logger:
    def __init__(self, log_dir, use_tensorboard=True):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = open(os.path.join(log_dir, "train.log"), "a", encoding="utf-8")

        self.notes_path = os.path.join(log_dir, "notes.md")
        self._notes_header_written = os.path.exists(self.notes_path) and os.path.getsize(self.notes_path) > 0
        self.notes_file = open(self.notes_path, "a", encoding="utf-8")

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

    def log_progress(self, iteration, n_iters, loss, n_gaussians, it_per_sec=None, lr=None):
        """A more detailed progress line than pbar.set_postfix (which only
        shows in the console and is NOT written to train.log). Call this at
        log_interval to keep a full progress history in the file, even if
        the log is viewed after the notebook/terminal has closed (e.g. on
        Kaggle once the session has expired)."""
        parts = [f"iter {iteration}/{n_iters}", f"loss={loss:.4f}", f"n_gaussians={n_gaussians}"]
        if it_per_sec is not None:
            parts.append(f"{it_per_sec:.2f} it/s")
            remaining_s = (n_iters - iteration) / max(it_per_sec, 1e-6)
            parts.append(f"ETA={datetime.timedelta(seconds=int(remaining_s))}")
        if lr is not None:
            parts.append(f"lr_xyz={lr:.2e}")
        self.log_text("[progress] " + " | ".join(parts))

    # ------------------------------------------------------------------
    def log_artifact(self, kind: str, path: str, iteration=None, note: str = ""):
        """Write one row to notes.md for EVERY result file produced
        (checkpoint, .ply, rendered image, video, plot...). `path` can be a
        file or a directory (e.g. a directory holding many novel-view
        images - in that case the file count is reported instead of a
        single file's size).

        kind: short artifact type, e.g. "checkpoint", "point_cloud",
              "camera_trajectory", "novel_view", "novel_view_video".
        iteration: the iteration this was produced at (None if not
              applicable, e.g. an artifact created during inference after
              training has finished).
        note: free-form extra description, e.g. "psnr=21.3, ssim=0.71" or
              "216 images".
        """
        if not self._notes_header_written:
            self.notes_file.write(
                "# Training results\n\n"
                "| Time | Iteration | Kind | Path | Size | Note |\n"
                "|---|---|---|---|---|---|\n"
            )
            self._notes_header_written = True

        if os.path.isdir(path):
            n_files = sum(len(files) for _, _, files in os.walk(path))
            size_str = f"{n_files} files"
        elif os.path.isfile(path):
            size_str = _human_size(os.path.getsize(path))
        else:
            size_str = "N/A (does not exist yet)"

        iter_str = str(iteration) if iteration is not None else "-"
        row = f"| {self._timestamp()} | {iter_str} | {kind} | `{path}` | {size_str} | {note} |\n"
        self.notes_file.write(row)
        self.notes_file.flush()

    def log_summary(self, title: str, lines: list):
        """Write a markdown summary block (list of lines) to the end of
        notes.md - used at the end of training/inference for a quick-read
        summary without having to scroll through the whole artifact table."""
        self.notes_file.write(f"\n## {title}\n\n")
        for line in lines:
            self.notes_file.write(f"- {line}\n")
        self.notes_file.write("\n")
        self.notes_file.flush()

    def close(self):
        if self.tb:
            self.tb.close()
        self.log_file.close()
        self.notes_file.close()
