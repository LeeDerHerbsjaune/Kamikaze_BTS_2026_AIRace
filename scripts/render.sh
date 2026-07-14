#!/bin/bash
# Chỉ chạy inference render novel views từ checkpoint đã có sẵn
set -e
CONFIG=${1:-configs/default.yaml}
python -m inference.render_novel_views --config "$CONFIG"
