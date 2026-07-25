Telecom-DigitalTwin/
│
├── configs/
│
├── dataloader/
│
├── models/
│
├── renderer/
│
├── losses/
│
├── trainers/
│
├── evaluation/
│
├── inference/
│
├── visualization/
│
├── utils/
│
├── scripts/
│
├── outputs/
│
└── train.py

workflow
                      Competition Dataset
                               │
                               ▼
                    configs/default.yaml
                               │
                               ▼
                         train.py
                               │
        ┌──────────────────────┼────────────────────────┐
        ▼                      ▼                        ▼
  Load Config            Initialize Logger        Set Random Seed
        │
        ▼
  Build Dataset
        │
        ▼
Preprocessing
        │
        ▼
 Build Gaussian Model
        │
        ▼
 Build Renderer
        │
        ▼
 Build Loss
        │
        ▼
 Build Optimizer
        │
        ▼
       Trainer
        │
        ▼
  ┌──────────────────────────────┐
  │      Training Loop           │
  │                              │
  │ Load Batch                   │
  │      │                       │
  │      ▼                       │
  │ Render Image                 │
  │      ▼                       │
  │ Compute Loss                 │
  │      ▼                       │
  │ Backward                     │
  │      ▼                       │
  │ Optimizer Step               │
  │      ▼                       │
  │ Densify                      │
  │      ▼                       │
  │ Prune                        │
  └──────────────────────────────┘
        │
        ▼
 Validation
        │
        ▼
 Save Checkpoint
        │
        ▼
 Inference
        │
        ▼
 Novel View Images


 configs/
       ║
       ║
       ║═ default.yaml: combine tất cả các file cấu hình lại
       ║
       ║═ dataset.yaml: chứa đường dẫn dataset, format data, các hiệu chỉnh cấu hình và thông số pose
       ║
       ║═ cấu hình cho model: 