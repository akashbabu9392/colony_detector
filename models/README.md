# Model weights

Weights are not committed (size). Drop files here, or point the env vars at
them; the service picks up whatever is present when `CD_ENGINES=auto`.

| File (auto-detected)                   | Env var               | Engine   | How to get it |
| -------------------------------------- | --------------------- | -------- | ------------- |
| `yolo11s_tiles.pt` / `best.pt`         | `CD_YOLO_WEIGHTS`     | yolo     | `tools/train_yolo.py` |
| `rfdetr_tiles.pth`                     | `CD_RFDETR_WEIGHTS`   | rfdetr   | `tools/train_rfdetr.py` (+ `CD_RFDETR_SIZE`) |
| `cellpose_colony` / `colony_project_model` | `CD_CELLPOSE_WEIGHTS` | cellpose | fine-tuned Cellpose (CPnet, cellpose<4) |
| `fusion.json`                          | —                     | fusion   | `tools/tune_fusion.py` on your gold plates |

The classical engine needs no weights and always runs.

A learned engine whose count differs from the classical count by more than
`CD_OUTLIER_RATIO` (2.5x) on a plate is treated as out of domain for that
plate: it is left out of the vote and the result is flagged
`ENGINE_DISAGREEMENT` for review. MicroID's existing Cellpose model
(`backend/models/finetuned/models/colony_project_model`) behaves like that on
the public OpenCFU sample plates (it finds 1–21 of 50–500 colonies), so
validate any model with `tools/evaluate.py --per-engine` before enabling it.
