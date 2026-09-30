# Colony Detector — accurate CFU counting on agar plates

Counts bacterial colonies (CFU) on Petri-dish photos and returns every
colony's outline, a total, quality signals and a "needs review" verdict.
It is a drop-in replacement for the **cfu-model-2** inference service that
MicroID calls (`AI_SERVICE_URL` → `POST /predict`), and also runs as a CLI.

```bash
pip install -r requirements.txt
python -m colony_detector plate.jpg --out results/      # CLI: count + annotated image + JSON
uvicorn colony_detector.service:app --port 8000        # HTTP service (MicroID compatible)
```

It works out of the box with **no trained weights** (the classical engine),
and gets more accurate as you add detectors trained on your own plates. The
tooling to do that (hand counting, tiled dataset builder, YOLO / RF-DETR
training, fusion tuning, evaluation) is included.

---

## How it counts

```
image ──► plate finder ──► classical engine ──┐
          (Hough + colour     (always on)      │
           region fallback)                    ├─► consensus fusion ──► quality + review ──► JSON / overlay
                           YOLO11 (tiled) ─────┤     (weighted votes,
                           RF-DETR (tiled) ────┤      out-of-domain guard)
                           Cellpose ───────────┤
                           Grounding DINO ─────┘
                           (zero-shot, opt-in)
```

**Plate finder**: finds the dish as a circle, so the table, labels and
dish wall never reach the counter. It tries Hough circles checked against
edge support, then takes the outermost concentric ring (angled photos show
the meniscus inside the wall). If edges are too weak, it falls back to a
round region that differs in colour from the table. If no dish is found it
uses the frame's inscribed circle, which fits MicroID's pre-cropped disk
images, and flags `PLATE_NOT_FOUND`.

**Classical engine** (`colony_detector/engines/classical.py`, training-free):

1. **Background model**: a two-pass median-filtered agar background
   (colonies masked out on the second pass), then per-pixel Lab colour
   distance. This removes vignetting, lighting gradients and shadows. It
   works for dark colonies on light agar, light colonies on dark agar
   (reflected light) and coloured colonies on coloured media.
2. **Rim flattening**: if a wall or meniscus band is visible, concentric
   structure is removed with a per-ring, per-sector median (lid lines, wall
   shading). Colonies growing in the meniscus are still found.
3. **Thresholding**: hysteresis threshold in robust-noise units, with local
   noise normalisation so grainy agar patches don't turn into speckle.
4. **Artifact rejection**, each counted in `diagnostics.rejected`:
   - air bubbles: hollow rings, lens-like centres, bright arcs with a dark
     lens
   - pen marks, scratches and hairs: long skeleton of constant width
   - marker ink: minority polarity
   - dust: specks far smaller than the plate's typical colony
   - wall glints: a strict outer band where only round, compact,
     high-contrast objects count
5. **Touching colonies** are split by a watershed seeded from colony
   *cores* (above half the blob's peak contrast), which ignores the soft
   skirt that makes necks look thick. A cut is undone if the neck is not
   clearly narrower than the colony. Overlapping clumps that still can't be
   split are estimated from their area (`shape_source: "cluster_estimate"`,
   `count: n`).
6. **Classes**: `colony` / `fuzzy_colony`. Fuzzy means an edge much softer
   than this plate's typical colony, or an irregular spreading outline.
   These are the classes MicroID's UI already styles.

**Learned engines** (optional, `requirements-ml.txt`):

| engine | what | weights |
| --- | --- | --- |
| `rfdetr` (**recommended**) | RF-DETR: a real-time detection transformer on a **DINOv2 foundation-model backbone**. It transfers better than YOLO from small labelled sets, needs no NMS, and handles small, crowded objects well. Runs on **native-resolution overlapping tiles** (SAHI-style) plus a low-res global pass for big colonies; seam duplicates are merged by IoU + intersection-over-smaller | `models/rfdetr_tiles.pth` |
| `yolo` | Ultralytics YOLO11/v8, same tiling: faster and lighter, a good second vote in the ensemble | `models/yolo11s_tiles.pt` |
| `cellpose` | Cellpose CPnet instance segmentation, rescaled with the classical engine's colony-size estimate | `models/colony_project_model` |
| `gdino` | zero-shot Grounding DINO (text prompt "bacterial colony"). This is the detector half of *Colony Grounded SAM2* (arXiv 2603.13393). Needs no training data, so it's handy for bootstrapping labels | HF model id `CD_GDINO_MODEL` |

A detector can be trained with extra classes such as `bubble`, `debris` or
`text`. They teach it what *not* to count and are dropped at inference.

**Fusion** (`colony_detector/fusion.py`): detections from different engines
that land on the same spot are one colony. A colony is counted when
`Σ weight·confidence / Σ weight ≥ threshold`, so agreement raises
confidence. Outlines come from the best mask engine that saw the colony.
Weights and threshold are **fitted to your gold plates** by
`tools/tune_fusion.py` (written to `models/fusion.json`).

**Out-of-domain guard**: if a learned engine's count on a plate differs from
the classical count by more than 2.5×, that engine is excluded for that
plate and the result is flagged `ENGINE_DISAGREEMENT`. This stops a model
that has never seen this kind of plate from silently corrupting the count.

**Review rules** (MicroID TRD §9/§12): `needs_review` with reason codes
`PLATE_NOT_FOUND`, `LOW_FOCUS`, `GLARE`, `OVERGROWTH`, `TNTC` (> 300 by
default), `MANY_LOW_CONFIDENCE` and `ENGINE_DISAGREEMENT`. An annotated
image is produced **only when there is at least one detection**.

---

## Accuracy — what has actually been measured

### 0. Your lab dataset (dataset-v1 release) — the numbers that matter

1,324 plates (3434 × 3434 px, 19,187 hand-labelled colonies), split
deterministically by sample id into 1,051 train / 135 valid / 138 test.
The 138 test plates are never used for training or tuning.

| engine (138 held-out test plates) | exact | within ±1 colony | within ±2 | mean abs. error | detection F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical (no training) | 22.5 % | – | – | 8.6 | 0.19 |
| RF-DETR Nano, half-res tiles, 1 CPU epoch, conf 0.5 | **49.3 %** | **83.3 %** | **92.8 %** | 1.25 | 0.86 |
| RF-DETR Nano, full-res tiles, 3 more CPU epochs, conf 0.75 (fitted on valid) | 35.5 % | 76.1 % | 88.4 % | 2.56 | 0.76 |
| **RF-DETR Medium, full-res tiles, 10 epochs on a Kaggle T4, conf 0.5 (fitted on valid)** | 39.9 % | 79.7 % | 89.9 % | 1.44 | 0.85 |

The half-res row's cut-off (0.5) was picked on the test plates, so it is
slightly optimistic; the full-res row's cut-off was fitted on validation.

The Medium model's weights and fitted `detector.json` are in the
`dataset-v2` release (`results.zip`); put both in `models/` to serve them.

**Label quality is the current ceiling.** On the 65 test plates labelled
with one colony, the Medium model counts more on 29; zoomed in, many of those
"extra" detections are specks identical to specks that *were* labelled on
other plates (inconsistent labels), the rest are scratches and fibres (real
model errors). Reviewing the disagreements by eye:
the model's most confident "false positives" are mostly real colonies that
were never boxed; many labelled colonies are 5–12 px specks that are barely
visible; and some boxes are not colonies (a paper sticker, agar grid texture,
overgrown lawns labelled as 1). The full-res model finds more colonies (87 %
of labelled colonies on sparse plates vs 75 % at half-res) and is penalised
for the unlabelled ones it finds. A label review pass on the disagreeing
plates is needed before exact-count accuracy can be measured or improved
reliably.

The classical engine misses most tiny colonies on these plates (it counted
0 on 47 of the 104 plates with 1–3 colonies), so a trained detector is
required. The RF-DETR row is an early checkpoint from CPU training; the
full-resolution RF-DETR Medium run in `notebooks/train_rfdetr_colab.ipynb`
(GPU) is the intended production model.

### 1. Synthetic plates with exact ground truth (held-out seeds)

Generated by `colony_detector/synth.py`. Each plate has 5–180 colonies,
about 15% of them touching, fuzzy colonies, bubbles, pen marks, dust,
lighting gradients, blur, noise, and both transmitted- and reflected-light
styles. The counts below come from seeds not used during tuning.

| configuration | count MAPE | MAE | within ±5 % | within ±10 % | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| classical only (no training) | 11.1 % | 12.9 | 10 % | 45 % | 0.99 | 0.87 | 0.92 |
| YOLO11n, 12 CPU epochs on 60 synthetic plates | 4.2 % | 4.5 | 65 % | 95 % | 0.96 | 0.94 | 0.95 |
| **RF-DETR Nano, 8 CPU epochs on the same plates** | **1.5 %** | **1.9** | **95 %** | **100 %** | 0.96 | 0.96 | **0.96** |
| classical + RF-DETR (fusion defaults) | 1.6 % | 2.0 | 100 % | 100 % | 0.95 | 0.95 | 0.95 |

On the same validation tiles RF-DETR reached mAP50 0.999 / mAP50-95 0.83
against YOLO11n's 0.73 / 0.44, which is why it is the recommended model.
RF-DETR needs `CD_RFDETR_CONF` around 0.4: at YOLO's usual 0.2 it
over-counts (8.9 % error).

The learned rows are **in-domain** (trained and tested on synthetic plates),
so they show that the tiling, fusion and tuning machinery works. They are
not a claim about real plates. The regression tests in
`tests/test_counting.py` fail if classical accuracy drops.

### 2. Real plates (public OpenCFU samples, classical engine only)

| plate | result | my visual count | notes |
| --- | ---: | ---: | --- |
| A (orange colonies, striped backlight) | 67 | ≈ 70 | 2–3 colonies under the lid rim missed |
| B (orange colonies, pale agar) | 67 | ≈ 69 | colonies on the wall line itself missed |
| C (≈ 500 small colonies, pen marks) | 500 | — | pen strokes and white markers rejected; flagged TNTC |
| two-colours (dark and light red) | 98 | ≈ 107 | angled photo: colonies beyond the meniscus reflection are outside the detected dish |
| G (reflected light, dark agar) | 112 | — | ink rejected by polarity; specular glints on bubbles still counted (≈ 10 FP) |
| BUBBLE15 (bubbles and 3–4 colonies) | 11 | ≈ 4 | isolated bubbles rejected; clusters of small touching bubbles still counted |

The visual counts are mine, made while developing, and are not certified
gold counts. Real accuracy must be measured on **your** plates with
`tools/evaluate.py` (below).

### 3. What does *not* work well yet

- **Bubbles** in reflected light, and clusters of tiny touching bubbles
  (above).
- **Very faint** colonies (low contrast, out of focus) are under-counted.
- **Angled photos** where the dish wall isn't a visible circle.
- **MicroID's current Cellpose model** (`colony_project_model`) finds only
  1–21 colonies on the OpenCFU plates and takes 30–130 s per plate on CPU.
  The out-of-domain guard excludes it automatically; validate it before
  enabling.

All of these are data problems. A detector trained on your own imaging setup,
with `bubble` as a negative class, is the way to close them.

---

## Training on a labelled dataset (recommended path)

Put the dataset in the repo (or anywhere) in any common format: a Roboflow /
Ultralytics **YOLO** export, a **COCO** export, **AGAR** JSON, hand-count
JSON, or a `counts.csv`. The format is auto-detected by
`colony_detector/datasets.py`, and non-colony classes such as `bubble` or
`debris` are dropped.

```bash
pip install -r requirements-ml.txt "rfdetr[train]"
# 1. baseline of the training-free engine on the dataset's test split
python tools/evaluate.py --dataset data/colony_dataset --split test
# 2. tiles (the dataset's own train/valid split is kept, test stays held out)
python tools/build_training_set.py --dataset data/colony_dataset --synthetic 200 --coco --out datasets/tiles
# 3. train RF-DETR (GPU; a Colab T4 works) and optionally YOLO11 as a second vote
python tools/train_rfdetr.py --dataset datasets/tiles/coco --size medium --epochs 60
python tools/train_yolo.py --data datasets/tiles/data.yaml --model yolo11s.pt
# 4. fit the fusion weights on the valid split, then report on test
python tools/tune_fusion.py --dataset data/colony_dataset --split valid
python tools/evaluate.py --dataset data/colony_dataset --split test --per-engine
```

## Getting to production accuracy on your own plates (no dataset yet)

1. **Hand-count 50–100 plates** covering your media, lighting and densities.
   The model pre-fills its detections, so you only fix mistakes:
   `python tools/hand_count.py data/plates` (needs `pip install opencv-python`
   for the GUI). This writes `eval/gold/*.json`.
2. **Baseline**: `python tools/evaluate.py --images data/plates --gold eval/gold`
3. **Build tiles**: `python tools/build_training_set.py --images data/plates --gold eval/gold --synthetic 300 --coco`
   (synthetic plates pre-train; train/val are split by plate, never by tile).
4. **Train** (GPU / Colab, roughly 1–3 h):
   `python tools/train_rfdetr.py --size medium` (recommended) and optionally
   `python tools/train_yolo.py --model yolo11s.pt`. The weights land in `models/`.
5. **Fit fusion** on a *separate* set of gold plates:
   `python tools/tune_fusion.py --images data/tune --gold eval/gold`
6. **Evaluate** on held-out plates, per engine and ensemble:
   `python tools/evaluate.py --images data/test --gold eval/gold --per-engine`

Plates that are empty, low-density, dense/TNTC, contaminated or bubbly all
belong in the gold set; the review flags need them too.

---

## HTTP API

`POST /predict` (alias `POST /v1/count`), multipart: `file` (the image) and
optional `include_image` (default `true`: base64 PNG overlay, only when at
least one colony is found). `Authorization: Bearer <CD_API_KEY>` if a key is set.

```jsonc
{
  "model_version": "colony-detector-1.0.0[classical-1.0+yolo:1a2b3c4d]",
  "image": {"width": 1538, "height": 1536},
  "counts": {"colony": 66, "fuzzy_colony": 1},
  "total_count": 67,
  "tntc": false,
  "inference_ms": 1510,
  "boxes": [{
    "class_name": "colony", "confidence": 0.87,
    "xyxy": [482, 98, 494, 110], "centroid": [488, 104], "radius_px": 6,
    "contour": [[484, 99], ...], "shape_source": "contour", "count": 1,
    "sources": ["classical", "yolo"]
  }],
  "plate": {"found": true, "method": "hough", "center": [762.8, 785.8], "radius_px": 737.8, "agar_radius_px": 719.0},
  "quality": {"focus_score": 0.52, "glare_score": 0.0, "plate_found": true, "overgrowth_detected": false},
  "confidence": {"overall_score": 0.84, "needs_review": false, "reason_codes": []},
  "engines": {"classical": {"status": "ok", "count": 67, "ms": 1400}},
  "diagnostics": {"polarity": "dark", "typical_radius_px": 10.1, "rejected": {"bubble": 0, "stroke": 11, "rim": 30}},
  "annotated_image": "iVBORw0KGgo..."
}
```

`GET /health` reports the loaded engines, model version and any engine that
failed to load.

### MicroID

Set `AI_BACKEND=remote`, `AI_SERVICE_URL=http://<host>:8000` and, if used,
`AI_SERVICE_API_KEY` (= `CD_API_KEY`). `tests/test_service.py` replicates
MicroID's `RemoteColonyCounter._to_contract` read path, so a contract break
fails CI.

### Configuration (env)

| var | default | meaning |
| --- | --- | --- |
| `CD_ENGINES` | `auto` | `auto` = classical + every engine whose weights exist, or a list such as `classical,yolo` |
| `CD_MODELS_DIR` | `./models` | where weights and `fusion.json` are looked up |
| `CD_YOLO_WEIGHTS` / `CD_RFDETR_WEIGHTS` / `CD_CELLPOSE_WEIGHTS` | auto-detected | explicit weight paths |
| `CD_TILE_SIZE` / `CD_TILE_OVERLAP` | 640 / 0.25 | tiled inference (must match training) |
| `CD_DETECTOR_CONF` | 0.2 | minimum box-detector confidence (YOLO, Grounding DINO) |
| `CD_RFDETR_CONF` | 0.4 | minimum RF-DETR confidence |
| `CD_FUSE_THRESHOLD` | 0.25 (or `fusion.json`) | consensus threshold |
| `CD_OUTLIER_RATIO` | 2.5 | out-of-domain guard |
| `CD_TNTC_LIMIT` | 300 | too-numerous-to-count limit |
| `CD_POLARITY` | auto | force `dark` / `bright` / `both` colonies |
| `CD_DEVICE` | "" | e.g. `cuda:0` |
| `CD_API_KEY` | "" | require a bearer token |

---

## Docker

```bash
docker build -t colony-detector .                        # classical engine only (small image)
docker build --build-arg INSTALL_ML=1 -t colony-detector:ml .
docker run -p 8000:8000 -v $PWD/models:/app/models colony-detector:ml
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -q          # accuracy regression, artifacts, fusion, tiling, HTTP contract
ruff check .
```

## References

- MicroID `backend/app/services/colony_counter/remote_client.py`
  (response contract) and `backend/docs/TRD.md` (review and annotation
  rules).
- D. Korporaal et al., *Colony Grounded SAM2: Zero-shot detection and
  segmentation of bacterial colonies using foundation models*, arXiv
  2603.13393 (2026). This is the basis of the `gdino` engine and of using
  zero-shot detection for label bootstrapping.
- Majchrowska et al., *AGAR: a microbial colony dataset for deep learning
  detection*, arXiv 2108.01234. This is a public pre-training source that
  can be converted with `build_training_set.py --yolo-labels`.
- Geissmann, *OpenCFU* (sample plates used for the real-image checks above).
