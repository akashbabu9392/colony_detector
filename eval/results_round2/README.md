# Round-2 results (RF-DETR Medium, Kaggle T4, 2026-10-01)

Outputs of `notebooks/train_rfdetr_kaggle_round2.ipynb`, extracted from the
notebook's `results.zip` (also on the `dataset-v2` release). Round 2
fine-tunes the round-1 Medium model for 6 epochs on **cleaned labels** (boxes
under 0.3 mm dropped, colonies the model was >= 0.9 sure of added where the
labels missed them), then refits the cut-offs on the reviewed answer key.

| file | what it is | written by |
| --- | --- | --- |
| `train_log.txt` | 6 epochs, best EMA mAP metric 0.8089 | `tools/train_rfdetr.py` |
| `clean_log.txt` | 375 boxes < 0.3 mm dropped, 28 confident detections added; 8 046 train / 1 192 valid tiles | `tools/build_training_set.py` |
| `gold_report.txt` | fitted cut-offs, per-plate metrics and the 2-fold cross-validated estimate | `tools/score_gold.py --tune` |
| `gold_results.csv` | one row per answer-key plate for the fitted cut | `tools/score_gold.py --csv` |
| `test_dets.pkl` | detection cache for the 135 countable test plates | `tools/score_gold.py` |

The fitted cut for these weights is `models/detector.json` (`conf` 0.75,
`dense_conf` 0.35, `dense_min` 15): exact 79.3 %, within +/-1 94.8 %,
within +/-2 97.8 %, MAE 0.41, colony F1 0.935 — 75.9 / 92.4 / 95.6 %, MAE
0.49, F1 0.915 cross-validated.

The fine-tuned checkpoint (`rfdetr_tiles.pth`, 128 MB) is **not in git**:
GitHub rejects blobs over 100 MB and `models/README.md` keeps weights out of
the tree. Download it from the `dataset-v2` release (`results.zip`) or the
Kaggle notebook output and drop it in `models/`.

Re-score these plates offline from the committed cache (no weights, no
images, no GPU needed):

    python tools/score_gold.py --gold eval/gold_test_ruleB \
        --cache eval/results_round2/test_dets.pkl --detector models/detector.json \
        --csv runs/gold_results_round2.csv

Re-fit the cut-offs from the cache:

    python tools/score_gold.py --gold eval/gold_test_ruleB \
        --cache eval/results_round2/test_dets.pkl --tune models/detector.json