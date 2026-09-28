# Gold-standard hand counts

One JSON per plate image, written by `tools/hand_count.py`:

```json
{"image": "plate_017.jpg", "count": 57, "points": [[412.5, 233.0, 7.5], ...]}
```

`points` are colony centres `[x, y, radius]` in image pixels. A file with only
`count` is fine for count metrics; points also enable precision/recall/F1 and
can be turned into training boxes by `tools/build_training_set.py`.

Keep plates used for training or `tools/tune_fusion.py` separate from the
plates you report accuracy on.
