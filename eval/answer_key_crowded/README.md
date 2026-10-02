# Crowded training plates: label review

`candidates.json` is `tools/answer_key.py prepare` on the 297 train/valid
plates with more than 5 labelled colonies, against the round-2 model
(`model-v2`): 15,439 spots where labels and model agree, and 2,781 disputed
spots on 273 plates (1,698 found only by the model, 952 only labelled, 131
larger than 25 mm). Paths are relative to the dataset root.

The disputed spots are reviewed on the click-through page built by
`answer_key.py web` (`tools/review_page/index.html`). When the review is
done, export the page's `plates` documents and run:

    cp eval/answer_key_crowded/candidates.json review/crowded/   # restore absolute paths first (see answer_key_test)
    python tools/answer_key.py from-web --out review/crowded --docs plates.json
    python tools/answer_key.py apply --out review/crowded --gold eval/reviewed_train
    python tools/build_training_set.py --dataset data/raw --reviewed eval/reviewed_train ...

## Reviewers without a claude.ai account: offline kit

    python tools/answer_key.py kit --out review/crowded --kit review/colony-review-kit --batches 6

writes `colony-review-kit.zip` (about 45 MB): the same page with its data
inlined, the plate photos and crop sheets, and `READ ME FIRST.txt`. Each
technician unzips it, opens `index.html` in Chrome or Edge, types a name,
picks a batch and clicks **Save decisions file** when done. Merge all
returned files (and, if used, the online page's documents) in one go:

    python tools/answer_key.py from-web --out review/crowded --docs crowded-review_batch*.json
