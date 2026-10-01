# Answer key review (138 test plates, counting rule B)

`candidates.json` is the output of `tools/answer_key.py prepare` on the test
split (paths relative to the dataset root); `decisions.json` holds the
reviewer's call for every disputed spot (`colony` / `not` / `small` /
`unsure`), corrections to auto-accepted spots (`auto:<image>:<index>`) and
plate flags (`plate:<image>`: `tntc` / `overgrown`).

Rebuild the gold files with:

    cp eval/answer_key_test/*.json review/test/   # candidates + decisions
    python tools/answer_key.py apply --out review/test --gold eval/gold_test_ruleB

The review was done by eye on zoomed crops by the AI assistant under rule B
(count >= 0.5 mm, 0.3-0.5 mm specks flagged, smaller ignored). Spots that
could not be decided are `unsure` and the scorer accepts either reading.
A microbiologist can overrule any decision in `decisions.json` and re-run
`apply`.
