# Phase 2 (BDD100K data pipeline) — HANDOFF

**Date:** 2026-06-23 · Branch: `pivot/vehicle-avoidance` · Plan: `2026-06-23-bdd100k-data-pipeline.md`

## Done & committed (Tasks 1–5)
6 commits, full suite **44 passing**, black/flake8 clean:
- `95f79e4` kaggle dep + `configs/bdd100k.yaml` (nc=3)
- `1e6d3fe` JSON→YOLO conversion core (`src/utils/bdd100k.py`)
- `cdeaa34` collect/split/materialize helpers
- `5e18c9d` rewrite `scripts/preprocess.py`
- `437e1d4` sys.path shim so `python scripts/X.py` runs bare (also fixed pre-existing bug in `validate_data.py`)
- `25059f9` `scripts/download_bdd100k.py`

Data downloaded & extracted (7.6 GB). Real paths (differ from plan guess):
- images: `data/raw/bdd100k/bdd100k/images/100k/{train,val,test}` — **train is SHARDED into subdirs `trainA/ trainB/ …`**
- labels: `data/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{train,val}.json`

## 🔴 BLOCKER — fix before Task 6 is real
`collect_frames` resolves images as `images_dir / fl.name` (flat). solesensei's **train** images are nested in subfolders, so only 1,154 of 70k matched → broken splits (train 946 / val 208). Val (flat) was fine → test 10k is correct. The `data/processed/bdd100k/` on disk now is from this broken run; it gets overwritten on re-run.

**Fix (1 function, in `src/utils/bdd100k.py`):** make `collect_frames` build a recursive index instead of a flat join:
```python
def collect_frames(images_dir, labels_json):
    index = {p.name: p for p in Path(images_dir).rglob("*.jpg")}  # recursive
    pairs, missing = [], 0
    for entry in load_bdd_json(labels_json):
        fl = convert_frame(entry)
        img = index.get(fl.name)
        if img is None:
            missing += 1; continue
        pairs.append((img, fl))
    if missing: print(f"  warning: {missing} frames had no image in {images_dir}")
    return pairs
```
Add a TDD test (nested-subdir tree) first. Then re-run preprocess with the real paths above → expect ~57k train / ~13k val / 10k test. Re-validate all 3 at `--num-classes 3` (currently all PASS but on the tiny broken subset).

## Remaining
- **Task 6** (re-run after fix): `data/processed/bdd100k/{train,val,test}` + `attributes.csv`, all splits PASS.
- **Task 7:** write `docs/bdd100k_data_validation.md` (R1 report: split sizes, class dist, PASS) + flip CLAUDE.md status table Plan 2→done, Plan 3→next. Then `superpowers:finishing-a-development-branch`.

## Notes
- Class dist healthy & imbalanced as expected (vehicle≫person≫two_wheeler): test split 108363 / 13911 / 1459.
- All images 1280×720. Images **symlinked** (not copied). `seg/` deleted to save disk; ~14 GB free.
- Resume execution via `superpowers:subagent-driven-development`.
</content>
