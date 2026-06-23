# Plan 3 (Vehicle Training/Eval/Docs) — HANDOFF

**Date:** 2026-06-23 · Branch: `pivot/vehicle-avoidance` · Plan: `2026-06-23-vehicle-training-eval-docs.md`
**Executed via:** superpowers:subagent-driven-development (fresh implementer + reviewer per task).

## Status: all CODE/DOCS tasks done & reviewed. GPU/KITTI runs + final review remain.

**Verified now:** `pytest -q` → **54 passed**; `flake8 .` → clean; working tree clean (except two pre-existing untracked Plan 2 docs, see bottom).

### Done & committed (Plan 3 range `6a0058c..8c5b679`, 11 commits)
Each task: TDD by an implementer subagent, then a reviewer subagent (spec + quality). All approved.

| Task | Commit(s) | What |
|------|-----------|------|
| Plan doc | `6a0058c` | The Plan 3 implementation plan |
| 1 | `d681c2d` | `modal_train.py`: fix `--data-root` → `/vol/processed/bdd100k` (was `/vol/processed`, broken); vehicle naming (`cpv-vehicle-perception`/`cpv-bdd100k`); extracted pure `build_train_cmd()` + 2 tests |
| 3 | `04b259f` | `scripts/evaluate.py`: `--classwise` per-class mAP, data-config guard, drop all `visdrone5.yaml` refs |
| 4 | `a99a278` | `scripts/evaluate_robustness.py` (+test): attribute-sliced mAP by `timeofday`/`weather` from `attributes.csv` |
| 5 | `28ad9c0` | `src/utils/kitti.py` (+test), `scripts/preprocess_kitti.py`, `configs/kitti.yaml`: KITTI→YOLO 3-class converter (field-indexing reviewer-verified correct) |
| 6 | `86a6305` | `scripts/validate_risk.py` (+test): Approach C — risk labels vs KITTI 3D distance (`precision_danger_near`) |
| 7 | `75e7987`,`3e04ce4` | `docs/training_pipeline.md` (446 lines: Modal runbook, eval, KITTI zero-shot, risk validation, locked decisions, R-round map) |
| 8 | `694d024`,`e438e77` | `prototype/web_app.py`: kill stale "Avoidance" copy → risk-advisory/FCW framing; wire detect→track→risk + ego-path overlay; model selector; 3 classes |
| 9 | `8c5b679` | `CLAUDE.md` status table flipped (Plan 3 code done; GPU/KITTI runs operator-pending) |

**Note:** Task 2 was a MANUAL operator runbook (no code) — not dispatched; see below.

## 🟡 REMAINING — pick up here

### 1. FINAL whole-branch review (NOT yet run)
Skipped to conserve budget. Run it next:
```
scripts/review-package 0a21a7b HEAD        # (script in superpowers/subagent-driven-development/scripts)
```
Dispatch the final reviewer (superpowers:requesting-code-review's code-reviewer.md) on **opus** over that package. Feed it the Minor findings rolled up in `.superpowers/sdd/progress.md` (Plan 3 section) for triage.

### 2. Minor cleanups for the final review to decide
- **`scripts/__init__.py`** — added out-of-scope in Task 6, **empty + harmless but unnecessary** (`scripts.*` was already importable via namespace package; the implementer's justification was wrong). Recommend `git rm scripts/__init__.py` and confirm `pytest -q` still 54/54 (it will — Task 4's `from scripts.evaluate_robustness import` worked before it existed).
- **`modal_train.py` docstring** says "~8000 files" — stale; the real processed set is ~80k images (`docs/training_pipeline.md` says ~80k, which is correct). Fix the script comment for consistency.
- Other deferred Minors (cosmetic) are listed per-task in `.superpowers/sdd/progress.md`.

### 3. MANUAL OPERATOR runs (need Modal account + KITTI download) — produce the R3 artifacts
All code is ready; these are the GPU/data steps a human runs. Full commands in `docs/training_pipeline.md`.
- **Training (Modal L4):** upload dataset (`tar czf processed.tar.gz -C data processed` → `modal volume create cpv-bdd100k` → `modal volume put …` → `modal run modal_train.py::extract_dataset`), then `modal run modal_train.py::main --model {yolov8n,yolov8m,rtdetr} --epochs 50 --fresh`, then `modal run modal_train.py::fetch --model <m>` → `models/<m>-best.pt`.
  - ⚠️ The `models/yolov8n-best.pt`/`yolov8m-best.pt` currently on disk are **stale VisDrone (5-class) drone weights** — `--fresh` overwrites them; verify `model.names` has 3 entries before trusting eval.
- **BDD100K test eval:** `scripts/evaluate.py … --classwise --output reports/R3/<m>_metrics.json` (×3 models).
- **Robustness:** `scripts/evaluate_robustness.py --by timeofday|weather …` → `reports/R3/<m>_robustness_*.json`.
- **KITTI:** download raw (registration required) to `data/raw/kitti/{image_2,label_2}` → `python scripts/preprocess_kitti.py` → validate → `scripts/evaluate.py --data configs/kitti.yaml --split val …` (zero-shot) + `python scripts/validate_risk.py --output reports/R3/risk_validation.json`.

### 4. Finish the branch
After final review + (optionally) the operator runs: use **superpowers:finishing-a-development-branch** to merge/PR `pivot/vehicle-avoidance` → `main`.

## Resume map
- **Ledger:** `.superpowers/sdd/progress.md` (Plan 3 section) — per-task commits, review verdicts, all deferred Minors. Trust it + `git log` over memory after compaction.
- **Briefs/reports/diffs:** `.superpowers/sdd/task-N-{brief,report}.md`, `review-*.diff` (git-ignored scratch).
- **merge-base with main:** `0a21a7b`. **Plan 3 HEAD:** `8c5b679`.

## Housekeeping
Two untracked docs predate this session (Plan 2 artifacts, never committed): `docs/superpowers/plans/2026-06-23-bdd100k-data-pipeline.md` and `…-phase2-HANDOFF.md`. They belong in the repo — commit them when convenient (not done here to avoid scope creep).
