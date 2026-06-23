# Perception→Risk Code Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the drone avoidance planner with a monocular dashcam **risk assessor** that tags each tracked object SAFE/CAUTION/DANGER, leaving the detect→track pipeline intact and runnable on any video.

**Architecture:** Keep `BaseDetector.detect → BaseTracker.update`, then swap the third stage: delete `src/avoidance/` and add `src/risk/` with `BaseRiskAssessor` + `RiskZoneAssessor`. Risk = object's ground-contact point inside a frame-relative ego-path trapezoid (WHERE) combined with a bbox-area-growth closing proxy from the Kalman filter (HOW CLOSE). The tracker gains one backward-compatible field (`scale_velocity`) to surface the area-growth rate the assessor needs.

**Tech Stack:** Python 3, NumPy, OpenCV (`cv2`), filterpy Kalman, pytest. No new dependencies.

This is **Plan 1 of 3** for the vehicle-avoidance pivot (spec: `docs/superpowers/specs/2026-06-22-vehicle-avoidance-pivot-design.md`). Plan 2 = BDD100K data pipeline; Plan 3 = training + eval + docs/Streamlit rewrite. This plan is pure local code, fully testable without any dataset or GPU.

## Global Constraints

- **Line length: 88** (Black default); CI runs `black --check .` then `flake8 .` — both must pass.
- **flake8 ignores** `E203,W503` (Black compatibility); `prototype/` is excluded from flake8.
- **Imports use `src.*` absolute paths**; run pytest/scripts from the repo root.
- **No new runtime dependencies** beyond what `requirements.txt` already pins.
- Risk labels are exactly the strings `"SAFE"`, `"CAUTION"`, `"DANGER"`.

---

### Task 1: Surface the bbox-area-growth closing proxy on Track

The Kalman filter already estimates area velocity in state `kf.x[6]` (ds/dt). Expose it as `Track.scale_velocity` so the assessor can compute a closing proxy without re-deriving it. New field is optional (defaults to `0.0`) so existing `Track(...)` call sites and tests keep working.

**Files:**
- Modify: `src/tracking/tracker.py` (the `Track` dataclass)
- Modify: `src/tracking/kalman_tracker.py` (`KalmanBoxTracker.get_scale_velocity`, populate in `KalmanTracker.update`)
- Test: `tests/test_tracking.py`

**Interfaces:**
- Produces: `Track.scale_velocity: float` (area pixels²/frame; positive = box growing = object approaching). `KalmanBoxTracker.get_scale_velocity() -> float`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tracking.py`:

```python
def test_track_has_scale_velocity_default():
    t = Track(track_id=1, bbox=[10, 10, 50, 50])
    assert t.scale_velocity == 0.0


def test_kalman_tracker_reports_growing_scale_velocity():
    from src.detection.detector import Detection
    from src.tracking.kalman_tracker import KalmanTracker

    tracker = KalmanTracker(min_hits=1)
    # Box grows each frame (area increasing) -> object approaching the camera.
    tracker.update([Detection(bbox=[100, 100, 200, 200], confidence=0.9, class_id=0)])
    tracker.update([Detection(bbox=[95, 95, 210, 210], confidence=0.9, class_id=0)])
    tracks = tracker.update(
        [Detection(bbox=[90, 90, 220, 220], confidence=0.9, class_id=0)]
    )
    assert len(tracks) == 1
    assert tracks[0].scale_velocity > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracking.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword` is NOT expected; instead `AttributeError`/assertion: `Track` has no `scale_velocity`.

- [ ] **Step 3: Add the field to the Track dataclass**

In `src/tracking/tracker.py`, add the field after `velocity`:

```python
@dataclass
class Track:
    track_id: int
    bbox: List[float]
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    scale_velocity: float = 0.0
    age: int = 0
```

- [ ] **Step 4: Expose and populate scale_velocity in the Kalman tracker**

In `src/tracking/kalman_tracker.py`, add a method to `KalmanBoxTracker` (next to `get_velocity`):

```python
    def get_scale_velocity(self):
        """Returns the area (scale) velocity estimate ds/dt."""
        return float(self.kf.x[6, 0])
```

Then in `KalmanTracker.update`, set it on the emitted `Track` (the `ret.append(Track(...))` block):

```python
                ret.append(
                    Track(
                        track_id=trk.id,
                        bbox=[float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                        velocity=trk.get_velocity(),
                        scale_velocity=trk.get_scale_velocity(),
                        age=trk.age,
                    )
                )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tracking.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/tracking/tracker.py src/tracking/kalman_tracker.py tests/test_tracking.py
git commit -m "feat(tracking): expose bbox-area growth as Track.scale_velocity"
```

---

### Task 2: Create the risk package contract

Define the base interface and the `RiskedTrack` output type. No logic yet — just the contract Tasks 3–5 build on.

**Files:**
- Create: `src/risk/__init__.py`
- Create: `src/risk/assessor.py`
- Test: `tests/test_risk.py`

**Interfaces:**
- Produces:
  - `RiskLevel.SAFE`, `RiskLevel.CAUTION`, `RiskLevel.DANGER` (str constants)
  - `RiskedTrack(track: Track, risk: str, in_path: bool, ttc_proxy: float)` (dataclass)
  - `BaseRiskAssessor.assess(tracks: List[Track], frame_shape: Tuple[int, int]) -> List[RiskedTrack]` where `frame_shape = (height, width)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_risk.py`:

```python
from src.risk.assessor import BaseRiskAssessor, RiskedTrack, RiskLevel
from src.tracking.tracker import Track


def test_risk_level_constants():
    assert RiskLevel.SAFE == "SAFE"
    assert RiskLevel.CAUTION == "CAUTION"
    assert RiskLevel.DANGER == "DANGER"


def test_risked_track_wraps_a_track():
    t = Track(track_id=7, bbox=[0, 0, 10, 10])
    rt = RiskedTrack(track=t, risk=RiskLevel.SAFE, in_path=False, ttc_proxy=0.0)
    assert rt.track.track_id == 7
    assert rt.risk == "SAFE"
    assert rt.in_path is False


def test_base_assessor_is_abstract():
    import pytest

    with pytest.raises(NotImplementedError):
        BaseRiskAssessor().assess([], (640, 640))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_risk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.risk'`.

- [ ] **Step 3: Create the package**

Create empty `src/risk/__init__.py` (zero bytes).

Create `src/risk/assessor.py`:

```python
from dataclasses import dataclass
from typing import List, Tuple

from src.tracking.tracker import Track


class RiskLevel:
    """String constants for the three advisory risk levels."""

    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"


@dataclass
class RiskedTrack:
    track: Track
    risk: str
    in_path: bool
    ttc_proxy: float


class BaseRiskAssessor:
    def assess(
        self, tracks: List[Track], frame_shape: Tuple[int, int]
    ) -> List[RiskedTrack]:
        """Tag each track SAFE/CAUTION/DANGER. frame_shape is (height, width)."""
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_risk.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/risk/__init__.py src/risk/assessor.py tests/test_risk.py
git commit -m "feat(risk): add BaseRiskAssessor + RiskedTrack contract"
```

---

### Task 3: Implement RiskZoneAssessor (ego-path + risk labels)

The runtime engine (Approach A). Ego-path is a trapezoid: narrow at the horizon line, wide at the bottom of the frame, centered horizontally. An object's ground-contact point is its bbox **bottom-center** `((x1+x2)/2, y2)`. In-path = that point falls inside the trapezoid. Closing proxy `ttc_proxy = scale_velocity / area` (fractional area growth per frame; 0 if area ≤ 0). Labels: off-path → SAFE; in-path and (big OR growing) → DANGER; in-path otherwise → CAUTION. All thresholds are constructor params with defaults.

**Files:**
- Create: `src/risk/zone_assessor.py`
- Test: `tests/test_risk.py` (extend)

**Interfaces:**
- Consumes: `Track.bbox`, `Track.scale_velocity` (Task 1); `RiskedTrack`, `RiskLevel`, `BaseRiskAssessor` (Task 2).
- Produces: `RiskZoneAssessor(horizon_ratio=0.5, top_width=0.1, bottom_width=0.9, large_area_frac=0.05, growth_thresh=0.02)` with `.assess(tracks, frame_shape) -> List[RiskedTrack]` and helper `.ego_path_polygon(frame_shape) -> List[Tuple[int, int]]` (4 points, for the visualizer).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_risk.py`:

```python
from src.risk.zone_assessor import RiskZoneAssessor

FRAME = (640, 640)  # (height, width)


def _assess_one(track):
    return RiskZoneAssessor().assess([track], FRAME)[0]


def test_off_path_object_is_safe():
    # Bottom-center x=10 is left of the trapezoid edge (half-width ~272 at y=620,
    # so the in-path band is x in [48, 592]); 10 is outside -> off-path.
    t = Track(track_id=1, bbox=[0, 580, 20, 620])  # bottom-center x=10
    rt = _assess_one(t)
    assert rt.in_path is False
    assert rt.risk == RiskLevel.SAFE


def test_object_above_horizon_is_safe():
    # Bottom-center y=100 is above the horizon line (320) -> not in path.
    t = Track(track_id=2, bbox=[300, 60, 340, 100])
    rt = _assess_one(t)
    assert rt.in_path is False
    assert rt.risk == RiskLevel.SAFE


def test_large_in_path_object_is_danger():
    # Centered, near bottom, large box (~0.1 of frame area).
    t = Track(track_id=3, bbox=[220, 420, 420, 620])  # 200x200, bottom-center (320,620)
    rt = _assess_one(t)
    assert rt.in_path is True
    assert rt.risk == RiskLevel.DANGER


def test_small_growing_in_path_object_is_danger():
    # Small box but area growing fast: growth_rate = 20/400 = 0.05 >= 0.02.
    t = Track(track_id=4, bbox=[310, 590, 330, 610], scale_velocity=20.0)
    rt = _assess_one(t)
    assert rt.in_path is True
    assert rt.risk == RiskLevel.DANGER
    assert rt.ttc_proxy > 0


def test_small_stable_in_path_object_is_caution():
    # Small box (area frac ~0.001 < 0.05), not growing -> caution.
    t = Track(track_id=5, bbox=[310, 590, 330, 610], scale_velocity=0.0)
    rt = _assess_one(t)
    assert rt.in_path is True
    assert rt.risk == RiskLevel.CAUTION


def test_ego_path_polygon_has_four_points():
    poly = RiskZoneAssessor().ego_path_polygon(FRAME)
    assert len(poly) == 4
    assert all(len(pt) == 2 for pt in poly)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_risk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.risk.zone_assessor'`.

- [ ] **Step 3: Implement the assessor**

Create `src/risk/zone_assessor.py`:

```python
from typing import List, Tuple

from src.risk.assessor import BaseRiskAssessor, RiskedTrack, RiskLevel
from src.tracking.tracker import Track


class RiskZoneAssessor(BaseRiskAssessor):
    """Monocular dashcam risk assessor (Approach A).

    WHERE: is the object's ground-contact point inside the ego-path trapezoid?
    HOW CLOSE: is its bounding box large or growing (Kalman area velocity)?
    """

    def __init__(
        self,
        horizon_ratio: float = 0.5,
        top_width: float = 0.1,
        bottom_width: float = 0.9,
        large_area_frac: float = 0.05,
        growth_thresh: float = 0.02,
    ):
        self.horizon_ratio = horizon_ratio
        self.top_width = top_width
        self.bottom_width = bottom_width
        self.large_area_frac = large_area_frac
        self.growth_thresh = growth_thresh

    def _half_width_at(self, py: float, h: int, w: int):
        """Half-width of the trapezoid at image row py, or None if above horizon."""
        horizon_y = self.horizon_ratio * h
        if py < horizon_y:
            return None
        t = (py - horizon_y) / max(h - horizon_y, 1e-6)
        t = min(max(t, 0.0), 1.0)
        top_half = self.top_width * w / 2.0
        bottom_half = self.bottom_width * w / 2.0
        return top_half + (bottom_half - top_half) * t

    def ego_path_polygon(self, frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Four trapezoid corners (top-left, top-right, bottom-right, bottom-left)."""
        h, w = frame_shape
        cx = w / 2.0
        horizon_y = self.horizon_ratio * h
        top_half = self.top_width * w / 2.0
        bottom_half = self.bottom_width * w / 2.0
        return [
            (int(cx - top_half), int(horizon_y)),
            (int(cx + top_half), int(horizon_y)),
            (int(cx + bottom_half), int(h)),
            (int(cx - bottom_half), int(h)),
        ]

    def assess(
        self, tracks: List[Track], frame_shape: Tuple[int, int]
    ) -> List[RiskedTrack]:
        h, w = frame_shape
        cx = w / 2.0
        frame_area = float(h * w)
        out = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            contact_x = (x1 + x2) / 2.0
            contact_y = y2
            area = max((x2 - x1) * (y2 - y1), 0.0)

            half_width = self._half_width_at(contact_y, h, w)
            in_path = half_width is not None and abs(contact_x - cx) <= half_width

            growth_rate = track.scale_velocity / area if area > 0 else 0.0
            ttc_proxy = growth_rate

            if not in_path:
                risk = RiskLevel.SAFE
            else:
                big = area / frame_area >= self.large_area_frac
                growing = growth_rate >= self.growth_thresh
                risk = RiskLevel.DANGER if (big or growing) else RiskLevel.CAUTION

            out.append(
                RiskedTrack(
                    track=track,
                    risk=risk,
                    in_path=in_path,
                    ttc_proxy=ttc_proxy,
                )
            )
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_risk.py -v`
Expected: PASS (all risk tests, including the 6 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/risk/zone_assessor.py tests/test_risk.py
git commit -m "feat(risk): implement RiskZoneAssessor ego-path + risk labels"
```

---

### Task 4: Risk overlay in the visualizer

Add `draw_risk` rendering the ego-path trapezoid plus color-coded boxes (green=SAFE, yellow=CAUTION, red=DANGER) with a per-object risk label. Keep the old `draw_tracks` removed (it printed drone yaw/altitude). The visualizer must not import anything drone-specific.

**Files:**
- Modify: `src/utils/visualizer.py` (replace `draw_tracks` with `draw_risk`)
- Test: `tests/test_visualizer.py` (create)

**Interfaces:**
- Consumes: `RiskedTrack`, `RiskLevel` (Task 2); `RiskZoneAssessor.ego_path_polygon` (Task 3).
- Produces: `draw_risk(frame: np.ndarray, risked: List[RiskedTrack], ego_polygon: Optional[List[Tuple[int, int]]] = None) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualizer.py`:

```python
import numpy as np

from src.risk.assessor import RiskedTrack, RiskLevel
from src.tracking.tracker import Track
from src.utils.visualizer import draw_risk


def test_draw_risk_returns_same_shape_without_mutating_input():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    risked = [
        RiskedTrack(
            track=Track(track_id=1, bbox=[100, 300, 200, 460]),
            risk=RiskLevel.DANGER,
            in_path=True,
            ttc_proxy=0.1,
        )
    ]
    poly = [(310, 240), (330, 240), (600, 480), (40, 480)]
    out = draw_risk(frame, risked, ego_polygon=poly)
    assert out.shape == frame.shape
    # Original frame is untouched; output has drawn pixels.
    assert frame.sum() == 0
    assert out.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualizer.py -v`
Expected: FAIL — `ImportError: cannot import name 'draw_risk'`.

- [ ] **Step 3: Replace the visualizer body**

Overwrite `src/utils/visualizer.py`:

```python
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.risk.assessor import RiskedTrack, RiskLevel

_COLORS = {
    RiskLevel.SAFE: (0, 200, 0),      # green
    RiskLevel.CAUTION: (0, 215, 255),  # amber
    RiskLevel.DANGER: (0, 0, 255),     # red
}


def draw_risk(
    frame: np.ndarray,
    risked: List[RiskedTrack],
    ego_polygon: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Render the ego-path region and color-coded per-object risk boxes."""
    out = frame.copy()

    if ego_polygon:
        pts = np.array(ego_polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (255, 180, 0))
        out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)
        cv2.polylines(out, [pts], True, (255, 180, 0), 2)

    for rt in risked:
        x1, y1, x2, y2 = [int(v) for v in rt.track.bbox]
        color = _COLORS.get(rt.risk, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{rt.risk} ID:{rt.track.track_id}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/visualizer.py tests/test_visualizer.py
git commit -m "feat(viz): draw_risk renders ego-path + color-coded risk boxes"
```

---

### Task 5: Rewire main.py, delete src/avoidance/, update integration test

Swap the planner for the assessor in the demo pipeline, render with `draw_risk`, delete the drone avoidance package, and rewrite the integration test (which imported `GeometricPlanner`).

**Files:**
- Modify: `main.py`
- Delete: `src/avoidance/planner.py`, `src/avoidance/geometric_planner.py`, `src/avoidance/__init__.py`
- Modify: `tests/test_integration.py` (rewrite drone-planner tests as risk-assessor tests)

**Interfaces:**
- Consumes: `RiskZoneAssessor`, `RiskLevel` (Tasks 2–3); `draw_risk` (Task 4).

- [ ] **Step 1: Rewrite the integration test (failing)**

Overwrite `tests/test_integration.py`:

```python
from src.detection.detector import Detection
from src.risk.assessor import RiskLevel
from src.risk.zone_assessor import RiskZoneAssessor
from src.tracking.kalman_tracker import KalmanTracker


def test_tracker_assessor_integration():
    """Tracker output flows into the risk assessor without error."""
    tracker = KalmanTracker(min_hits=1)
    assessor = RiskZoneAssessor()

    det1 = Detection(bbox=[280, 400, 360, 480], confidence=0.9, class_id=0)
    tracker.update([det1])
    det2 = Detection(bbox=[275, 395, 365, 490], confidence=0.9, class_id=0)
    tracks = tracker.update([det2])

    assert len(tracks) == 1
    risked = assessor.assess(tracks, frame_shape=(640, 640))
    assert len(risked) == 1
    assert risked[0].risk in (RiskLevel.SAFE, RiskLevel.CAUTION, RiskLevel.DANGER)


def test_centered_approaching_object_is_in_path():
    """A centered object near the bottom is flagged in-path."""
    tracker = KalmanTracker(min_hits=1)
    assessor = RiskZoneAssessor()
    det = Detection(bbox=[220, 420, 420, 620], confidence=0.9, class_id=0)
    tracks = tracker.update([det])
    risked = assessor.assess(tracks, frame_shape=(640, 640))
    assert risked[0].in_path is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.avoidance'` is NOT it; the new test imports `src.risk.*` which exist, but it will fail/err only if something is wrong. Run it now and expect PASS for the new test EXCEPT the suite still contains the deleted-package import elsewhere. (If the new test passes immediately, that is fine — proceed to delete the package in Step 3, which is the real change.)

- [ ] **Step 3: Delete the drone avoidance package**

```bash
git rm src/avoidance/planner.py src/avoidance/geometric_planner.py src/avoidance/__init__.py
```

- [ ] **Step 4: Rewire main.py**

Apply these edits to `main.py`:

Replace the import line
```python
from src.avoidance.geometric_planner import GeometricPlanner
from src.utils.visualizer import draw_tracks
```
with
```python
from src.risk.zone_assessor import RiskZoneAssessor
from src.utils.visualizer import draw_risk
```

Change the `argparse` description and window title from drone wording to
`"Vehicle pedestrian/vehicle risk-advisory demo"` and `"Vehicle Risk Advisory"`.

In `run_pipeline`, replace the `planner` parameter with `assessor` and the per-frame body:
```python
def run_pipeline(
    frames,
    detector,
    tracker,
    assessor,
    delay_ms: int,
    reset_per_frame: bool = False,
    save_dir: Path = None,
    video_out: cv2.VideoWriter = None,
):
    print("Starting pipeline. Press 'q' to quit, any key to advance.")
    for name, frame in frames:
        if reset_per_frame:
            tracker.reset()
        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        risked = assessor.assess(tracks, frame_shape=(h, w))
        vis = draw_risk(frame, risked, ego_polygon=assessor.ego_path_polygon((h, w)))

        if video_out:
            video_out.write(vis)
        elif save_dir:
            out_path = save_dir / name
            cv2.imwrite(str(out_path), vis)
            print(f"Saved {out_path}")

        cv2.imshow("Vehicle Risk Advisory", vis)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
```

In `main()`, replace the planner construction
```python
    planner = GeometricPlanner(frame_width=w, frame_height=h)
```
with
```python
    assessor = RiskZoneAssessor()
```
and update the `run_pipeline(...)` call to pass `assessor` instead of `planner`
(keep all other arguments identical).

- [ ] **Step 5: Run the full suite + lint**

Run:
```bash
pytest -v
black --check .
flake8 .
```
Expected: all tests PASS; black reports nothing to reformat; flake8 reports no errors. If black flags files, run `black .` and re-commit.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_integration.py
git commit -m "feat: wire RiskZoneAssessor into demo pipeline; remove drone avoidance package"
```

---

### Task 6: Verify the demo runs end-to-end on a real frame

A smoke test that the wired pipeline produces an annotated frame, proving Plan 1 yields working software (no dataset needed — uses COCO-pretrained `yolov8m.pt`, which Ultralytics auto-downloads, or any small video the user has).

**Files:**
- Create: `tests/test_demo_smoke.py`

**Interfaces:**
- Consumes: `YoloDetector`, `KalmanTracker`, `RiskZoneAssessor`, `draw_risk`.

- [ ] **Step 1: Write the smoke test**

Create `tests/test_demo_smoke.py`:

```python
import numpy as np
import pytest

from src.risk.zone_assessor import RiskZoneAssessor
from src.tracking.kalman_tracker import KalmanTracker
from src.utils.visualizer import draw_risk


def test_pipeline_smoke_without_detector():
    """Track + assess + draw on a synthetic frame, no model download required."""
    from src.detection.detector import Detection

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = KalmanTracker(min_hits=1)
    assessor = RiskZoneAssessor()

    tracks = tracker.update(
        [Detection(bbox=[280, 300, 360, 440], confidence=0.9, class_id=0)]
    )
    risked = assessor.assess(tracks, frame_shape=frame.shape[:2])
    vis = draw_risk(frame, risked, ego_polygon=assessor.ego_path_polygon((480, 640)))

    assert vis.shape == frame.shape
    assert vis.sum() > 0
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_demo_smoke.py -v`
Expected: PASS.

- [ ] **Step 3: Run the entire suite + lint one final time**

Run:
```bash
pytest -v
black --check .
flake8 .
```
Expected: all PASS / clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_demo_smoke.py
git commit -m "test: end-to-end smoke test for track->risk->draw pipeline"
```

---

## Notes for the implementer

- **Manual demo check (optional, not a test):** with a webcam or any clip,
  `python main.py --source <video.mp4>` should show the blue ego-path trapezoid
  and red/amber/green boxes. Requires a display; CI does not run this.
- **Class-aware risk is out of scope here.** `Track` carries no `class_id` today,
  so risk is class-agnostic per the spec. Propagating detection class into tracks
  (to weight pedestrians higher / label boxes by type) is a later enhancement.
- **Thresholds** (`large_area_frac`, `growth_thresh`, trapezoid widths) are
  constructor defaults now; Plan 3 may move them into a YAML config alongside the
  training configs.
