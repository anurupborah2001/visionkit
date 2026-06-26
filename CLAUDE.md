# CLAUDE.md — VisionKit Developer Guide

This file provides project context, architecture notes, and conventions for working on VisionKit with Claude Code.

---

## Project Overview

**VisionKit** is a Python computer vision library wrapping [MediaPipe Tasks API](https://developers.google.com/mediapipe/solutions/guide) with developer-friendly abstractions. It is structured as a package under `visionkit/` with two primary namespaces:

- `visionkit.lib` — detector and segmentation classes
- `visionkit.capture` — video/screen capture utilities and loop templates

Package management uses **uv** (`pyproject.toml` + `uv.lock`). Python >= 3.11.8 required.

---

## Repository Layout

```
visionkit/           # installable Python package
  __init__.py        # exposes __version__ — managed by python-semantic-release
  lib/               # detector classes (MediaPipe Tasks API)
  capture/           # capture utilities (OpenCV, mss)
  utility/           # shared helpers
tests/               # pytest test suite (mirrors package structure)
.github/
  workflows/         # CI/CD GitHub Actions
.pre-commit-config.yaml  # pre-commit hook definitions
Makefile             # developer convenience targets
main.py              # entry point (currently a stub)
pyproject.toml       # project metadata, dependencies, tool config
uv.lock              # locked dependency tree
.python-version      # pinned Python version (uv reads this)
models/              # NOT committed — each developer provides .tflite/.task files
```

---

## Key Dependencies

| Package | Role |
|---|---|
| `mediapipe >= 0.10.35` | All inference (Tasks API) |
| `opencv-python >= 4.13.0.92` | Frame capture, drawing, image I/O |
| `imageio >= 2.37.3` | GIF recording in `VideoRecorder` |
| `mss >= 10.2.0` | Screen capture in `ScreenCapture` |
| `pyautogui >= 0.9.54` | Window centering in `video_capture_template` |

### Dev-only dependencies (in `[dependency-groups] dev`)

| Package | Role |
|---|---|
| `black`, `isort`, `ruff`, `flake8` | Code formatting and linting |
| `pyupgrade` | Auto-upgrade syntax to target Python version |
| `pre-commit` | Git hook framework |
| `pytest`, `pytest-cov` | Test runner and coverage |
| `python-semantic-release >= 9.21.1` | Automated semver from conventional commits |
| `pip-audit` | Python CVE scanning |
| `twine` | Wheel/sdist verification before publish |
| `hatchling` | Build backend |

---

## Architecture Conventions

### Detector pattern

Every detector in `visionkit/lib/` follows this structure:

1. `__init__` — accepts `model_path`, `running_mode` (`"IMAGE"` | `"VIDEO"`), and task-specific thresholds. Constructs a `vision.*Options` and calls `create_from_options`.
2. `_to_mp_image(image)` — converts BGR numpy array to `mp.Image(SRGB)`. Present on every detector.
3. A primary detection method (e.g. `detect_faces`, `draw_landmarks`, `detect`) that returns `(annotated_image, structured_result)`.
4. Helper methods for filtering, sorting, cropping, measuring.

### Image format contract

- **Input to all detectors:** BGR numpy array (as returned by `cv2.imread` / `cv2.VideoCapture.read`)
- **Internally converted to RGB** before passing to MediaPipe
- **Output:** BGR numpy array (annotated copy, safe to pass to `cv2.imshow`)

Never pass RGB frames to detector methods — they do the conversion internally.

### Running modes

All detectors respect `vision.RunningMode`:

- `IMAGE` — `detector.detect(mp_image)`
- `VIDEO` — `detector.detect_for_video(mp_image, timestamp_ms)`. `PoseDetector` auto-increments `frame_count * 33` if no timestamp is supplied.

Mixing modes within one detector instance is not supported by MediaPipe.

### Model files

Models are **not bundled** in the repository. Each detector expects a `model_path` argument pointing to a local `.tflite` or `.task` file. Convention: store models in `./models/` relative to the project root. Never commit binary model files to git.

---

## Detectors Quick Reference

### FaceDetector (`lib/face_detector.py`)

- MediaPipe Face Detection Tasks API
- Returns detection dicts with keys: `id`, `score`, `bbox` (xywh), `bbox_xyxy`, `center`, `area`, `normalized_keypoints`
- `detect_faces(image)` → `(annotated, detections)`
- Notable helpers: `filter_by_confidence`, `get_largest_face`, `crop_faces`, `sort_faces`, `get_iou`

### FaceMeshDetector (`lib/face_mesh_detector.py`)

- MediaPipe Face Landmarker v2 — 478 landmarks per face
- `face_mesh_detection(img)` → `(annotated, faces, blendshapes, matrices, bboxes)`
  - `faces[i]` — list of 478 `[x, y]` pixel coords
  - `blendshapes[i]` — dict of 52 expression coefficients
  - `matrices[i]` — 4×4 numpy head-pose matrix
- Key methods: `get_emotion(blend)`, `get_eye_gaze_direction`, `get_mouth_openness_ratio`, `get_head_pose_angles(matrix)`, `overlay_ar_filter`
- Landmark index constants are class attributes: `LEFT_IRIS_CENTER`, `NOSE_TIP`, `MOUTH_LEFT`, etc.

### HandDetector (`lib/hand_detector.py`)

- MediaPipe Hand Landmarker — 21 landmarks per hand
- `draw_landmarks(img)` → annotated BGR image (also calls `set_landmarks_image` internally)
- `get_landmarks(img)` → list of hand dicts: `landmarks_list`, `bounding_box`, `center_point`, `hand_type`
- `landmarks_list[i]` format: `[id, x, y, z]`
- Finger tip indices stored in `self.fingerTips = [4, 8, 12, 16, 20]`
- `fingers_up(lm)` → `[thumb, index, middle, ring, little]` (1=up)
- Distance estimation requires prior calibration via `calibration_samples` in `__init__`
- `is_fingers_joined_2` is the more robust variant (adds pixel distance guard)

### PoseDetector (`lib/pose_detector.py`)

- MediaPipe Pose Landmarker — 33 landmarks
- Recommended mode: `VIDEO` (auto-managed timestamps)
- `detect(frame)` → `(annotated, PoseLandmarkerResult)`
- `calculate_angle(img, result, p1, p2, p3)` → `(annotated, angle_degrees)` — angle at joint p2
- Workout rep counter state stored in instance (`rep_count`, `stage`, `rep_times`)
- `detect_exercise` uses a heuristic scoring system; returns one of: `"Bicep Curl"`, `"Shoulder Press"`, `"Squat"`, `"Push-Up"`, `"Lunge"`, `"Standing"`, `"Straight Pose"`
- `output_segmentation_masks=True` required for `draw_segmentation_mask`

### ObjectDetector (`lib/object_detector.py`)

- MediaPipe Object Detector with EfficientDet Lite
- `detect_objects(image)` → annotated BGR image
- `detect(image)` → `(DetectionResult, mp_image)` for raw access
- Supports `category_allowlist` / `category_denylist` to filter classes

### SelfieSegmentation (`lib/selfie_segmentation.py`)

- MediaPipe Image Segmenter (DeepLab V3)
- Multiple compositing methods: `remove_background`, `blur_background`, `replace_background`, `color_background`, `alpha_blend`, `optimize_virtual_background`, `optimize_virtual_background_improved`
- `optimize_virtual_background_improved` keeps only the largest foreground blob (removes background people)
- `confidence_alpha_blend` requires `output_confidence_masks=True`
- `fast_process(frame, scale=0.5)` downscales before inference for speed

### HairSegmentation (`lib/hair_segmentation.py`)

- MediaPipe Image Segmenter (hair model)
- `process(rgb_image)` → segmentation result; access mask via `result.category_mask.numpy_view()`
- Input must be **RGB** (unlike other detectors which accept BGR)

---

## Capture Utilities

### video_capture_template (`capture/video_template.py`)

Drop-in webcam loop. Pass `custom_logic: Callable[[frame], frame]` for processing. Returns when ESC is pressed or the video source is exhausted.

Key parameters:
- `video_source` — `int` (webcam index) or `str` (file path)
- `resolution` — `(width, height)` tuple, sets `CAP_PROP_FRAME_WIDTH/HEIGHT`
- `loop_forever` — rewind file-source when it ends (default `True`)
- `enable_auto_recording` + `record_format` — start recording from frame 1; `"mp4"` or `"gif"`
- `enable_manual_recording` — press `R` to toggle recording; creates a new file each time
- `enable_screenshot` — press `S` to save a frame; also triggers auto-screenshot
- `auto_screenshot_after_seconds` + `auto_screenshot_repeat` — timed screenshot; repeat fires every N seconds when `True`
- `state` — arbitrary `dict` shared across frames and key handlers
- `key_manager` — `KeyEventManager` instance for registering custom key callbacks
- `mouse_callback` — OpenCV mouse event callback

**`KeyEventManager`** (`capture/video_template.py`):
Registers per-key callbacks dispatched inside the loop.
```python
km = KeyEventManager()
km.register(ord("r"), lambda frame, state: ...)   # callback(frame, state)
```
Only one handler per key; re-registering the same key overwrites the previous handler.

**`save_screenshot(frame, output_dir, prefix)`**:
Writes a timestamped PNG to `output_dir` (created if absent). Returns the absolute path string.

**`VideoRecorder`** (`capture/video_recorder.py`):
Dataclass managing MP4 (`cv2.VideoWriter`) or GIF (`imageio.mimsave`) output. Key methods:
`start(frame_shape)`, `write(frame)`, `pause()`, `resume()`, `stop()`, `get_elapsed_time()`.
MP4 expects BGR frames (OpenCV standard); GIF converts BGR→RGB internally.

### ScreenCapture (`capture/screen_capture.py`)

Wraps `mss` for fast screen grabbing. Returns BGR numpy array from `grab()`.

### image_template (`capture/image_template.py`)

Single-image equivalent of `video_capture_template`. Loads from disk, applies optional `custom_logic(frame) -> frame` callback, resizes to `resolution`, auto-centers window. Blocks until key press (`waitKey(0)`).

### TextDetector (`lib/text_detector.py`)

Tesseract OCR wrapper. **Not MediaPipe-based — different dependency chain.**

- Requires: `pytesseract` + Tesseract binary, `imutils`, `pandas`, `scikit-image`, `Pillow`
- Optional: `spacy` + `en_core_web_sm` for NLP methods
- Constructor takes an `image` (BGR numpy array), not a `model_path`
- `preprocess=True` applies grayscale → histogram equalization → adaptive threshold before OCR
- `detect_text()` → `str`; `detect_words()` → `(list[dict], annotated)`; `detect_characters()` → `(list[dict], annotated)`
- Image matching via ORB keypoints: `compare_matches_knn_matcher`, `compare_matches_bf_matcher`; SSIM fallback: `fallback_ssim`
- NLP methods (`extract_entities`, `extract_keywords`, `summarize`, `extract_relations`) return empty results gracefully when spaCy is unavailable
- `HairSegmentation.process` expects **RGB**; `TextDetector` methods expect **BGR** (conversion done internally where needed)

---

## Development Workflow

### First-time setup

```bash
git clone https://github.com/your-org/visionkit.git
cd visionkit
make setup          # runs: uv sync --all-groups + pre-commit install
```

`make setup` is idempotent — safe to re-run.

### Common Make targets

```bash
make format         # black + isort (auto-fix in place)
make format-check   # same check, no writes (used in CI)
make lint           # ruff + flake8 (report only)
make lint-fix       # ruff --fix (auto-fix ruff-fixable issues)
make typecheck      # mypy visionkit/ --ignore-missing-imports
make test           # pytest -m "not integration" (fast, no model files needed)
make test-cov       # pytest with HTML coverage report → htmlcov/
make check          # format-check + lint + typecheck (pre-push sanity)
make hooks-run      # run all pre-commit hooks against all files
make hooks-update   # bump pre-commit hook revs to latest
make clean          # remove __pycache__, .pytest_cache, dist/, htmlcov/
```

### Run the entry point

```bash
uv run python main.py
```

### Run a specific script

```bash
uv run python path/to/script.py
```

### Add a dependency

```bash
uv add <package>            # runtime dep → [project] dependencies
uv add --dev <package>      # dev-only dep → [dependency-groups] dev
```

Never edit `uv.lock` by hand.

---

## Commit Convention

All commits **must** follow [Conventional Commits](https://www.conventionalcommits.org/). The `conventional-pre-commit` hook enforces this at `commit-msg` stage — non-conforming messages are rejected.

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types and their release effect:**

| Type | Semver bump |
|---|---|
| `feat` | minor |
| `fix`, `perf`, `refactor` | patch |
| `feat!` or `BREAKING CHANGE:` footer | major |
| `chore`, `docs`, `test`, `ci`, `build`, `style` | no release |

Examples:

```
feat(hand-detector): add pinch gesture recognition
fix(pose-detector): correct angle calculation for right arm
chore(deps): bump mediapipe to 0.10.36
feat!: remove deprecated detect_v1 API
```

---

## Versioning

Version is managed by **python-semantic-release v9**. Do not edit version strings manually.

- Source of truth: `pyproject.toml → [project] version` and `visionkit/__init__.py → __version__`
- PSR updates both atomically on release
- Version format: `MAJOR.MINOR.PATCH` (e.g. `0.2.1`)
- Tag format: `vMAJOR.MINOR.PATCH` (e.g. `v0.2.1`)
- Release commits use message `chore(release): vX.Y.Z [skip ci]` — the `[skip ci]` guard prevents infinite loop

```toml
# pyproject.toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
version_variables = ["visionkit/__init__.py:__version__"]
branch = "main"
tag_format = "v{version}"
commit_message = "chore(release): v{version} [skip ci]"
build_command = "uv build"
upload_to_pypi = false       # publish.yml handles PyPI via OIDC
upload_to_release = true     # attaches sdist + wheel to GitHub Release
```

To preview the next version without making changes:

```bash
uv run semantic-release version --print
```

---

## Pre-commit Hooks

Hooks run automatically on `git commit`. Configuration is in `.pre-commit-config.yaml`.

| Stage | Hook | What it checks |
|---|---|---|
| `pre-commit` | `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-merge-conflict` | File hygiene |
| `pre-commit` | `black` | Code formatting |
| `pre-commit` | `isort` | Import ordering |
| `pre-commit` | `pyupgrade --py311-plus` | Modern Python syntax |
| `pre-commit` | `ruff` + `ruff-format` | Fast linting + format check |
| `pre-commit` | `flake8` | Additional lint rules (bugbear, comprehensions, simplify) |
| `pre-commit` | `hadolint` | Dockerfile linting |
| `pre-commit` | `actionlint` | GitHub Actions YAML validation |
| `commit-msg` | `conventional-pre-commit` | Conventional commit message format |
| `pre-push` | `pytest -m "not integration"` | Unit test gate before push |

First run after clone will auto-fix most formatting issues — re-stage and commit again.

---

## CI/CD Workflows

All workflows live in `.github/workflows/`.

### `ci-unit.yml` — Unit tests

- **Triggers:** push to any branch; PR to `main`/`master`
- **Matrix:** Python 3.11 + 3.12
- **Command:** `pytest tests/ -m "not integration" --cov=visionkit --cov-report=xml`
- Codecov upload on Python 3.11 only
- Requires `CODECOV_TOKEN` secret (optional)

### `ci-integration.yml` — Integration tests

- **Triggers:** push/PR to `main`/`master`; `workflow_dispatch`
- **Command:** `pytest tests/ -m "integration"`
- Model files retrieved via `actions/cache@v4` (cache key: `models-v1`)
- Installs Tesseract binary via `apt`
- Mark tests with `@pytest.mark.integration` when model files are available

### `ci-security.yml` — Security scan

- **Triggers:** push/PR to `main`/`master`; daily cron `0 2 * * *` (02:00 UTC)
- **Jobs:**
  - `pip-audit` — scans all deps against OSV/PyPI advisory DB; outputs CycloneDX SBOM artifact; fails CI on any unfixed CVE
  - `trivy` — filesystem + SCA scan; uploads SARIF to GitHub Security tab; fails on CRITICAL
  - `codeql` — Python static analysis with `security-and-quality` queries
- Requires `security-events: write` permission

### `renovate.yml` — Dependency updates

- **Triggers:** weekly Monday 01:00 UTC; `workflow_dispatch` with `dry_run` option
- Uses `renovatebot/github-action@v40`
- Configuration in `renovate.json`:
  - `osvVulnerabilityAlerts: true` — immediate security PRs regardless of schedule
  - Auto-merge minor/patch for dev tools
  - Manual review required for `mediapipe` and `opencv-python` major bumps
- Requires `RENOVATE_TOKEN` secret (GitHub PAT, `repo` scope)

### `semantic-release.yml` — Version bump + GitHub Release

- **Triggers:** push to `main`/`master`; skips if commit message contains `[skip ci]`
- **Concurrency:** `cancel-in-progress: false` — never interrupt a release mid-flight
- Steps: full history checkout (`fetch-depth: 0`) → `uv sync` → `semantic-release version --vcs-release`
- PSR writes version to `pyproject.toml` + `visionkit/__init__.py`, commits, tags, creates GitHub Release
- `trigger-publish` job dispatches `publish.yml` at the new tag ref via `workflow_dispatch`
- Requires `SEMANTIC_RELEASE_TOKEN` secret (PAT with `contents: write` + `pull-requests: write`)

### `publish.yml` — PyPI publish

- **Triggers:** `release: published` event; `workflow_dispatch` with `target` choice (`testpypi` | `pypi`)
- **Jobs:** `test` → `build` → `publish-testpypi` (pre-releases) / `publish-pypi` (stable)
- Uses `pypa/gh-action-pypi-publish@release/v1` with OIDC trusted publishing — no API tokens stored
- Requires GitHub environments configured on pypi.org and test.pypi.org:
  - Environment name `pypi` → pypi.org
  - Environment name `testpypi` → test.pypi.org

---

## Tests

Tests live in `tests/` mirroring the package structure (e.g. `tests/lib/test_face_detector.py`).

### Markers

```python
@pytest.mark.unit         # fast, pure-Python, no model files or external services
@pytest.mark.integration  # requires real .tflite/.task model files
@pytest.mark.slow         # takes > 5 s
```

### Running tests

```bash
uv run pytest tests/ -v                        # all tests
uv run pytest tests/ -m "not integration" -v   # unit only (no model files needed)
uv run pytest tests/ -m "integration" -v       # integration only
uv run pytest tests/ --cov=visionkit           # with coverage
```

All unit tests must pass without model files. Mark any test that instantiates a real detector with `@pytest.mark.integration`.

---

## Required GitHub Secrets

| Secret | Used by | Scope |
|---|---|---|
| `SEMANTIC_RELEASE_TOKEN` | `semantic-release.yml` | PAT: `contents: write`, `pull-requests: write` |
| `RENOVATE_TOKEN` | `renovate.yml` | PAT: `repo` |
| `CODECOV_TOKEN` | `ci-unit.yml` | Codecov upload token (optional) |

PyPI publishing uses OIDC trusted publishing — no PyPI API token needed.

---

## Boundaries

**Never:**
- Commit `.tflite` or `.task` model files to git (binary, large, user-provided)
- Commit `.env` files or any file containing credentials
- Pass RGB arrays to detector methods — all `lib/` detectors expect BGR (except `HairSegmentation.process` which expects RGB)
- Share a stateful detector instance across multiple sessions without resetting instance state (especially `PoseDetector` rep counters)
- Mix `running_mode` after construction — MediaPipe does not allow switching modes on a live detector
- Add dependencies without updating `pyproject.toml` via `uv add`
- Modify `uv.lock` by hand — always let `uv` manage it
- Edit `pyproject.toml:project.version` or `visionkit/__init__.py:__version__` by hand — python-semantic-release owns these
- Skip the pre-commit hooks with `git commit --no-verify` except in genuine emergencies
- Push directly to `main`/`master` — the `no-commit-to-branch` hook blocks this; use PRs

---

## New Detector Pattern (Code Example)

Minimal skeleton for a new detector. Every field shown is required — don't omit any.

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MyDetector:
    def __init__(
        self,
        model_path: str = "./models/my_model.tflite",
        running_mode: str = "IMAGE",   # "IMAGE" | "VIDEO"
        min_detection_confidence: float = 0.5,
    ):
        self.running_mode = getattr(vision.RunningMode, running_mode)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.MyDetectorOptions(          # replace with actual Options class
            base_options=base_options,
            running_mode=self.running_mode,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = vision.MyDetector.create_from_options(options)

    def _to_mp_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def detect(self, image, timestamp_ms=None):
        mp_image = self._to_mp_image(image)          # BGR in → RGB mp.Image
        if self.running_mode == vision.RunningMode.IMAGE:
            result = self.detector.detect(mp_image)
        else:
            result = self.detector.detect_for_video(mp_image, timestamp_ms or 0)
        annotated = image.copy()
        # ... draw on annotated ...
        return annotated, result                     # BGR out
```

---

## Adding a New Detector

1. Create `visionkit/lib/my_detector.py`.
2. Follow the existing detector pattern: `__init__` with `model_path` + `running_mode`, `_to_mp_image`, primary detection method returning `(annotated, result)`.
3. Accept BGR input; convert to RGB internally before passing to MediaPipe.
4. Add the corresponding model file name to the README model table.
5. Add unit tests in `tests/lib/test_my_detector.py` — mock the MediaPipe result; mark integration tests with `@pytest.mark.integration`.
6. Document the class and all public methods following the existing docstring style.

---

## Common Pitfalls

- **Wrong image format:** Passing RGB to a detector that expects BGR will produce incorrect colors. All detectors in `lib/` accept BGR — they handle BGR→RGB conversion internally. `HairSegmentation.process` is the exception: it expects RGB.
- **Missing model file:** Detectors raise a file-not-found error at construction time if `model_path` does not exist. Always verify the path before instantiating.
- **VIDEO mode without timestamps:** `PoseDetector` auto-increments its counter, but other detectors require `timestamp_ms` to increase monotonically between calls in VIDEO mode.
- **Calibrated distance without calibration samples:** `HandDetector.estimate_distance_cm` returns `None` if `self.model` is not set. Pass `calibration_samples` to `__init__` to enable it.
- **Workout rep counter is per-instance state:** Do not share a single `PoseDetector` instance across multiple people or sessions without resetting `rep_count` and related state.
- **Version string drift:** `pyproject.toml:project.version` and `visionkit/__init__.py:__version__` must always match. PSR keeps them in sync — never edit them manually or they will diverge on next release.
- **Non-conventional commit message:** `conventional-pre-commit` will reject commits that don't match the `<type>: <desc>` format. Fix the message, don't bypass the hook.
