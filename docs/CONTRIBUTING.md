# Contributing

## Prerequisites

- Python 3.11 or 3.12 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone https://github.com/Ferapagar/national-defense-hackathon
cd national-defense-hackathon
uv sync          # installs runtime + dev dependencies into .venv
```

## Available commands

<!-- AUTO-GENERATED from pyproject.toml -->
| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (runtime + dev) |
| `pytest object-detection/tests/` | Run the test suite |
| `pytest object-detection/tests/ --cov=object-detection --cov-report=term-missing` | Run tests with coverage report |
| `ruff check .` | Lint all Python files |
| `ruff format .` | Auto-format all Python files |
| `python object-detection/pipeline/run_pipeline.py --help` | Full detection pipeline CLI |
| `python object-detection/calibration/estimate_relative_pose.py --help` | Camera pose estimation CLI |
| `python object-detection/calibration/multi_view.py --help` | N-view consistency check CLI |
| `python object-detection/pipeline/build_world_frame.py --help` | World-frame construction CLI |
<!-- END AUTO-GENERATED -->

## Testing

We use **pytest**. Write tests before implementation (TDD):

1. Write a failing test (RED)
2. Write minimal code to pass it (GREEN)
3. Refactor (IMPROVE)
4. Verify coverage stays ≥ 80%

Tests are in `object-detection/tests/`. They are fully synthetic — no video files or calibration data required. Run them from the repo root:

```bash
pytest object-detection/tests/ -v
```

### Writing new tests

Follow the AAA pattern and use descriptive names:

```python
def test_motion_extraction_picks_up_moving_pixels(tmp_path):
    # Arrange
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video, n_frames=10)

    # Act
    frames = list(extract_motion(video, camera_id=0, threshold=20))

    # Assert
    assert len(frames) >= 5
    assert all(f.motion_total > 0 for f in frames)
```

Use `pytest.mark.unit` / `pytest.mark.integration` to categorise.

## Code style

- **Formatter**: `ruff format` (line length 110)
- **Linter**: `ruff check` (target Python 3.12)
- **Types**: type-annotate all function signatures
- **Immutability**: prefer `@dataclass(frozen=True)` and `NamedTuple`; never mutate in-place
- **Comments**: only when the _why_ is non-obvious

## PR checklist

- [ ] Tests added / updated for all changed behaviour
- [ ] `pytest` passes with no failures
- [ ] `ruff check .` returns no errors
- [ ] Functions < 50 lines, files < 800 lines
- [ ] No hardcoded paths or magic numbers
- [ ] No secrets or credentials committed
