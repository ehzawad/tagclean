"""End-to-end smoke test on the synthetic English fixture using heuristic mode (no GPT)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = REPO_ROOT / "examples" / "smoke" / "config.yaml"
SMOKE_ARTIFACTS = REPO_ROOT / "examples" / "smoke" / "artifacts"


@pytest.fixture(autouse=True)
def _clean_artifacts():
    if SMOKE_ARTIFACTS.exists():
        shutil.rmtree(SMOKE_ARTIFACTS)
    yield
    # Leave artifacts on disk after the test so a developer can poke around.


def test_smoke_pipeline_runs_in_heuristic_mode():
    cmd = [sys.executable, "-m", "tagclean.cleaner", "stage6", "--config", str(FIXTURE_CONFIG), "--no-resume"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    run_dir = SMOKE_ARTIFACTS / "smoke"
    cleaned = run_dir / "stage6" / "question_tag.cleaned.csv"
    top40 = run_dir / "stage6" / "question_tag.top40.csv"
    manifest = run_dir / "run_manifest.json"
    assert cleaned.exists()
    assert top40.exists()
    assert manifest.exists()

    cleaned_df = pd.read_csv(cleaned)
    top40_df = pd.read_csv(top40)
    assert set(cleaned_df["tag_clean"].unique()) == {
        "password_reset_self",
        "password_reset_failed",
        "password_reset_help_request",
    }
    # Every kept row must come from one of the three input tags.
    assert set(top40_df["tag"].unique()).issubset(set(cleaned_df["tag_clean"].unique()))
    # Heuristic mode should still keep at least a few per tag in this small fixture.
    assert (cleaned_df.groupby("tag_clean").size() >= 1).all()


def test_seed_tag_resolves_close_cluster():
    """--seed-tag X should pull X's two siblings into target_tags automatically."""
    cmd = [
        sys.executable,
        "-m",
        "tagclean.cleaner",
        "stage6",
        "--config",
        str(FIXTURE_CONFIG),
        "--seed-tag",
        "password_reset_self",
        "--no-resume",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    # The CLI should print the resolved cluster.
    assert "[seed] resolved cluster from 'password_reset_self'" in result.stdout
    for tag in ("password_reset_self", "password_reset_failed", "password_reset_help_request"):
        assert tag in result.stdout

    import json
    manifest = json.loads((SMOKE_ARTIFACTS / "smoke" / "run_manifest.json").read_text())
    assert manifest["seed_tag"] == "password_reset_self"
