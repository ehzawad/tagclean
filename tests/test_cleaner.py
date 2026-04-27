from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tagclean.cleaner import (
    CleanerConfig,
    _stable_family_id,
    _symlink_shared_stages,
    artifact_score,
    canonical_tag_name,
    coerce_optional_str_list,
    comparison_key,
    find_close_tag_clusters,
    normalize_question,
    resolve_target_tags,
    run_compose,
    run_stage0,
    run_stage8,
    run_stage9,
)


def _write_question_tag(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "tag"])
        writer.writerows(rows)


# ----------------------------- helpers -----------------------------

def test_coerce_optional_str_list_avoids_numpy_truthiness() -> None:
    assert coerce_optional_str_list(None) == []
    assert coerce_optional_str_list(["a", "b"]) == ["a", "b"]
    assert coerce_optional_str_list(np.array(["x", "y"])) == ["x", "y"]


def test_normalize_question_folds_digits_quotes_and_whitespace() -> None:
    text = "  NID পোর্টালে “লকড”  ১২৩  "
    assert normalize_question(text) == 'nid পোর্টালে "লকড" 123'
    assert comparison_key(text) == "nid পোর্টালে লকড 123"


def test_canonical_tag_name_prefers_base_semantic_tag() -> None:
    counts = {
        "nid_fee_01_followup_a": 100,
        "nid_fee": 10,
        "nid_fee_01": 20,
    }
    assert canonical_tag_name(list(counts), counts) == "nid_fee"


def test_artifact_score_flags_short_and_repetitive_text() -> None:
    assert artifact_score("ok") >= 0.5
    assert artifact_score("aaaaa repeated") > 0.0
    long_clean = "এনআইডি অ্যাকাউন্ট লক হলে কীভাবে আনলক করতে হয়?"
    assert artifact_score(long_clean) < 0.25


# ----------------------------- stage 0 -----------------------------

def test_stage0_jettisons_duplicates_context_and_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "question_tag.csv"
    _write_question_tag(
        csv_path,
        [
            ("NID কার্ড হারিয়েছে কী করব?", "lost_card"),
            ("NID কার্ড হারিয়েছে কী করব?", "lost_card"),
            ("হ্যাঁ", "lost_card"),
            ("What format? Might go beyond. But answer only says full address required.", "process_requirements"),
            ("ভোটার এলাকা ট্রান্সফারের ফর্ম নং কত?", "transfer_form"),
            ("ভোটার এলাকা ট্রান্সফারের ফর্ম নং কত?", "transfer_form_followup_a"),
        ],
    )
    tag_answer = tmp_path / "tag_answer.json"
    tag_answer.write_text("{}", encoding="utf-8")
    cfg = CleanerConfig(
        input_csv=csv_path,
        tag_answer_json=tag_answer,
        artifact_root=tmp_path / "artifacts",
        run_id="test_run",
        embedding_backend="hashing",
    )

    run_stage0(cfg, resume=False)

    df = pd.read_parquet(cfg.run_dir() / "stage0" / "intake.parquet")
    assert (df["status"] == "keep").sum() == 3
    assert (df["pre_reason"] == "duplicate").sum() == 1
    assert (df["pre_reason"] == "context_dependent").sum() == 1
    assert (df["pre_reason"] == "synthetic_artifact").sum() == 1

    conflicts = pd.read_json(cfg.run_dir() / "stage0" / "cross_tag_duplicates.jsonl", lines=True)
    assert len(conflicts) == 1
    assert set(conflicts.iloc[0]["tags"]) == {"transfer_form", "transfer_form_followup_a"}


# --------------------------- target scope --------------------------

def test_resolve_target_tags_keeps_corpus_scope_separate() -> None:
    rows = pd.DataFrame(
        {
            "tag": ["a", "b", "c", "d"],
            "canonical_tag": ["a", "b_canon", "c", "d"],
        }
    )
    cfg = CleanerConfig(target_max_tags=2)
    tag_to_canon = {"b": "b_canon"}

    assert resolve_target_tags(rows, cfg, tag_to_canon) == {"a", "b_canon"}


# ----------------------- E5-only cluster discovery -----------------

def test_find_close_tag_clusters_groups_above_threshold_e5_only() -> None:
    """E5-only signature after the Gemma drop. The optional edge_predicate
    is the kNN-overlap multi-gate; tested separately."""
    tags = ["a", "b", "c", "d"]
    sim_e5 = np.array(
        [
            [1.0, 0.92, 0.10, 0.10],
            [0.92, 1.0, 0.10, 0.10],
            [0.10, 0.10, 1.0, 0.10],
            [0.10, 0.10, 0.10, 1.0],
        ],
        dtype=np.float32,
    )

    clusters = find_close_tag_clusters(tags, sim_e5, threshold=0.85, max_cluster_size=10)
    assert clusters == [["a", "b"]]

    none = find_close_tag_clusters(tags, sim_e5, threshold=0.99, max_cluster_size=10)
    assert none == []


def test_find_close_tag_clusters_edge_predicate_can_veto_pair() -> None:
    """Cosine alone passes; the kNN-overlap predicate vetoes the edge."""
    tags = ["a", "b"]
    sim_e5 = np.array([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32)

    veto = find_close_tag_clusters(
        tags, sim_e5, threshold=0.85, max_cluster_size=10,
        edge_predicate=lambda i, j: False,
    )
    assert veto == []

    ok = find_close_tag_clusters(
        tags, sim_e5, threshold=0.85, max_cluster_size=10,
        edge_predicate=lambda i, j: True,
    )
    assert ok == [["a", "b"]]


# --------------------------- Stage 9 -------------------------------

def test_stage9_audits_emit_expected_artifacts(tmp_path: Path) -> None:
    cleaned_csv = tmp_path / "cleaned.csv"
    rows = [
        ("the cat sat on the mat", "feline"),
        ("a cat purred softly", "feline"),
        ("kittens love yarn", "feline"),
        ("dogs chase squirrels", "canine"),
        ("the puppy barked loudly", "canine"),
        ("a wolf howled at the moon", "canine"),
        ("apples are sweet fruit", "fruit"),
        ("oranges grow on trees", "fruit"),
        ("ripe banana tastes good", "fruit"),
    ]
    with cleaned_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "tag"])
        w.writerows(rows)

    tag_answer = tmp_path / "tag_answer.json"
    tag_answer.write_text("{}", encoding="utf-8")
    cfg = CleanerConfig(
        input_csv=cleaned_csv,
        tag_answer_json=tag_answer,
        artifact_root=tmp_path / "artifacts",
        run_id="stage9_test",
        embedding_backend="hashing",
        e5_use_prefixes=False,
        e5_audit_top_k=3,
        stage9_input_csv=str(cleaned_csv),
    )

    run_stage9(cfg, resume=False)

    stage_dir = cfg.run_dir() / "stage9"
    audit = pd.read_csv(stage_dir / "e5_neighbor_audit.csv")
    kept = pd.read_csv(stage_dir / "production_filtered.csv")
    dropped = pd.read_csv(stage_dir / "e5_dropped.csv")
    report = json.loads((stage_dir / "audit_report.json").read_text())

    assert len(audit) == len(rows)
    assert set(audit.columns) >= {
        "question",
        "tag",
        "top1_neighbor_tag",
        "top1_correct",
        "own_share_top_k",
        "neighbor_tag_dist",
    }
    assert audit["audit_top_k"].iloc[0] == 3
    assert len(kept) + len(dropped) == len(audit)
    assert (audit["top1_correct"]).sum() == len(kept)
    assert ((~audit["top1_correct"]).sum()) == len(dropped)
    assert report["rows_in"] == len(rows)
    assert report["rows_kept"] + report["rows_dropped"] == report["rows_in"]
    assert 0.0 <= report["top1_accuracy_in"] <= 1.0


def test_stage9_no_drop_mode_keeps_all_rows(tmp_path: Path) -> None:
    cleaned_csv = tmp_path / "cleaned.csv"
    rows = [
        ("alpha apple", "a"),
        ("beta banana", "b"),
        ("alpha apricot", "a"),
        ("beta blueberry", "b"),
    ]
    with cleaned_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "tag"])
        w.writerows(rows)

    tag_answer = tmp_path / "tag_answer.json"
    tag_answer.write_text("{}", encoding="utf-8")
    cfg = CleanerConfig(
        input_csv=cleaned_csv,
        tag_answer_json=tag_answer,
        artifact_root=tmp_path / "artifacts",
        run_id="stage9_no_drop",
        embedding_backend="hashing",
        e5_use_prefixes=False,
        e5_audit_top_k=2,
        stage9_input_csv=str(cleaned_csv),
        e5_audit_drop_on_top1_mismatch=False,
    )
    run_stage9(cfg, resume=False)
    stage_dir = cfg.run_dir() / "stage9"
    kept = pd.read_csv(stage_dir / "production_filtered.csv")
    dropped = pd.read_csv(stage_dir / "e5_dropped.csv")
    assert len(kept) == len(rows)
    assert len(dropped) == 0


# ------------------------- compose ---------------------------------

def test_compose_concats_and_dedupes_top40s(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    a_dir = artifact_root / "family_a" / "stage6"
    a_dir.mkdir(parents=True)
    (a_dir / "question_tag.top40.csv").write_text("question,tag\nq1,a\nq2,b\n", encoding="utf-8")
    b_dir = artifact_root / "family_b" / "stage6"
    b_dir.mkdir(parents=True)
    (b_dir / "question_tag.top40.csv").write_text("question,tag\nq1,a\nq3,c\n", encoding="utf-8")
    cfg = CleanerConfig(artifact_root=artifact_root)
    out = artifact_root / "production" / "composed.csv"

    run_compose(cfg, ["family_a", "family_b"], "top40", out)

    df = pd.read_csv(out)
    assert list(df.columns) == ["question", "tag"]
    assert len(df) == 3
    assert sorted(zip(df.question, df.tag)) == [("q1", "a"), ("q2", "b"), ("q3", "c")]
    assert list(df.tag) == ["a", "b", "c"]


def test_compose_errors_on_missing_source(tmp_path: Path) -> None:
    cfg = CleanerConfig(artifact_root=tmp_path / "artifacts")
    out = tmp_path / "out.csv"
    try:
        run_compose(cfg, ["nope"], "top40", out)
    except FileNotFoundError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("compose should raise when a source run is missing")


# --------------------- discover / scaling --------------------------

def test_stable_family_id_is_order_invariant() -> None:
    a = _stable_family_id(["alpha", "beta", "gamma"])
    b = _stable_family_id(["gamma", "alpha", "beta"])
    c = _stable_family_id(["alpha", "beta", "gamma"])
    assert a == b == c
    assert a.startswith("fam_")
    assert _stable_family_id(["alpha", "beta", "delta"]) != a


def test_symlink_shared_stages_refuses_geometry_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    for run_id in ("centroids_a", "centroids_b"):
        for stage in ("stage0", "stage1", "stage2"):
            (artifact_root / run_id / stage).mkdir(parents=True)

    family_id = "fam_test"
    _symlink_shared_stages(artifact_root, family_id, "centroids_a")
    try:
        _symlink_shared_stages(artifact_root, family_id, "centroids_b")
    except RuntimeError as e:
        assert "centroids_b" in str(e)
    else:
        raise AssertionError(
            "_symlink_shared_stages should refuse to overwrite a symlink that "
            "points at a different centroids run."
        )


# ------------- deterministic-only end-to-end regression ------------

def test_deterministic_pipeline_stage8_then_stage9(tmp_path: Path) -> None:
    """Smoke test for the deterministic-only chain.

    Stage 8 must reach Stage 6 directly (no Stage QA in the chain) and
    Stage 9 must read its input from stage6/cleaned.csv. Tiny synthetic
    fixture: 3 well-separated tags x ~10 rows. Only validates wiring +
    output shape, not embedding quality.
    """
    csv_path = tmp_path / "question_tag.csv"
    _write_question_tag(
        csv_path,
        [
            ("the cat sat on the mat", "feline"),
            ("a cat purred softly", "feline"),
            ("kittens love yarn", "feline"),
            ("a small kitten meowed", "feline"),
            ("the tabby cat napped", "feline"),
            ("a cat licked its paw", "feline"),
            ("dogs chase squirrels", "canine"),
            ("the puppy barked loudly", "canine"),
            ("a wolf howled at the moon", "canine"),
            ("the dog wagged its tail", "canine"),
            ("a beagle ran in the yard", "canine"),
            ("the husky pulled the sled", "canine"),
            ("apples are sweet fruit", "fruit"),
            ("oranges grow on trees", "fruit"),
            ("ripe banana tastes good", "fruit"),
            ("a pear is juicy", "fruit"),
            ("grapes hang from vines", "fruit"),
            ("mango is a tropical fruit", "fruit"),
        ],
    )
    tag_answer = tmp_path / "tag_answer.json"
    tag_answer.write_text("{}", encoding="utf-8")

    cfg = CleanerConfig(
        input_csv=csv_path,
        tag_answer_json=tag_answer,
        artifact_root=tmp_path / "artifacts",
        run_id="det_only_smoke",
        embedding_backend="hashing",
        e5_use_prefixes=False,
        language="none",
        top_n=4,
    )

    run_stage8(cfg, resume=False)

    run_dir = cfg.run_dir()
    cleaned = pd.read_csv(run_dir / "stage6" / "question_tag.cleaned.csv")
    top40 = pd.read_csv(run_dir / "stage6" / "question_tag.top40.csv")
    assert len(cleaned) > 0
    assert len(top40) > 0
    forbidden = {"audit_status", "reason_code", "rationale"}
    assert forbidden.isdisjoint(set(cleaned.columns))
    # Stage QA dir must NOT exist; Stage 8 chains through Stage 6 directly.
    assert not (run_dir / "stage5").exists()

    # Stage 8's LOO report
    cleaning_report = json.loads((run_dir / "stage8" / "cleaning_report.json").read_text())
    assert "cleaned" in cleaning_report and "top40" in cleaning_report

    # Stage 9 reads stage6/cleaned.csv by default (no stage5/ preference).
    run_stage9(cfg, resume=False)
    stage9_dir = run_dir / "stage9"
    assert (stage9_dir / "audit_report.json").exists()
    assert (stage9_dir / "production_filtered.csv").exists()


def test_compose_does_not_look_at_stage5(tmp_path: Path) -> None:
    """Regression: compose must read stage6/ only. If a stage5/ dir
    exists from an earlier branch, it must be ignored."""
    artifact_root = tmp_path / "artifacts"
    fam_dir = artifact_root / "fam_x"
    (fam_dir / "stage5").mkdir(parents=True)
    (fam_dir / "stage5" / "question_tag.top40.csv").write_text(
        "question,tag\nfrom_stage5,wrong\n", encoding="utf-8"
    )
    (fam_dir / "stage6").mkdir(parents=True)
    (fam_dir / "stage6" / "question_tag.top40.csv").write_text(
        "question,tag\nfrom_stage6,right\n", encoding="utf-8"
    )
    cfg = CleanerConfig(artifact_root=artifact_root)
    out = tmp_path / "composed.csv"
    run_compose(cfg, ["fam_x"], "top40", out)
    df = pd.read_csv(out)
    assert list(df.tag) == ["right"]
    assert "from_stage5" not in set(df.question)
