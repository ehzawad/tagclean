from __future__ import annotations

import asyncio
import csv
import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tagclean import claude_cli
from tagclean.cleaner import (
    CleanerConfig,
    FlaggedRow,
    STAGE_QA_PROMPT_VERSION,
    StageQAResult,
    _schema_fingerprint,
    _stable_family_id,
    _symlink_shared_stages,
    aggregate_qa_results,
    artifact_score,
    build_stage_qa_packets,
    build_stage_qa_prompt,
    canonical_tag_name,
    coerce_optional_str_list,
    comparison_key,
    find_close_tag_clusters,
    normalize_question,
    resolve_target_tags,
    run_compose,
    run_stage0,
    run_stage9,
    token_alignment_score,
    write_jsonl,
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


def test_token_alignment_scores_must_have_and_must_avoid() -> None:
    text_with = "তথ্য ঠিক দিলেও সমাধান হচ্ছে না। কখন আবার চেষ্টা করব?"
    text_without = "অ্যাকাউন্ট লক, কীভাবে আনলক করব?"
    must_have = ["তথ্য ঠিক", "আবার"]
    must_avoid = ["আনলক করব"]

    assert token_alignment_score(text_with, must_have, must_avoid) == 1.0
    assert token_alignment_score(text_without, must_have, must_avoid) == -1.0
    assert token_alignment_score("", [], []) == 0.0


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
        judge_mode="heuristic",
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


# --------------------------- packet shaping ------------------------

def test_build_stage_qa_packets_groups_by_tag_and_caps_size() -> None:
    """Stage QA packets are per-tag, packet-size-capped, sorted by
    composite_score desc within each tag (strongest rows first)."""
    rows = []
    for tag in ["x", "y"]:
        for idx in range(30):
            rows.append({
                "row_id": len(rows) + 1,
                "canonical_tag": tag,
                "composite_score": (30 - idx) / 30.0,
                "question_raw": f"Q{idx}",
                "question_norm": f"q{idx}",
            })
    packets = build_stage_qa_packets(pd.DataFrame(rows), packet_size=8)

    assert all(packet["canonical_tag"].nunique() == 1 for packet in packets)
    assert all(len(packet) <= 8 for packet in packets)
    assert sum(len(p) for p in packets) == 60
    # First row in any packet has the highest composite_score within that packet
    for p in packets:
        assert p["composite_score"].iloc[0] == p["composite_score"].max()


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
    """Cosine alone passes; the kNN-overlap predicate vetoes the edge.
    Replaces the prior dual-cosine min(E5, Gemma) gate."""
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
        judge_mode="heuristic",
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
        judge_mode="heuristic",
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


# --------------------- Stage QA -----------------------------------

def test_stage_qa_prompt_does_not_leak_tag_name() -> None:
    """Anonymization invariant: the tag name must never appear in the
    Stage QA prompt sent to Claude. Codex's primary recommendation."""
    packet = pd.DataFrame([
        {"row_id": 1, "question_raw": "raw1", "question_norm": "norm1"},
        {"row_id": 2, "question_raw": "raw2", "question_norm": "norm2"},
    ])
    prompt = build_stage_qa_prompt(packet)
    payload = json.loads(prompt)
    assert "tag" not in payload
    assert "tag_description" not in payload
    assert "tag_one_line_intent" not in payload
    assert "rival_context" not in payload  # rival exemplars dropped
    # The schema in the contract should also be name-free.
    assert "tag" not in payload["required_json_schema"]["properties"]


def test_stage_qa_prompt_includes_majority_theme_framing() -> None:
    """Codex addition: prompt must tell Claude to infer the majority
    theme of the packet and judge each row against THAT, not against
    the weirdest row or the (absent) tag label."""
    packet = pd.DataFrame([{"row_id": 1, "question_raw": "x", "question_norm": "x"}])
    payload = json.loads(build_stage_qa_prompt(packet))
    assert "MAJORITY THEME" in payload["reference_frame"]


def test_stage_qa_schema_is_strict_required_nullable() -> None:
    """Anthropic strict --json-schema mode requires every property to
    appear in `required`. related_row_id is nullable in value but the
    field must be required at the schema level."""
    schema = StageQAResult.model_json_schema()
    flagged_def = schema["$defs"]["FlaggedRow"]
    assert set(flagged_def["required"]) == {"row_id", "why", "related_row_id"}
    # related_row_id type allows null
    rrid = flagged_def["properties"]["related_row_id"]
    assert rrid.get("anyOf") or rrid.get("type") == ["integer", "null"]


def test_aggregate_qa_results_marks_incomplete_packets(tmp_path: Path) -> None:
    """Coverage failure is packet-level (Codex): if a packet returned
    audit_status=audit_incomplete, its rows are kept by default but
    surfaced in the summary so partial coverage is visible."""
    stage_dir = tmp_path / "stage5"
    (stage_dir / "qa_packets").mkdir(parents=True)
    pkt = {
        "packet_id": "qa_packet:000000",
        "row_ids": [100, 101],
        "audit_status": "audit_incomplete",
        "result": StageQAResult(
            audited_row_ids=[100],  # 101 missing
            flagged_rows=[],
        ).model_dump(),
    }
    (stage_dir / "qa_packets" / "packet_000000.json").write_text(
        json.dumps(pkt), encoding="utf-8",
    )
    per_row, summary = aggregate_qa_results(stage_dir)
    assert summary["incomplete_packets"] == 1
    assert per_row[100]["flagged"] is False
    assert per_row[100]["audit_status"] == "audit_incomplete"


def test_aggregate_qa_results_records_flags_and_telemetry(tmp_path: Path) -> None:
    """Successful packet: flagged rows get a why + related_row_id; the
    rest are audit_status='audited'; summary captures empty-flagged rate."""
    stage_dir = tmp_path / "stage5"
    (stage_dir / "qa_packets").mkdir(parents=True)
    pkt = {
        "packet_id": "qa_packet:000000",
        "row_ids": [10, 11, 12],
        "audit_status": "audited",
        "result": StageQAResult(
            audited_row_ids=[10, 11, 12],
            flagged_rows=[
                FlaggedRow(row_id=11, why="off-intent", related_row_id=None),
                FlaggedRow(row_id=12, why="dup of row 10", related_row_id=10),
            ],
        ).model_dump(),
    }
    (stage_dir / "qa_packets" / "packet_000000.json").write_text(
        json.dumps(pkt), encoding="utf-8",
    )
    per_row, summary = aggregate_qa_results(stage_dir)
    assert summary["flagged_rows"] == 2
    assert summary["empty_flagged_packets"] == 0
    assert per_row[10]["flagged"] is False
    assert per_row[11]["flagged"] is True
    assert per_row[11]["why"] == "off-intent"
    assert per_row[12]["related_row_id"] == 10


def test_schema_fingerprint_changes_when_schema_changes() -> None:
    """Resume hashes embed the schema fingerprint so a Pydantic model
    rev forces a re-run rather than reusing stale artifacts."""
    fp_a = _schema_fingerprint(StageQAResult.model_json_schema())
    fp_b = _schema_fingerprint(FlaggedRow.model_json_schema())
    assert fp_a != fp_b
    assert fp_a == _schema_fingerprint(StageQAResult.model_json_schema())


def test_stage_qa_prompt_version_set() -> None:
    assert STAGE_QA_PROMPT_VERSION
    assert "qa" in STAGE_QA_PROMPT_VERSION.lower()


# --------------------- claude_cli wrapper --------------------------

def _make_stub_claude(dest: Path, body: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"#!/usr/bin/env python3\n{body}\n")
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return dest


@pytest.fixture(autouse=True)
def _reset_claude_cache():
    claude_cli._reset_cache_for_tests()
    yield
    claude_cli._reset_cache_for_tests()


def test_build_subprocess_env_strips_anthropic_routing_vars(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-foo")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "Bearer-foo")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")
    monkeypatch.setenv("KEEP_ME", "ok")
    env = claude_cli.build_subprocess_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_BASE_URL" not in env
    assert env.get("KEEP_ME") == "ok"


def test_resolve_claude_binary_raises_when_absent(tmp_path, monkeypatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(claude_cli.ClaudeNotFound):
        claude_cli.resolve_claude_binary()


def test_resolve_claude_binary_finds_via_path(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    stub = _make_stub_claude(bin_dir / "claude", "print('{\"subtype\": \"success\"}')")
    monkeypatch.setenv("PATH", str(bin_dir))
    assert claude_cli.resolve_claude_binary() == stub.resolve()


def test_spawn_claude_success_returns_structured_output(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    _make_stub_claude(
        bin_dir / "claude",
        "import sys, json; sys.stdin.read(); "
        "print(json.dumps({"
        "'subtype': 'success', 'is_error': False, "
        "'structured_output': {'ok': True}, "
        "'total_cost_usd': 0.02, "
        "'usage': {'cache_creation_input_tokens': 100, 'cache_read_input_tokens': 50},"
        "'num_turns': 1, 'duration_ms': 1234, 'session_id': 'sess-x'"
        "}))",
    )
    monkeypatch.setenv("PATH", str(bin_dir))
    res = asyncio.run(claude_cli.spawn_claude(
        prompt="anything", model="opus", fallback_model="sonnet",
    ))
    assert res.ok
    assert res.structured_output == {"ok": True}
    assert res.total_cost_usd == pytest.approx(0.02)
    assert res.cache_creation_input_tokens == 100
    assert res.cache_read_input_tokens == 50
    assert res.subtype == "success"


def test_spawn_claude_classifies_non_success_subtype_as_transient(tmp_path, monkeypatch) -> None:
    """Codex correctness ask: don't trust is_error alone — check subtype.
    Claude returns valid JSON envelopes for many failures with subtype set."""
    bin_dir = tmp_path / "bin"
    _make_stub_claude(
        bin_dir / "claude",
        "import sys, json; sys.stdin.read(); "
        "print(json.dumps({"
        "'subtype': 'error_max_structured_output_retries', 'is_error': False, "
        "'result': 'gave up after 5 retries', 'total_cost_usd': 0.04"
        "}))",
    )
    monkeypatch.setenv("PATH", str(bin_dir))
    res = asyncio.run(claude_cli.spawn_claude(
        prompt="x", model="opus", fallback_model="sonnet",
    ))
    assert not res.ok
    assert res.error_kind == claude_cli.ErrorKind.TRANSIENT
    assert "error_max_structured_output_retries" in res.error_message
    # cost still recorded for telemetry
    assert res.total_cost_usd == pytest.approx(0.04)


def test_spawn_claude_classifies_auth_failure_as_infra(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    _make_stub_claude(
        bin_dir / "claude",
        "import sys; sys.stdin.read(); "
        "sys.stderr.write('Not logged in. Please run `claude /login`.'); "
        "sys.exit(1)",
    )
    monkeypatch.setenv("PATH", str(bin_dir))
    res = asyncio.run(claude_cli.spawn_claude(
        prompt="x", model="opus", fallback_model="sonnet",
    ))
    assert not res.ok
    assert res.error_kind == claude_cli.ErrorKind.INFRA


def test_spawn_claude_parse_failure_classified_as_parse(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    _make_stub_claude(
        bin_dir / "claude",
        "import sys; sys.stdin.read(); print('not json at all')",
    )
    monkeypatch.setenv("PATH", str(bin_dir))
    res = asyncio.run(claude_cli.spawn_claude(
        prompt="x", model="opus", fallback_model="sonnet",
    ))
    assert res.error_kind == claude_cli.ErrorKind.PARSE


def test_spawn_claude_timeout_classified_as_transient(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    _make_stub_claude(
        bin_dir / "claude",
        "import time; time.sleep(60)",
    )
    monkeypatch.setenv("PATH", str(bin_dir))
    res = asyncio.run(claude_cli.spawn_claude(
        prompt="x", model="opus", fallback_model="sonnet", timeout=0.5,
    ))
    assert res.error_kind == claude_cli.ErrorKind.TRANSIENT
    assert "timed out" in res.error_message.lower()
