from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tagclean.cleaner import (
    CleanerConfig,
    JudgePacketResult,
    JudgeResult,
    RowJudgeDecision,
    artifact_score,
    build_audit_packets,
    build_judge_packets,
    build_judge_prompt,
    canonical_tag_name,
    coerce_optional_str_list,
    comparison_key,
    find_close_tag_clusters,
    missing_row_ids_in_packet_result,
    normalize_question,
    resolve_target_tags,
    run_stage0,
    token_alignment_score,
    _resolve_consistency,
)


def _write_question_tag(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "tag"])
        writer.writerows(rows)


def test_coerce_optional_str_list_avoids_numpy_truthiness() -> None:
    assert coerce_optional_str_list(None) == []
    assert coerce_optional_str_list(["a", "b"]) == ["a", "b"]
    assert coerce_optional_str_list(np.array(["x", "y"])) == ["x", "y"]


def test_build_judge_prompt_serializes_with_ndarray_profile_cells() -> None:
    cfg = CleanerConfig()
    profiles = {
        "primary": {
            "description": "Primary tag",
            "central_questions": np.array(["central one", "central two"]),
            "discriminative_phrases": np.array(["unique phrase"]),
        },
        "competing": {
            "description": "Competing tag",
            "central_questions": np.array(["other one"]),
            "discriminative_phrases": np.array(["other phrase"]),
        },
    }
    row = pd.Series(
        {
            "canonical_tag": "primary",
            "question_raw": "raw question",
            "question_norm": "norm question",
            "e5_top1_competing_tag": "competing",
            "gemma_top1_competing_tag": "competing",
            "e5_top10_evidence": json.dumps([{"rank": 1, "tag": "competing", "row_id": 9}]),
            "gemma_top10_evidence": json.dumps([{"rank": 1, "tag": "competing", "row_id": 9}]),
            "embedding_reconciliation": json.dumps({}),
        }
    )

    prompt = build_judge_prompt(row, profiles, cfg)
    payload = json.loads(prompt)
    assert payload["current_tag"] == "primary"
    assert payload["current_tag_central_examples"] == ["central one", "central two"]
    assert payload["current_tag_discriminative_phrases"] == ["unique phrase"]
    assert any(c["tag"] == "competing" for c in payload["competing_tags"])


def test_normalize_question_folds_digits_quotes_and_whitespace() -> None:
    text = "  NID পোর্টালে “লকড”  ১২৩  "
    assert normalize_question(text) == 'nid পোর্টালে "লকড" 123'
    assert comparison_key(text) == "nid পোর্টালে লকড 123"


def test_stage0_jettisons_duplicates_context_and_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "question_tag.csv"
    _write_question_tag(
        csv_path,
        [
            ("NID কার্ড হারিয়েছে কী করব?", "lost_card"),
            ("NID কার্ড হারিয়েছে কী করব?", "lost_card"),
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


def test_canonical_tag_name_prefers_base_semantic_tag() -> None:
    counts = {
        "nid_fee_01_followup_a": 100,
        "nid_fee": 10,
        "nid_fee_01": 20,
    }
    assert canonical_tag_name(list(counts), counts) == "nid_fee"


def test_missing_row_ids_in_packet_result_finds_omitted_rows() -> None:
    packet = pd.DataFrame({"row_id": [10, 11, 12, 13]})
    decision = RowJudgeDecision(
        row_id=11,
        decision="keep",
        quality_score=80,
        ambiguity_score=20,
        context_dependent=False,
        reason_code="clean",
        rationale="ok",
    )
    result = JudgePacketResult(
        packet_id="packet:000000",
        decisions=[decision],
        packet_rationale="partial",
    )

    assert missing_row_ids_in_packet_result(packet, result) == [10, 12, 13]


def test_inconsistent_judge_passes_jettison_row() -> None:
    keep = JudgeResult(
        decision="keep",
        quality_score=90,
        ambiguity_score=5,
        context_dependent=False,
        reason_code="clean",
        rationale="clean",
    )
    drop = JudgeResult(
        decision="jettison",
        quality_score=0,
        ambiguity_score=100,
        context_dependent=False,
        reason_code="sibling_collision",
        rationale="ambiguous",
    )

    resolved = _resolve_consistency(123, [keep, drop])

    assert resolved["decision"] == "jettison"
    assert resolved["consistent"] is False
    assert resolved["reason_code"] == "sibling_collision"


def test_build_judge_packets_groups_three_tags_with_bounded_rows() -> None:
    rows = []
    for tag in ["a", "b", "c", "d"]:
        for idx in range(5):
            rows.append(
                {
                    "row_id": len(rows) + 1,
                    "route": "judge",
                    "canonical_tag": tag,
                    "e5_margin": idx / 10,
                    "gemma_margin": idx / 10,
                }
            )
    cfg = CleanerConfig(tags_per_judge_call=3, rows_per_tag_per_judge_call=2)

    packets = build_judge_packets(pd.DataFrame(rows), cfg)

    assert packets
    assert all(packet["canonical_tag"].nunique() <= 3 for packet in packets)
    for packet in packets:
        counts = packet.groupby("canonical_tag").size()
        assert int(counts.max()) <= 2


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


def test_find_close_tag_clusters_groups_above_threshold() -> None:
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
    sim_gemma = sim_e5.copy()

    clusters = find_close_tag_clusters(tags, sim_e5, sim_gemma, threshold=0.85, max_cluster_size=10)
    assert clusters == [["a", "b"]]

    none = find_close_tag_clusters(tags, sim_e5, sim_gemma, threshold=0.99, max_cluster_size=10)
    assert none == []


def test_build_audit_packets_groups_by_tag_and_caps_size() -> None:
    rows = []
    for tag in ["x", "y"]:
        for idx in range(30):
            rows.append(
                {
                    "row_id": len(rows) + 1,
                    "canonical_tag": tag,
                    "audit_buffer": True,
                    "composite_score": (30 - idx) / 30.0,
                    "question_raw": f"Q{idx}",
                    "question_norm": f"q{idx}",
                }
            )
    rows.append(
        {
            "row_id": 999,
            "canonical_tag": "x",
            "audit_buffer": False,
            "composite_score": 0.0,
            "question_raw": "below buffer",
            "question_norm": "below buffer",
        }
    )
    cfg = CleanerConfig(audit_rows_per_packet=8)

    packets = build_audit_packets(pd.DataFrame(rows), cfg)

    assert all(packet["canonical_tag"].nunique() == 1 for packet in packets)
    assert all(len(packet) <= cfg.audit_rows_per_packet for packet in packets)
    assert sum(len(p) for p in packets) == 60


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
