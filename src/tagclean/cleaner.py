#!/usr/bin/env python3
"""
Fully automated dataset-cleaning harness for full_dataset/question_tag.csv.

The harness is intentionally stage-based and resumable. Heavy dependencies
such as FAISS, sentence-transformers, and OpenAI are imported lazily so the
deterministic stages and tests can run without loading models.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Defaults are relative to the working directory; CLI/config resolves them.
PROJECT_ROOT = Path.cwd()
DEFAULT_INPUT_CSV = Path("data/question_tag.csv")
DEFAULT_TAG_ANSWER_JSON = Path("data/tag_answer.json")
DEFAULT_ARTIFACT_ROOT = Path("artifacts")

# Module-level toggle for language-specific normalization.
# Set by main() to config.language; read by normalize_question.
_NORMALIZATION_LANGUAGE = "bn"

# E5-instruct prefix config (inlined; was ec-faq-bot internal).
# Override `e5_instruction` in your config.yaml to use a domain-specific prompt.
DEFAULT_E5_INSTRUCTION = (
    "You are a careful matcher of FAQ questions to canonical intents. "
    "Identify the most semantically relevant question, considering context, "
    "intent, and specific details. Use semantic similarity and contextual "
    "understanding; prioritize exact phrase matches and context-aware matching."
)


def coerce_optional_str_list(value: Any) -> list[str]:
    """Convert tag-profile example fields to plain str lists.

    Parquet round-trips can yield numpy arrays; avoid `value or []` which raises
    ValueError on multi-element ndarray truthiness checks.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    try:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return [str(arr.item())]
        return [str(x) for x in arr.ravel().tolist()]
    except (TypeError, ValueError):
        return [str(value)]


Decision = Literal["keep", "jettison", "merge_candidate"]
ReasonCode = Literal[
    "duplicate",
    "too_generic",
    "wrong_intent",
    "sibling_collision",
    "synthetic_artifact",
    "context_dependent",
    "malformed",
    "clean",
]


@dataclass
class CleanerConfig:
    input_csv: Path = DEFAULT_INPUT_CSV
    tag_answer_json: Path = DEFAULT_TAG_ANSWER_JSON
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    run_id: str | None = None
    max_rows: int | None = None
    max_tags: int | None = None
    max_rows_per_tag: int | None = None
    target_tags: list[str] | None = None
    target_max_tags: int | None = None

    # Runtime and model settings.
    embedding_backend: str = "sentence-transformers"  # sentence-transformers | hashing
    e5_model: str = "intfloat/multilingual-e5-large-instruct"
    gemma_model: str = "google/embeddinggemma-300m"
    gemma_dim: int = 768
    hashing_dim: int = 384
    batch_size: int = 128
    device: str | None = None

    # E5-instruct prefix; override `e5_instruction` for domain-specific prompts.
    e5_use_prefixes: bool = True
    e5_instruction: str = DEFAULT_E5_INSTRUCTION

    # Language-specific text normalization. "bn" enables Bengali Unicode
    # normalization via bnunicodenormalizer; "none" disables.
    language: str = "bn"

    # GPT settings.
    openai_model: str = "gpt-5.4"
    openai_reasoning_effort: str = "high"
    judge_mode: str = "sync"  # sync | agents | batch_prepare | batch_submit | batch_collect | heuristic
    openai_batch_endpoint: str = "/v1/responses"
    openai_completion_window: str = "24h"
    concurrency: int = 6  # bounded; gpt-5.5 high reasoning saturates well below 32
    self_consistency_passes: int = 2
    judge_granularity: str = "tag_batch"  # tag_batch | row
    tags_per_judge_call: int = 3
    rows_per_tag_per_judge_call: int = 12
    # Stage 7 was a per-row second-pass over the bottom quantile of Stage 6
    # keeps; superseded by Stage 5 buffer audit in the new design. Default OFF.
    review_enabled: bool = False
    review_rows_per_tag: int = 8
    review_low_score_quantile: float = 0.25

    # Thresholds. "auto" values are calibrated from each run.
    e5_merge_threshold: float = 0.90
    gemma_merge_threshold: float = 0.88
    knn_overlap_threshold: float = 0.30
    near_duplicate_threshold: float = 0.98
    high_margin_quantile: float = 0.75
    low_support_threshold: int = 5
    top_k_competing_tags: int = 3
    evidence_top_k: int = 10
    central_examples: int = 5
    top_n: int = 40
    outlier_trim_fraction: float = 0.05

    # Redesigned audit pipeline: GPT moves out of per-row scoring.
    # Stage 3 writes a tag-boundary policy for close-tag clusters; Stage 4
    # ranks deterministically; Stage 5 audits only the top-N buffer per tag.
    audit_buffer_size: int = 80
    audit_rows_per_packet: int = 24
    boundary_policy_threshold: float = 0.85
    boundary_policy_max_cluster_size: int = 6

    # Validation.
    validation_top_k: int = 5
    validation_chunk_size: int = 4096

    # Stage 9: E5-only post-clean audit. Mimics production retrieval (E5 alone)
    # over the cleaned corpus and emits per-row neighborhood diagnostics.
    # Default drops rows whose top-1 LOO neighbor is a different tag — same
    # rows Stage 8 reports as confusions, but materialized as a filtered set.
    e5_audit_top_k: int = 10
    e5_audit_drop_on_top1_mismatch: bool = True
    stage9_input_csv: str | None = None  # override: external CSV (union of multiple runs)

    # Discover (multi-family scaling): regex pattern of tag names to exclude
    # from family discovery. Default empty; for the Bengali NID corpus pass
    # `_followup_[a-d]$` to skip dialog-turn artifacts that shouldn't form
    # close-tag siblings.
    discover_exclude_pattern: str = ""

    def resolved_run_id(self) -> str:
        return self.run_id or time.strftime("run_%Y%m%d_%H%M%S")

    def run_dir(self) -> Path:
        return self.artifact_root / self.resolved_run_id()


def resolve_target_tags(rows: pd.DataFrame, config: CleanerConfig, tag_to_canon: dict[str, str] | None = None) -> set[str]:
    """Return canonical tags selected for cleaning, while corpus can remain global."""
    if config.target_tags:
        selected = [str(tag) for tag in config.target_tags]
    elif config.target_max_tags is not None:
        selected = list(dict.fromkeys(rows["tag"].tolist()))[: config.target_max_tags]
    else:
        selected = list(dict.fromkeys(rows["tag"].tolist()))
    if tag_to_canon:
        return {tag_to_canon.get(tag, tag) for tag in selected}
    return set(selected)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    quality_score: float = Field(ge=0, le=100)
    ambiguity_score: float = Field(ge=0, le=100)
    context_dependent: bool
    reason_code: ReasonCode
    rationale: str = Field(min_length=1, max_length=500)


class RowJudgeDecision(JudgeResult):
    row_id: int


class JudgePacketResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_id: str
    decisions: list[RowJudgeDecision]
    packet_rationale: str = Field(min_length=1, max_length=1000)


class MergeCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    same_intent: bool
    confidence: float = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=500)


class AnswerSafetyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    same_answer: bool
    confidence: float = Field(ge=0, le=100)
    rationale: str = Field(min_length=1, max_length=500)


AuditDecisionLiteral = Literal["keep", "flag"]
AuditReasonCode = Literal[
    "clean",
    "wrong_intent",
    "sibling_collision",
    "too_generic",
    "duplicate",
    "synthetic_artifact",
    "context_dependent",
]


class TagBoundaryRule(BaseModel):
    """Per-tag distinguishing rule inside a close-tag cluster."""

    model_config = ConfigDict(extra="forbid")

    tag: str
    one_line_intent: str = Field(min_length=1, max_length=300)
    # OpenAI strict schemas require every property to also be in `required`.
    # Keeping these as required (no default_factory) so the GPT call validates;
    # GPT can supply empty arrays for tags with no specific concepts.
    must_have_concepts: list[str]
    must_avoid_concepts: list[str]


class BoundaryPolicyResult(BaseModel):
    """GPT output for a close-tag cluster: per-tag distinguishing rules."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    rules: list[TagBoundaryRule]
    cluster_rationale: str = Field(min_length=1, max_length=800)


class AuditRowDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: int
    decision: AuditDecisionLiteral
    reason_code: AuditReasonCode
    rationale: str = Field(min_length=1, max_length=400)


class AuditPacketResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packet_id: str
    decisions: list[AuditRowDecision]
    packet_rationale: str = Field(min_length=1, max_length=800)


def load_config(path: Path | None) -> CleanerConfig:
    raw: dict[str, Any] = {}
    if path:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    path_fields = {"input_csv", "tag_answer_json", "artifact_root"}
    for key in path_fields & raw.keys():
        raw[key] = Path(raw[key]).expanduser()
        if not raw[key].is_absolute():
            raw[key] = (PROJECT_ROOT / raw[key]).resolve()

    return CleanerConfig(**raw)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_hash(df: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha256()
    for row in df[columns].itertuples(index=False, name=None):
        h.update("\t".join("" if v is None else str(v) for v in row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def stage_done(stage_dir: Path, expected_hash: str | None) -> bool:
    manifest = read_json(stage_dir / "manifest.json")
    if not manifest:
        return False
    return expected_hash is None or manifest.get("input_hash") == expected_hash


def finish_stage(stage_dir: Path, config: CleanerConfig, input_hash: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "input_hash": input_hash,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
    }
    if extra:
        payload.update(extra)
    write_json(stage_dir / "manifest.json", payload)


_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_QUOTE_MAP = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "…": "...",
})


def normalize_question(text: str) -> str:
    """Normalize text for comparison/scoring while preserving raw text elsewhere."""
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = text.translate(_QUOTE_MAP).translate(_BENGALI_DIGITS)
    text = text.lower()

    if _NORMALIZATION_LANGUAGE == "bn":
        try:
            from bnunicodenormalizer import Normalizer

            normalizer = Normalizer()
            normalized = normalizer(text)
            if isinstance(normalized, dict) and normalized.get("normalized"):
                text = normalized["normalized"]
        except Exception:
            pass

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;:?!।])", r"\1", text)
    return text.strip()


def comparison_key(text: str) -> str:
    text = normalize_question(text)
    text = re.sub(r"[^\w\s\u0980-\u09ff]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_context_dependent(norm: str) -> bool:
    if not norm:
        return True
    tokens = norm.split()
    if len(tokens) <= 2 and norm in {
        "হ্যাঁ",
        "হা",
        "হ্যা",
        "জি",
        "না",
        "ok",
        "okay",
        "sure",
        "noted",
    }:
        return True
    generic = {
        "কিভাবে করবো",
        "কীভাবে করবো",
        "কি করবো",
        "কোথায় যাব",
        "কোথায় যাবো",
        "কোন অফিসে যোগাযোগ করবো",
        "পরবর্তী ধাপ কী",
        "আবেদন কিভাবে করবো",
        "আবেদন কীভাবে করবো",
    }
    return norm in generic


def artifact_like(raw: str) -> bool:
    raw = raw or ""
    suspicious = [
        "might go beyond",
        "answer only says",
        "could still lead",
        "however it's okay",
        "todo",
        "fixme",
    ]
    lower = raw.lower()
    return any(s in lower for s in suspicious)


def load_intake(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "stage0" / "intake.parquet")


def kept_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "keep"].reset_index(drop=True)


def run_stage0(config: CleanerConfig, resume: bool = True) -> None:
    stage_dir = config.run_dir() / "stage0"
    input_hash = file_sha256(config.input_csv)
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage0] skip: {stage_dir}")
        return

    rows: list[dict[str, Any]] = []
    with config.input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["question", "tag"]:
            raise ValueError(f"Expected CSV columns question,tag; got {reader.fieldnames}")
        seen_tags: list[str] = []
        seen_tag_set: set[str] = set()
        rows_seen_by_tag: Counter[str] = Counter()
        for idx, row in enumerate(reader, start=1):
            if config.max_rows is not None and idx > config.max_rows:
                break
            raw = (row.get("question") or "").strip()
            tag = (row.get("tag") or "").strip()
            if tag and tag not in seen_tag_set:
                if config.max_tags is not None and len(seen_tags) >= config.max_tags:
                    continue
                seen_tags.append(tag)
                seen_tag_set.add(tag)
            if config.max_tags is not None and tag not in seen_tag_set:
                continue
            if config.max_rows_per_tag is not None and rows_seen_by_tag[tag] >= config.max_rows_per_tag:
                continue
            rows_seen_by_tag[tag] += 1
            norm = normalize_question(raw)
            key = comparison_key(raw)
            status = "keep"
            reason = "clean"
            if not raw or not tag or not norm:
                status, reason = "jettison", "malformed"
            elif artifact_like(raw):
                status, reason = "jettison", "synthetic_artifact"
            elif is_context_dependent(norm):
                status, reason = "jettison", "context_dependent"
            rows.append(
                {
                    "row_id": idx,
                    "question_raw": raw,
                    "question_norm": norm,
                    "question_key": key,
                    "tag": tag,
                    "status": status,
                    "pre_reason": reason,
                }
            )

    df = pd.DataFrame(rows)
    keep_mask = df["status"] == "keep"
    duplicated_same_tag = df[keep_mask].duplicated(subset=["tag", "question_key"], keep="first")
    duplicate_indices = df[keep_mask][duplicated_same_tag].index
    df.loc[duplicate_indices, "status"] = "jettison"
    df.loc[duplicate_indices, "pre_reason"] = "duplicate"

    kept = df[df["status"] == "keep"]
    cross = (
        kept.groupby("question_key")
        .agg(tags=("tag", lambda x: sorted(set(x))), row_ids=("row_id", list), count=("row_id", "size"))
        .reset_index()
    )
    cross = cross[(cross["count"] > 1) & (cross["tags"].map(len) > 1)]

    stage_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(stage_dir / "intake.parquet", index=False)
    cross.to_json(stage_dir / "cross_tag_duplicates.jsonl", orient="records", lines=True, force_ascii=False)
    finish_stage(stage_dir, config, input_hash, {"rows": len(df), "kept": int((df["status"] == "keep").sum())})
    print(f"[stage0] wrote {stage_dir}")


def _hashing_embeddings(texts: list[str], dim: int) -> np.ndarray:
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        padded = f"  {text}  "
        features = [padded[i : i + n] for n in (2, 3, 4) for i in range(max(0, len(padded) - n + 1))]
        if not features:
            features = [text]
        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            col = value % dim
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vectors[row_idx, col] += sign
    return l2_normalize(vectors)


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _select_device(config: CleanerConfig) -> str | None:
    if config.device:
        return config.device
    if os.environ.get("STS_EMBEDDING_DEVICE"):
        return os.environ["STS_EMBEDDING_DEVICE"]
    return None


def _encode_sentence_transformer(model_id: str, texts: list[str], config: CleanerConfig) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    device = _select_device(config)
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(model_id, **kwargs)
    vectors = model.encode(
        texts,
        batch_size=config.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.astype(np.float32)


def _format_e5_passages(texts: list[str], config: CleanerConfig | None = None) -> list[str]:
    if config is not None and not config.e5_use_prefixes:
        return texts
    instruction = (config.e5_instruction if config else DEFAULT_E5_INSTRUCTION)
    prefix = f"Instruct: {instruction}\n"
    return [prefix + t for t in texts]


def _format_e5_queries(texts: list[str], config: CleanerConfig | None = None) -> list[str]:
    if config is not None and not config.e5_use_prefixes:
        return texts
    instruction = (config.e5_instruction if config else DEFAULT_E5_INSTRUCTION)
    prefix = f"Instruct: {instruction}\n"
    return [prefix + t for t in texts]


def _write_faiss_index(path: Path, vectors: np.ndarray) -> None:
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, str(path))


def run_stage1(config: CleanerConfig, resume: bool = True) -> None:
    run_stage0(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage1"
    intake = kept_rows(load_intake(run_dir))
    input_hash = dataframe_hash(intake, ["row_id", "question_norm", "tag"])
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage1] skip: {stage_dir}")
        return

    texts = intake["question_norm"].tolist()
    e5_passages = _format_e5_passages(texts, config)
    e5_queries = _format_e5_queries(texts, config)

    if config.embedding_backend == "hashing":
        emb_e5 = _hashing_embeddings(e5_passages, config.hashing_dim)
        emb_e5_query = _hashing_embeddings(e5_queries, config.hashing_dim)
        emb_gemma = _hashing_embeddings(texts, min(config.gemma_dim, config.hashing_dim))
    else:
        emb_e5 = _encode_sentence_transformer(config.e5_model, e5_passages, config)
        emb_e5_query = _encode_sentence_transformer(config.e5_model, e5_queries, config)
        emb_gemma = _encode_sentence_transformer(config.gemma_model, texts, config)
        if config.gemma_dim and emb_gemma.shape[1] > config.gemma_dim:
            emb_gemma = l2_normalize(emb_gemma[:, : config.gemma_dim])

    stage_dir.mkdir(parents=True, exist_ok=True)
    np.save(stage_dir / "emb_e5.npy", emb_e5)
    np.save(stage_dir / "emb_e5_query.npy", emb_e5_query)
    np.save(stage_dir / "emb_gemma.npy", emb_gemma)
    intake[["row_id", "tag", "question_norm", "question_raw"]].to_parquet(stage_dir / "embedding_rows.parquet", index=False)

    try:
        _write_faiss_index(stage_dir / "faiss_e5.idx", emb_e5)
        _write_faiss_index(stage_dir / "faiss_gemma.idx", emb_gemma)
    except ImportError:
        print("[stage1] faiss not installed; embeddings saved without FAISS indexes")

    finish_stage(stage_dir, config, input_hash, {"rows": len(intake), "e5_dim": emb_e5.shape[1], "gemma_dim": emb_gemma.shape[1]})
    print(f"[stage1] wrote {stage_dir}")


def _trimmed_indices(sims_to_centroid: np.ndarray, trim_fraction: float) -> np.ndarray:
    n = len(sims_to_centroid)
    if n < 10 or trim_fraction <= 0:
        return np.arange(n)
    order = np.argsort(sims_to_centroid)
    trim = max(1, int(n * trim_fraction))
    if n - 2 * trim <= 2:
        return np.arange(n)
    return order[trim:-trim]


def centroid(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        raise ValueError("Cannot compute centroid for empty vectors")
    return l2_normalize(vectors.mean(axis=0, keepdims=True))[0]


def _central_row_ids(group: pd.DataFrame, vectors: np.ndarray, local_indices: np.ndarray, k: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    raw_centroid = centroid(vectors[local_indices])
    sims = vectors @ raw_centroid
    central_order = np.argsort(-sims)[:k]
    row_ids = group.iloc[central_order]["row_id"].astype(int).tolist()
    return row_ids, raw_centroid, sims


def text_contains_concept(text: str, concept: str) -> bool:
    """Substring check with light normalization both sides."""
    if not concept:
        return False
    return normalize_question(concept) in normalize_question(text)


def token_alignment_score(text: str, must_have: list[str], must_avoid: list[str]) -> float:
    """Fraction of must_have concepts present in text minus avoid penalty.

    Returns a value roughly in [-1, 1]. Used as an additive feature in Stage 4
    scoring; weight is small so empty/missing policies degrade gracefully.
    """
    if not must_have and not must_avoid:
        return 0.0
    have_score = 0.0
    if must_have:
        hits = sum(1 for c in must_have if text_contains_concept(text, c))
        have_score = hits / len(must_have)
    avoid_pen = 0.0
    if must_avoid:
        hits = sum(1 for c in must_avoid if text_contains_concept(text, c))
        avoid_pen = hits / len(must_avoid)
    return have_score - avoid_pen


def artifact_score(text: str) -> float:
    """Higher = more artifact-like / suspicious. Range [0,1].

    Captures three deterministic synthesis tells: very short questions,
    repeated character runs, and visibly duplicated tokens. Used as a
    negative feature in Stage 4 ranking, never as a hard reject.
    """
    if not text:
        return 1.0
    norm = normalize_question(text)
    if not norm:
        return 1.0
    tokens = norm.split()
    score = 0.0
    if len(tokens) <= 3:
        score += 0.5
    if re.search(r"(.)\1{4,}", norm):
        score += 0.25
    seen: set[str] = set()
    dup_tokens = 0
    for t in tokens:
        if t in seen:
            dup_tokens += 1
        seen.add(t)
    if tokens and dup_tokens / len(tokens) > 0.3:
        score += 0.25
    return min(score, 1.0)


def _ngrams(text: str, n_values: tuple[int, ...] = (2, 3)) -> set[str]:
    tokens = text.split()
    grams: set[str] = set()
    for n in n_values:
        for i in range(0, len(tokens) - n + 1):
            grams.add(" ".join(tokens[i : i + n]))
    return grams


def discriminative_phrases(df: pd.DataFrame, tag: str, neighbor_tags: set[str], limit: int = 12) -> list[str]:
    tag_counts: Counter[str] = Counter()
    bg_counts: Counter[str] = Counter()
    for row in df[df["tag"] == tag]["question_norm"]:
        tag_counts.update(_ngrams(row))
    for row in df[df["tag"].isin(neighbor_tags)]["question_norm"]:
        bg_counts.update(_ngrams(row))
    tag_total = sum(tag_counts.values()) + 1
    bg_total = sum(bg_counts.values()) + 1
    scores = []
    for phrase, count in tag_counts.items():
        if len(phrase) < 4:
            continue
        score = math.log((count + 0.5) / tag_total) - math.log((bg_counts[phrase] + 0.5) / bg_total)
        scores.append((score, phrase))
    return [p for _, p in sorted(scores, reverse=True)[:limit]]


def run_stage2(config: CleanerConfig, resume: bool = True) -> None:
    run_stage1(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage2"
    rows = pd.read_parquet(run_dir / "stage1" / "embedding_rows.parquet")
    emb_e5 = np.load(run_dir / "stage1" / "emb_e5.npy")
    emb_gemma = np.load(run_dir / "stage1" / "emb_gemma.npy")
    input_hash = dataframe_hash(rows, ["row_id", "question_norm", "tag"])
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage2] skip: {stage_dir}")
        return

    profiles: list[dict[str, Any]] = []
    tag_order = sorted(rows["tag"].unique())
    e5_centroids: list[np.ndarray] = []
    gemma_centroids: list[np.ndarray] = []
    row_pos_by_id = {int(row_id): i for i, row_id in enumerate(rows["row_id"])}

    rough_centroids: dict[str, np.ndarray] = {}
    for tag in tag_order:
        idx = rows.index[rows["tag"] == tag].to_numpy()
        rough_centroids[tag] = centroid(emb_e5[idx])

    for tag in tag_order:
        group = rows[rows["tag"] == tag].reset_index(drop=False)
        idx = group["index"].to_numpy()
        rough = rough_centroids[tag]
        sims = emb_e5[idx] @ rough
        trimmed_local = _trimmed_indices(sims, config.outlier_trim_fraction)
        if len(trimmed_local) == 0:
            trimmed_local = np.arange(len(idx))

        central_ids, e5_c, e5_sims = _central_row_ids(group, emb_e5[idx], trimmed_local, config.central_examples)
        _, gemma_c, gemma_sims = _central_row_ids(group, emb_gemma[idx], trimmed_local, config.central_examples)

        e5_centroids.append(e5_c)
        gemma_centroids.append(gemma_c)
        outlier_scores = 1.0 - e5_sims
        central_questions = [
            rows.iloc[row_pos_by_id[row_id]]["question_raw"]
            for row_id in central_ids
            if row_id in row_pos_by_id
        ]

        neighbor_scores = [(other, float(e5_c @ c)) for other, c in rough_centroids.items() if other != tag]
        neighbor_tags = {t for t, _ in sorted(neighbor_scores, key=lambda item: item[1], reverse=True)[:25]}
        phrases = discriminative_phrases(rows, tag, neighbor_tags)

        description = synthesize_description(tag, central_questions)
        profiles.append(
            {
                "tag": tag,
                "row_count": int(len(idx)),
                "central_row_ids": central_ids,
                "central_questions": central_questions,
                "description": description,
                "description_stable": True,
                "discriminative_phrases": phrases,
                "e5_mean_sim_to_centroid": float(np.mean(e5_sims)),
                "e5_min_sim_to_centroid": float(np.min(e5_sims)),
                "e5_diversity": float(1.0 - np.mean((emb_e5[idx] @ e5_c))),
                "gemma_diversity": float(1.0 - np.mean((emb_gemma[idx] @ gemma_c))),
                "outlier_p95": float(np.quantile(outlier_scores, 0.95)) if len(outlier_scores) else 0.0,
            }
        )

    stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(profiles).to_parquet(stage_dir / "tag_profile.parquet", index=False)
    write_jsonl(stage_dir / "tag_description.jsonl", profiles)
    np.save(stage_dir / "tag_centroids_e5.npy", np.vstack(e5_centroids).astype(np.float32))
    np.save(stage_dir / "tag_centroids_gemma.npy", np.vstack(gemma_centroids).astype(np.float32))
    write_json(stage_dir / "tag_index.json", {"tags": tag_order})
    finish_stage(stage_dir, config, input_hash, {"tags": len(tag_order)})
    print(f"[stage2] wrote {stage_dir}")


def synthesize_description(tag: str, examples: list[str]) -> str:
    cleaned = tag.replace("_", " ")
    if not examples:
        return f"Questions about {cleaned}."
    sample = "; ".join(examples[:2])
    return f"Questions about {cleaned}. Central examples: {sample}"


def canonical_tag_name(tags: list[str], row_counts: dict[str, int]) -> str:
    def key(tag: str) -> tuple[int, int, int, str]:
        is_followup = 1 if "_followup_" in tag else 0
        has_numeric = 1 if re.search(r"(?:^|_)\d+$|_\d+_", tag) else 0
        return (is_followup, has_numeric, -row_counts.get(tag, 0), tag)

    return sorted(tags, key=key)[0]


def find_close_tag_clusters(
    tags: list[str],
    sim_e5: np.ndarray,
    sim_gemma: np.ndarray,
    threshold: float,
    max_cluster_size: int,
) -> list[list[str]]:
    """Group tags whose centroids are mutually close in BOTH embedding spaces.

    Edge: (i,j) if min(sim_e5[i,j], sim_gemma[i,j]) >= threshold.
    Returns connected components, sorted by size desc, capped per cluster.
    Singleton tags are excluded — boundary policy is only useful for >=2 tags.
    """
    n = len(tags)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if min(float(sim_e5[i, j]), float(sim_gemma[i, j])) >= threshold:
                union(i, j)

    groups: defaultdict[int, list[str]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(tags[i])

    clusters = [members for members in groups.values() if len(members) >= 2]
    clusters.sort(key=len, reverse=True)
    return [members[:max_cluster_size] for members in clusters]


def _knn_overlap(rows: pd.DataFrame, tag_a: str, tag_b: str, vectors: np.ndarray, k: int = 10) -> float:
    idx_a = rows.index[rows["tag"] == tag_a].to_numpy()
    idx_b = rows.index[rows["tag"] == tag_b].to_numpy()
    if len(idx_a) == 0 or len(idx_b) == 0:
        return 0.0
    sample_a = idx_a[: min(len(idx_a), 50)]
    sample_b = idx_b[: min(len(idx_b), 50)]
    all_idx = np.concatenate([idx_a, idx_b])
    sims_a = vectors[sample_a] @ vectors[all_idx].T
    sims_b = vectors[sample_b] @ vectors[all_idx].T
    b_positions = set(range(len(idx_a), len(all_idx)))
    a_positions = set(range(0, len(idx_a)))

    def frac_cross(sims: np.ndarray, target_positions: set[int]) -> float:
        hits = 0
        total = 0
        for row in sims:
            top = np.argsort(-row)[: min(k, len(row))]
            hits += sum(1 for pos in top if pos in target_positions)
            total += len(top)
        return hits / total if total else 0.0

    return (frac_cross(sims_a, b_positions) + frac_cross(sims_b, a_positions)) / 2


def heuristic_same_intent(tag_a: str, tag_b: str, examples_a: list[str], examples_b: list[str]) -> MergeCheckResult:
    base_a = re.sub(r"_followup_[a-z]$", "", tag_a)
    base_b = re.sub(r"_followup_[a-z]$", "", tag_b)
    if base_a == base_b:
        return MergeCheckResult(same_intent=True, confidence=92, rationale="Tags share the same base name.")
    token_a = set(base_a.split("_"))
    token_b = set(base_b.split("_"))
    overlap = len(token_a & token_b) / max(1, len(token_a | token_b))
    return MergeCheckResult(
        same_intent=overlap > 0.72,
        confidence=round(100 * overlap, 2),
        rationale=f"Token overlap={overlap:.2f}; heuristic fallback used.",
    )


def heuristic_same_answer(answer_a: str, answer_b: str) -> AnswerSafetyResult:
    key_a = set(comparison_key(answer_a).split())
    key_b = set(comparison_key(answer_b).split())
    overlap = len(key_a & key_b) / max(1, len(key_a | key_b))
    return AnswerSafetyResult(
        same_answer=overlap > 0.72,
        confidence=round(100 * overlap, 2),
        rationale=f"Answer token overlap={overlap:.2f}; heuristic fallback used.",
    )


def run_stage3(config: CleanerConfig, resume: bool = True) -> None:
    run_stage2(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage3"
    rows = pd.read_parquet(run_dir / "stage1" / "embedding_rows.parquet")
    profiles = pd.read_parquet(run_dir / "stage2" / "tag_profile.parquet")
    tag_index = read_json(run_dir / "stage2" / "tag_index.json")
    tags = tag_index["tags"]
    e5_centroids = np.load(run_dir / "stage2" / "tag_centroids_e5.npy")
    gemma_centroids = np.load(run_dir / "stage2" / "tag_centroids_gemma.npy")
    emb_e5 = np.load(run_dir / "stage1" / "emb_e5.npy")
    hash_parts = [
        dataframe_hash(profiles, ["tag", "row_count", "description"]),
        file_sha256(config.tag_answer_json),
    ]
    if config.target_tags:
        # Scope-aware hash: a target-restricted run is a different artifact
        # than a corpus-wide one and must not silently resume from each other.
        hash_parts.append("target_tags=" + ",".join(sorted(config.target_tags)))
    input_hash = hashlib.sha256("|".join(hash_parts).encode("utf-8")).hexdigest()
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage3] skip: {stage_dir}")
        return

    answers = read_json(config.tag_answer_json, {})
    row_counts = profiles.set_index("tag")["row_count"].astype(int).to_dict()
    profile_map = profiles.set_index("tag").to_dict(orient="index")

    sim_e5 = e5_centroids @ e5_centroids.T
    sim_gemma = gemma_centroids @ gemma_centroids.T
    candidates: list[dict[str, Any]] = []
    for i, tag_a in enumerate(tags):
        for j in range(i + 1, len(tags)):
            if sim_e5[i, j] < config.e5_merge_threshold or sim_gemma[i, j] < config.gemma_merge_threshold:
                continue
            tag_b = tags[j]
            overlap = _knn_overlap(rows, tag_a, tag_b, emb_e5)
            if overlap < config.knn_overlap_threshold:
                continue
            examples_a = coerce_optional_str_list(profile_map[tag_a].get("central_questions"))
            examples_b = coerce_optional_str_list(profile_map[tag_b].get("central_questions"))
            intent = heuristic_same_intent(tag_a, tag_b, examples_a, examples_b)
            if not intent.same_intent:
                continue
            safety = heuristic_same_answer(str(answers.get(tag_a, "")), str(answers.get(tag_b, "")))
            candidates.append(
                {
                    "tag_a": tag_a,
                    "tag_b": tag_b,
                    "e5_centroid_sim": float(sim_e5[i, j]),
                    "gemma_centroid_sim": float(sim_gemma[i, j]),
                    "knn_overlap": float(overlap),
                    "question_equivalence": intent.model_dump(),
                    "answer_safety": safety.model_dump(),
                    "merge": bool(intent.same_intent and safety.same_answer),
                    "boundary_confusion": bool(intent.same_intent and not safety.same_answer),
                }
            )

    parent = {tag: tag for tag in tags}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        canon = canonical_tag_name([ra, rb], row_counts)
        other = rb if canon == ra else ra
        parent[other] = canon
        parent[canon] = canon

    for candidate in candidates:
        if candidate["merge"]:
            union(candidate["tag_a"], candidate["tag_b"])

    groups: defaultdict[str, list[str]] = defaultdict(list)
    for tag in tags:
        groups[find(tag)].append(tag)

    remap = []
    for _, members in groups.items():
        canon = canonical_tag_name(members, row_counts)
        for tag in members:
            remap.append(
                {
                    "old_tag": tag,
                    "canonical_tag": canon,
                    "merged": tag != canon,
                    "merge_confidence": 100.0 if tag != canon else 0.0,
                    "reason": "high_confidence_question_and_answer_equivalence" if tag != canon else "canonical_or_unmerged",
                }
            )

    stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(remap).to_csv(stage_dir / "tag_merge_map.csv", index=False)
    pd.DataFrame(candidates).to_json(stage_dir / "merge_candidates.jsonl", orient="records", lines=True, force_ascii=False)

    # --- Boundary-policy authoring for close-tag clusters that did NOT merge.
    # The merge gates are tighter than the boundary threshold; clusters that
    # hover near each other but stayed separate are exactly where Stage 5
    # audit needs explicit discriminative rules.
    canon_for_tag = {row["old_tag"]: row["canonical_tag"] for row in remap}
    raw_clusters = find_close_tag_clusters(
        list(tags),
        np.asarray(sim_e5, dtype=np.float32),
        np.asarray(sim_gemma, dtype=np.float32),
        threshold=config.boundary_policy_threshold,
        max_cluster_size=config.boundary_policy_max_cluster_size,
    )
    distinct_clusters: list[list[str]] = []
    target_canon = (
        [canon_for_tag.get(t, t) for t in config.target_tags]
        if config.target_tags
        else None
    )
    if target_canon is not None and len(set(target_canon)) == 1:
        # Singleton target — no sibling cluster to disambiguate against.
        # Emit no boundary policy; Stage 5 audits the lone tag's buffer with
        # central exemplars + discriminative phrases only.
        distinct_clusters = []
    elif target_canon is not None and len(set(target_canon)) >= 2:
        # User-provided cluster: --target-tags or --seed-tag's resolved set
        # IS the cluster Stage 3 should author a policy for. Skipping the
        # union-find avoids two failure modes at corpus scale:
        # (1) target tags landing in a 400+-tag mega-component that gets
        #     truncated to alphabetically-first 6 by max_cluster_size,
        #     dropping the targets entirely;
        # (2) burning ~100+ GPT calls authoring policies for irrelevant
        #     clusters Stage 5 will never use.
        distinct_clusters = [list(dict.fromkeys(target_canon))]
    else:
        for members in raw_clusters:
            canonicals = {canon_for_tag.get(t, t) for t in members}
            if len(canonicals) < 2:
                continue
            if target_canon is not None and not (set(target_canon) & canonicals):
                continue
            distinct_clusters.append(members)

    policy_records: list[dict[str, Any]] = []
    for cluster_idx, cluster_tags in enumerate(distinct_clusters):
        cluster_id = f"cluster_{cluster_idx:04d}"
        policy = compute_boundary_policy(cluster_id, cluster_tags, profile_map, config)
        policy_records.append({
            "cluster_id": cluster_id,
            "tags": cluster_tags,
            "rules": [r.model_dump() for r in policy.rules],
            "rationale": policy.cluster_rationale,
        })
    write_jsonl(stage_dir / "tag_boundary_policy.jsonl", policy_records)

    finish_stage(
        stage_dir,
        config,
        input_hash,
        {
            "merge_candidates": len(candidates),
            "merged_tags": sum(1 for r in remap if r["merged"]),
            "boundary_clusters": len(policy_records),
        },
    )
    print(f"[stage3] wrote {stage_dir}")


def _recompute_canonical_centroids(rows: pd.DataFrame, vectors: np.ndarray, tag_col: str = "canonical_tag") -> tuple[list[str], np.ndarray]:
    tags = sorted(rows[tag_col].unique())
    cents = []
    for tag in tags:
        idx = rows.index[rows[tag_col] == tag].to_numpy()
        cents.append(centroid(vectors[idx]))
    return tags, np.vstack(cents).astype(np.float32)


def _tag_scores(vectors: np.ndarray, centroids_: np.ndarray) -> np.ndarray:
    return vectors @ centroids_.T


def _calibrated_threshold(values: np.ndarray, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    return float(np.quantile(finite, quantile))


def _neighbor_evidence(
    pos: int,
    query_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    rows: pd.DataFrame,
    top_k: int,
) -> list[dict[str, Any]]:
    sims = corpus_vectors @ query_vector
    sims[pos] = -np.inf
    top = np.argsort(-sims)[: min(top_k, max(0, len(rows) - 1))]
    evidence = []
    for rank, idx in enumerate(top, start=1):
        if not np.isfinite(sims[idx]):
            continue
        evidence.append(
            {
                "rank": rank,
                "row_id": int(rows.iloc[idx]["row_id"]),
                "tag": rows.iloc[idx]["canonical_tag"],
                "original_tag": rows.iloc[idx]["tag"],
                "similarity": round(float(sims[idx]), 6),
                "question": rows.iloc[idx]["question_raw"],
            }
        )
    return evidence


def _evidence_summary(e5_evidence: list[dict[str, Any]], gemma_evidence: list[dict[str, Any]], current_tag: str) -> dict[str, Any]:
    e5_tags = [item["tag"] for item in e5_evidence]
    gemma_tags = [item["tag"] for item in gemma_evidence]
    e5_counts = Counter(e5_tags)
    gemma_counts = Counter(gemma_tags)
    e5_ids = {item["row_id"] for item in e5_evidence}
    gemma_ids = {item["row_id"] for item in gemma_evidence}

    def first_rank(items: list[dict[str, Any]], tag: str) -> int | None:
        for item in items:
            if item["tag"] == tag:
                return int(item["rank"])
        return None

    return {
        "e5_tag_counts": dict(e5_counts.most_common()),
        "gemma_tag_counts": dict(gemma_counts.most_common()),
        "e5_current_tag_first_rank": first_rank(e5_evidence, current_tag),
        "gemma_current_tag_first_rank": first_rank(gemma_evidence, current_tag),
        "top_neighbor_overlap_count": len(e5_ids & gemma_ids),
        "top_neighbor_overlap_row_ids": sorted(e5_ids & gemma_ids),
        "e5_top_tag": e5_tags[0] if e5_tags else None,
        "gemma_top_tag": gemma_tags[0] if gemma_tags else None,
        "embedding_top_tag_agreement": bool(e5_tags and gemma_tags and e5_tags[0] == gemma_tags[0]),
    }


def run_stage4(config: CleanerConfig, resume: bool = True) -> None:
    run_stage3(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage4"
    rows = pd.read_parquet(run_dir / "stage1" / "embedding_rows.parquet").reset_index(drop=True)
    emb_e5 = np.load(run_dir / "stage1" / "emb_e5.npy")
    emb_gemma = np.load(run_dir / "stage1" / "emb_gemma.npy")
    merge_map = pd.read_csv(run_dir / "stage3" / "tag_merge_map.csv")
    tag_to_canon = dict(zip(merge_map["old_tag"], merge_map["canonical_tag"]))
    rows["canonical_tag"] = rows["tag"].map(tag_to_canon).fillna(rows["tag"])
    target_canonical_tags = resolve_target_tags(rows, config, tag_to_canon)
    rows["target_scope"] = rows["canonical_tag"].isin(target_canonical_tags)
    input_hash = dataframe_hash(rows, ["row_id", "question_norm", "canonical_tag", "target_scope"])
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage4] skip: {stage_dir}")
        return

    tags, e5_cents = _recompute_canonical_centroids(rows, emb_e5)
    _, gemma_cents = _recompute_canonical_centroids(rows, emb_gemma)
    tag_to_idx = {tag: i for i, tag in enumerate(tags)}
    e5_scores = _tag_scores(emb_e5, e5_cents)
    gemma_scores = _tag_scores(emb_gemma, gemma_cents)

    cross_dups = read_jsonl(run_dir / "stage0" / "cross_tag_duplicates.jsonl")
    cross_dup_ids = {int(row_id) for item in cross_dups for row_id in item.get("row_ids", [])}

    features = []
    e5_margins = []
    gemma_margins = []
    for pos, row in rows.iterrows():
        e5_evidence = _neighbor_evidence(pos, emb_e5[pos], emb_e5, rows, config.evidence_top_k)
        gemma_evidence = _neighbor_evidence(pos, emb_gemma[pos], emb_gemma, rows, config.evidence_top_k)
        evidence_summary = _evidence_summary(e5_evidence, gemma_evidence, row["canonical_tag"])
        own_idx = tag_to_idx[row["canonical_tag"]]
        e5_row = e5_scores[pos].copy()
        gemma_row = gemma_scores[pos].copy()
        e5_own = float(e5_row[own_idx])
        gemma_own = float(gemma_row[own_idx])
        e5_row[own_idx] = -np.inf
        gemma_row[own_idx] = -np.inf
        e5_comp_idx = int(np.argmax(e5_row))
        gemma_comp_idx = int(np.argmax(gemma_row))
        e5_comp = float(e5_row[e5_comp_idx])
        gemma_comp = float(gemma_row[gemma_comp_idx])
        e5_margin = e5_own - e5_comp
        gemma_margin = gemma_own - gemma_comp
        e5_margins.append(e5_margin)
        gemma_margins.append(gemma_margin)
        features.append(
            {
                "row_id": int(row["row_id"]),
                "question_raw": row["question_raw"],
                "question_norm": row["question_norm"],
                "tag": row["tag"],
                "canonical_tag": row["canonical_tag"],
                "target_scope": bool(row["target_scope"]),
                "e5_own_sim": e5_own,
                "e5_top1_competing_tag": tags[e5_comp_idx],
                "e5_top1_competing_sim": e5_comp,
                "e5_margin": e5_margin,
                "gemma_own_sim": gemma_own,
                "gemma_top1_competing_tag": tags[gemma_comp_idx],
                "gemma_top1_competing_sim": gemma_comp,
                "gemma_margin": gemma_margin,
                "rank_agreement": tags[e5_comp_idx] == tags[gemma_comp_idx],
                "e5_top10_evidence": json.dumps(e5_evidence, ensure_ascii=False),
                "gemma_top10_evidence": json.dumps(gemma_evidence, ensure_ascii=False),
                "embedding_reconciliation": json.dumps(evidence_summary, ensure_ascii=False),
                "cross_tag_duplicate": int(row["row_id"]) in cross_dup_ids,
            }
        )

    feature_df = pd.DataFrame(features)
    near_dup_counts = []
    for tag, group in feature_df.groupby("canonical_tag"):
        idx = group.index.to_numpy()
        sims = emb_e5[idx] @ emb_e5[idx].T
        counts = (sims > config.near_duplicate_threshold).sum(axis=1) - 1
        near_dup_counts.extend(zip(idx, counts.tolist()))
    feature_df["near_dup_count"] = 0
    for idx, count in near_dup_counts:
        feature_df.loc[idx, "near_dup_count"] = int(count)

    # Load boundary policy authored in Stage 3 (may be empty for non-clustered tags).
    policy_records = read_jsonl(run_dir / "stage3" / "tag_boundary_policy.jsonl")
    tag_to_rule: dict[str, dict[str, list[str]]] = {}
    for cluster in policy_records:
        for rule in cluster.get("rules", []) or []:
            tag = rule.get("tag")
            if not tag:
                continue
            tag_to_rule[tag] = {
                "must_have": [str(c) for c in rule.get("must_have_concepts") or []],
                "must_avoid": [str(c) for c in rule.get("must_avoid_concepts") or []],
            }

    feature_df["token_alignment"] = 0.0
    feature_df["artifact_score"] = 0.0
    for idx, row in feature_df.iterrows():
        rule = tag_to_rule.get(row["canonical_tag"])
        if rule is not None:
            feature_df.loc[idx, "token_alignment"] = token_alignment_score(
                row["question_norm"], rule["must_have"], rule["must_avoid"],
            )
        feature_df.loc[idx, "artifact_score"] = artifact_score(row["question_raw"])

    # Deterministic composite score. Weights chosen so a typical clean row lands
    # near 1.0; problematic rows fall well below. No GPT signal here.
    e5_own_n = _normalize_series(feature_df["e5_own_sim"])
    e5_margin_n = _normalize_series(feature_df["e5_margin"])
    gemma_own_n = _normalize_series(feature_df["gemma_own_sim"])
    gemma_margin_n = _normalize_series(feature_df["gemma_margin"])
    near_dup_n = _normalize_series(feature_df["near_dup_count"].astype(float))
    rank_bonus = feature_df["rank_agreement"].astype(float)
    cross_dup_pen = feature_df["cross_tag_duplicate"].astype(float)

    feature_df["composite_score"] = (
        0.30 * e5_own_n
        + 0.20 * e5_margin_n
        + 0.15 * gemma_own_n
        + 0.10 * gemma_margin_n
        + 0.10 * rank_bonus
        + 0.10 * feature_df["token_alignment"].clip(lower=-1.0, upper=1.0)
        - 0.05 * near_dup_n
        - 0.10 * cross_dup_pen
        - 0.10 * feature_df["artifact_score"]
    )

    # Audit buffer: top N per canonical tag (within target scope) by composite_score.
    feature_df["audit_buffer"] = False
    in_scope = feature_df[feature_df["target_scope"]].copy()
    if not in_scope.empty:
        in_scope = in_scope.sort_values(
            ["canonical_tag", "composite_score", "row_id"], ascending=[True, False, True]
        )
        buffer_idx = (
            in_scope.groupby("canonical_tag", sort=False)
            .head(config.audit_buffer_size)
            .index
        )
        feature_df.loc[buffer_idx, "audit_buffer"] = True

    stage_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(stage_dir / "row_features.parquet", index=False)
    feature_df[["row_id", "canonical_tag", "composite_score", "audit_buffer", "target_scope"]].to_json(
        stage_dir / "row_score.jsonl", orient="records", lines=True, force_ascii=False,
    )
    finish_stage(
        stage_dir,
        config,
        input_hash,
        {
            "audit_buffer_rows": int(feature_df["audit_buffer"].sum()),
            "out_of_scope": int((~feature_df["target_scope"]).sum()),
            "target_tags": sorted(target_canonical_tags),
            "boundary_policy_tags_covered": len(tag_to_rule),
        },
    )
    print(f"[stage4] wrote {stage_dir}")


def _compact_examples(values: Any, limit: int = 5) -> list[str]:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, str):
        try:
            decoded = json.loads(values)
            values = decoded
        except Exception:
            return [values][:limit]
    if not isinstance(values, list):
        return []
    return [str(v) for v in values[:limit]]


def _json_cell(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _candidate_tags_from_embedding_evidence(
    row: pd.Series,
    profiles: dict[str, dict[str, Any]],
    exclude_tags: set[str],
    limit: int,
) -> list[str]:
    """Rank candidate tags from concrete E5/Gemma top-k evidence.

    The centroid competitors are useful, but the strongest GPT context should
    come from the tags that actually dominate the two retrieval top-10 lists.
    """
    counts: Counter[str] = Counter()
    for source_name in ("e5_top10_evidence", "gemma_top10_evidence"):
        for item in _json_cell(row.get(source_name), []):
            tag = item.get("tag")
            rank = int(item.get("rank") or 999)
            if tag and tag in profiles and tag not in exclude_tags:
                # Weight earlier neighbors more, but still let repeated tags win.
                counts[tag] += max(1.0, 11.0 - min(rank, 10))

    for tag in (row.get("e5_top1_competing_tag"), row.get("gemma_top1_competing_tag")):
        if tag and tag in profiles and tag not in exclude_tags:
            counts[tag] += 0.5

    return [tag for tag, _ in counts.most_common(limit)]


def build_judge_prompt(row: pd.Series, profiles: dict[str, dict[str, Any]], config: CleanerConfig) -> str:
    current = profiles.get(row["canonical_tag"], {})
    competing_tags = _candidate_tags_from_embedding_evidence(
        row,
        profiles,
        exclude_tags={row["canonical_tag"]},
        limit=config.top_k_competing_tags,
    )

    payload = {
        "task": "Decide whether the target question is a clean example for its current tag. Do not relabel. If uncertain, jettison.",
        "target_question_raw": row["question_raw"],
        "target_question_normalized": row["question_norm"],
        "current_tag": row["canonical_tag"],
        "current_tag_description": current.get("description"),
        "current_tag_central_examples": _compact_examples(current.get("central_questions"), config.central_examples),
        "current_tag_discriminative_phrases": _compact_examples(current.get("discriminative_phrases"), 12),
        "embedding_reconciliation": _json_cell(row.get("embedding_reconciliation"), {}),
        "e5_top10_neighbors": _json_cell(row.get("e5_top10_evidence"), []),
        "gemma_top10_neighbors": _json_cell(row.get("gemma_top10_evidence"), []),
        "competing_tags": [
            {
                "tag": tag,
                "description": profiles[tag].get("description"),
                "central_examples": _compact_examples(profiles[tag].get("central_questions"), config.central_examples),
                "discriminative_phrases": _compact_examples(profiles[tag].get("discriminative_phrases"), 12),
            }
            for tag in competing_tags
        ],
        "decision_policy": {
            "keep": "The question is clear, self-contained, natural enough, and belongs to the current tag more than competitors.",
            "jettison": "Use for ambiguous, generic, context-dependent, wrong-intent, duplicate-like, or synthetic-artifact rows.",
            "merge_candidate": "Use only when this row suggests current and competing tags may be indistinguishable.",
        },
        "required_json_schema": JudgeResult.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_judge_packets(feature_df: pd.DataFrame, config: CleanerConfig) -> list[pd.DataFrame]:
    """Create stable, bounded judge packets grouped by canonical tag.

    Each packet contains up to `tags_per_judge_call` tags and up to
    `rows_per_tag_per_judge_call` rows per tag. This keeps context localized
    and prevents a single prompt from becoming a dumping ground for the whole
    dataset.
    """
    judge_rows = feature_df[(feature_df["route"] == "judge") & (feature_df.get("target_scope", True))].copy()
    if judge_rows.empty:
        return []

    packets: list[pd.DataFrame] = []
    chunks_by_tag: dict[str, list[pd.DataFrame]] = {}
    for tag, group in judge_rows.sort_values(["canonical_tag", "e5_margin", "gemma_margin"]).groupby("canonical_tag", sort=True):
        group = group.sort_values(["e5_margin", "gemma_margin", "row_id"], ascending=[True, True, True])
        chunks_by_tag[tag] = []
        for start in range(0, len(group), config.rows_per_tag_per_judge_call):
            chunks_by_tag[tag].append(group.iloc[start : start + config.rows_per_tag_per_judge_call])

    tags = sorted(chunks_by_tag)
    while any(chunks_by_tag[tag] for tag in tags):
        active_tags = [tag for tag in tags if chunks_by_tag[tag]][: config.tags_per_judge_call]
        chunks = [chunks_by_tag[tag].pop(0) for tag in active_tags]
        packets.append(pd.concat(chunks, ignore_index=True))
    return packets


def packet_result_path(stage_dir: Path, packet_id: str, pass_idx: int) -> Path:
    safe_id = packet_id.replace(":", "_")
    return stage_dir / "packet_results" / f"{safe_id}_pass_{pass_idx}.json"


def row_result_path(stage_dir: Path, row_id: int, pass_idx: int) -> Path:
    return stage_dir / "row_results" / f"row_{row_id}_pass_{pass_idx}.json"


def _decision_to_judge(decision: RowJudgeDecision) -> JudgeResult:
    return JudgeResult(
        decision=decision.decision,
        quality_score=decision.quality_score,
        ambiguity_score=decision.ambiguity_score,
        context_dependent=decision.context_dependent,
        reason_code=decision.reason_code,
        rationale=decision.rationale,
    )


def aggregate_incremental_judge_results(stage_dir: Path, expected_row_ids: Iterable[int]) -> None:
    grouped: defaultdict[int, list[JudgeResult]] = defaultdict(list)
    expected = {int(row_id) for row_id in expected_row_ids}

    for path in sorted((stage_dir / "row_results").glob("row_*_pass_*.json")):
        payload = read_json(path)
        if not payload:
            continue
        row_id = int(payload["row_id"])
        if row_id in expected:
            grouped[row_id].append(JudgeResult.model_validate(payload["result"]))

    for path in sorted((stage_dir / "packet_results").glob("packet_*_pass_*.json")):
        payload = read_json(path)
        if not payload:
            continue
        packet = JudgePacketResult.model_validate(payload["result"])
        for decision in packet.decisions:
            if decision.row_id in expected:
                grouped[decision.row_id].append(_decision_to_judge(decision))

    rows = [_resolve_consistency(row_id, grouped.get(row_id, [])) for row_id in sorted(expected)]
    write_jsonl(stage_dir / "judge_results.jsonl", rows)


def missing_row_ids_in_packet_result(packet: pd.DataFrame, result: JudgePacketResult) -> list[int]:
    expected = {int(v) for v in packet["row_id"].tolist()}
    received = {int(d.row_id) for d in result.decisions}
    return sorted(expected - received)


async def _fill_packet_misses(
    config: CleanerConfig,
    stage_dir: Path,
    packet: pd.DataFrame,
    packet_result: JudgePacketResult,
    pass_idx: int,
    profiles: dict[str, dict[str, Any]],
    judge_one: Any,
) -> int:
    """Run a row-level fallback for any expected row_id the packet response omitted.

    Per-row results land in row_results/ and the existing aggregator picks them up.
    Skips rows that already have a row-level result on disk so resumes are cheap.
    """
    misses = missing_row_ids_in_packet_result(packet, packet_result)
    if not misses:
        return 0
    print(f"[stage5] packet missing {len(misses)} decision(s); filling per-row")
    filled = 0
    for row_id in misses:
        out_path = row_result_path(stage_dir, row_id, pass_idx)
        if out_path.exists():
            continue
        row = packet[packet["row_id"] == row_id].iloc[0]
        prompt = build_judge_prompt(row, profiles, config)
        try:
            result = await judge_one(prompt)
        except Exception as exc:
            print(f"[stage5] row {row_id} fill failed after retries: {exc}")
            continue
        write_json(out_path, {"row_id": row_id, "pass_idx": pass_idx, "result": result.model_dump()})
        filled += 1
    return filled


def build_boundary_policy_prompt(
    cluster_id: str,
    cluster_tags: list[str],
    profiles: dict[str, dict[str, Any]],
    config: CleanerConfig,
) -> str:
    """Prompt GPT to author the distinguishing rules for a close-tag cluster."""

    def tag_context(tag: str) -> dict[str, Any]:
        profile = profiles.get(tag, {})
        return {
            "tag": tag,
            "row_count": int(profile.get("row_count", 0) or 0),
            "central_examples": _compact_examples(profile.get("central_questions"), config.central_examples),
            "discriminative_phrases": _compact_examples(profile.get("discriminative_phrases"), 10),
        }

    # Structured spec following the GPT-5 prompting guide: explicit task,
    # concrete rules without redundant emphasis, examples grounded in evidence.
    payload = {
        "task": (
            "Author distinguishing rules for each tag in this close-tag cluster. "
            "The downstream audit will use these rules to decide whether a question "
            "belongs to its current tag, without relabeling. For each tag produce: "
            "one_line_intent, must_have_concepts, must_avoid_concepts."
        ),
        "cluster_id": cluster_id,
        "tags_in_cluster": [tag_context(tag) for tag in cluster_tags],
        "concept_spec": {
            "language": "Use the natural language of the central_examples. Concepts "
                "may be words, short phrases, or short patterns (e.g. 'agent imperative').",
            "must_have_concepts": "3 to 7 concrete cues a clean question for THIS tag "
                "would mention or imply. Empty list is valid for the most generic tag "
                "in a cluster — do not invent forced cues.",
            "must_avoid_concepts": "0 to 5 cues that signal a SIBLING tag instead.",
            "no_generic_concepts": "Skip concepts that fit every tag in the cluster.",
            "prefer_concrete_phrases": True,
            "evidence_only": "Base concepts on the supplied central_examples and "
                "discriminative_phrases; do not invent.",
        },
        "required_json_schema": BoundaryPolicyResult.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _heuristic_boundary_policy(cluster_id: str, cluster_tags: list[str]) -> BoundaryPolicyResult:
    """Stub policy used when GPT is disabled or unreachable.

    Builds must_have_concepts from tag-name tokens (minus the shared prefix) so the
    Stage 4 token-alignment feature still has a small useful signal.
    """
    common_tokens: set[str] | None = None
    tokenized = []
    for tag in cluster_tags:
        toks = [t for t in tag.split("_") if t]
        tokenized.append(toks)
        common_tokens = set(toks) if common_tokens is None else common_tokens & set(toks)
    common = common_tokens or set()
    rules = []
    for tag, toks in zip(cluster_tags, tokenized):
        distinctive = [t for t in toks if t not in common][:6]
        rules.append(
            TagBoundaryRule(
                tag=tag,
                one_line_intent=f"Heuristic stub for {tag}.",
                must_have_concepts=distinctive,
                must_avoid_concepts=[],
            )
        )
    return BoundaryPolicyResult(
        cluster_id=cluster_id,
        rules=rules,
        cluster_rationale="Heuristic fallback (no GPT available); concepts derived from tag-name tokens.",
    )


def compute_boundary_policy(
    cluster_id: str,
    cluster_tags: list[str],
    profiles: dict[str, dict[str, Any]],
    config: CleanerConfig,
) -> BoundaryPolicyResult:
    """Run GPT (or heuristic fallback) to produce a boundary policy for the cluster.

    Single pass, fixed tag order, high reasoning. Structural validation only:
    schema must validate AND the returned rule set must cover the input tags.
    Empty must_have_concepts is accepted (correct for a cluster's generic tag).
    Fallback to heuristic only on real failure.
    """
    if config.judge_mode == "heuristic":
        return _heuristic_boundary_policy(cluster_id, cluster_tags)

    try:
        from openai import OpenAI
    except Exception as exc:
        print(f"[stage3] OpenAI client unavailable for boundary policy ({exc}); falling back to heuristic")
        return _heuristic_boundary_policy(cluster_id, cluster_tags)

    client = OpenAI()
    prompt = build_boundary_policy_prompt(cluster_id, list(cluster_tags), profiles, config)
    body = _responses_request_body(config, prompt, BoundaryPolicyResult.model_json_schema())
    try:
        response = client.responses.create(**body)
        policy = BoundaryPolicyResult.model_validate(json.loads(response.output_text))
    except Exception as exc:
        print(f"[stage3] boundary policy GPT call failed for cluster {cluster_id}: {exc}")
        return _heuristic_boundary_policy(cluster_id, cluster_tags)

    returned_tags = {r.tag for r in policy.rules}
    expected_tags = set(cluster_tags)
    if returned_tags != expected_tags:
        missing = expected_tags - returned_tags
        extra = returned_tags - expected_tags
        print(
            f"[stage3] boundary policy tag-set mismatch for {cluster_id} "
            f"(missing={sorted(missing)} extra={sorted(extra)}); falling back to heuristic"
        )
        return _heuristic_boundary_policy(cluster_id, cluster_tags)
    return policy


def build_packet_prompt(packet_id: str, packet: pd.DataFrame, profiles: dict[str, dict[str, Any]], config: CleanerConfig) -> str:
    tags_in_packet = list(dict.fromkeys(packet["canonical_tag"].tolist()))
    competing: list[str] = []
    for _, row in packet.iterrows():
        competing.extend(
            _candidate_tags_from_embedding_evidence(
                row,
                profiles,
                exclude_tags=set(tags_in_packet),
                limit=config.top_k_competing_tags,
            )
        )
        competing.extend([row["e5_top1_competing_tag"], row["gemma_top1_competing_tag"]])
    competing_tags = [t for t in dict.fromkeys(competing) if t in profiles and t not in tags_in_packet]

    def tag_context(tag: str) -> dict[str, Any]:
        profile = profiles.get(tag, {})
        return {
            "tag": tag,
            "description": profile.get("description"),
            "central_examples": _compact_examples(profile.get("central_questions"), config.central_examples),
            "discriminative_phrases": _compact_examples(profile.get("discriminative_phrases"), 12),
        }

    payload = {
        "task": (
            "Judge each target question independently. Decide whether it is a clean, self-contained "
            "example for its current tag. Do not relabel rows. If uncertain, jettison."
        ),
        "packet_id": packet_id,
        "policy": {
            "keep": "Question clearly belongs to its current tag more than competitors.",
            "jettison": "Use for ambiguous, generic, context-dependent, wrong-intent, sibling-collision, duplicate-like, or synthetic rows.",
            "merge_candidate": "Use only if the row strongly suggests two tag clusters may be indistinguishable.",
            "no_row_swaps": True,
        },
        "current_tag_contexts": [tag_context(tag) for tag in tags_in_packet],
        "competing_tag_contexts": [tag_context(tag) for tag in competing_tags[: max(config.top_k_competing_tags * len(tags_in_packet), 3)]],
        "target_rows": [
            {
                "row_id": int(row["row_id"]),
                "question_raw": row["question_raw"],
                "question_normalized": row["question_norm"],
                "current_tag": row["canonical_tag"],
                "e5_competing_tag": row["e5_top1_competing_tag"],
                "gemma_competing_tag": row["gemma_top1_competing_tag"],
                "embedding_reconciliation": _json_cell(row.get("embedding_reconciliation"), {}),
                "e5_top10_neighbors": _json_cell(row.get("e5_top10_evidence"), []),
                "gemma_top10_neighbors": _json_cell(row.get("gemma_top10_evidence"), []),
            }
            for _, row in packet.iterrows()
        ],
        "required_json_schema": JudgePacketResult.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _responses_request_body(config: CleanerConfig, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.openai_model,
        "reasoning": {"effort": config.openai_reasoning_effort},
        "input": [
            {
                "role": "system",
                "content": "You are a strict dataset cleaning judge. Return only schema-valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "dataset_cleaning_judge",
                "strict": True,
                "schema": schema,
            },
        },
    }


def prepare_row_judge_batch(config: CleanerConfig, feature_df: pd.DataFrame, profiles: dict[str, dict[str, Any]], stage_dir: Path) -> Path:
    rows = []
    for _, row in feature_df[feature_df["route"] == "judge"].iterrows():
        for pass_idx in range(config.self_consistency_passes):
            prompt = build_judge_prompt(row, profiles, config)
            rows.append(
                {
                    "custom_id": f"judge:{int(row['row_id'])}:pass:{pass_idx}",
                    "method": "POST",
                    "url": config.openai_batch_endpoint,
                    "body": _responses_request_body(config, prompt, JudgeResult.model_json_schema()),
                }
            )
    path = stage_dir / "judge_requests.jsonl"
    write_jsonl(path, rows)
    return path


def prepare_packet_judge_batch(config: CleanerConfig, feature_df: pd.DataFrame, profiles: dict[str, dict[str, Any]], stage_dir: Path) -> Path:
    rows = []
    packets = build_judge_packets(feature_df, config)
    manifest_rows = []
    for packet_idx, packet in enumerate(packets):
        packet_id = f"packet:{packet_idx:06d}"
        row_ids = [int(v) for v in packet["row_id"].tolist()]
        manifest_rows.append(
            {
                "packet_id": packet_id,
                "row_ids": row_ids,
                "tags": list(dict.fromkeys(packet["canonical_tag"].tolist())),
            }
        )
        for pass_idx in range(config.self_consistency_passes):
            prompt = build_packet_prompt(packet_id, packet, profiles, config)
            rows.append(
                {
                    "custom_id": f"{packet_id}:pass:{pass_idx}",
                    "method": "POST",
                    "url": config.openai_batch_endpoint,
                    "body": _responses_request_body(config, prompt, JudgePacketResult.model_json_schema()),
                }
            )
    write_jsonl(stage_dir / "judge_packet_manifest.jsonl", manifest_rows)
    path = stage_dir / "judge_requests.jsonl"
    write_jsonl(path, rows)
    return path


def prepare_judge_batch(config: CleanerConfig, feature_df: pd.DataFrame, profiles: dict[str, dict[str, Any]], stage_dir: Path) -> Path:
    if config.judge_granularity == "row":
        return prepare_row_judge_batch(config, feature_df, profiles, stage_dir)
    if config.judge_granularity != "tag_batch":
        raise ValueError(f"Unsupported judge_granularity: {config.judge_granularity}")
    return prepare_packet_judge_batch(config, feature_df, profiles, stage_dir)


def parse_responses_json_response(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("output_text"):
        return json.loads(body["output_text"])
    if "output" in body:
        for item in body["output"]:
            if item.get("type") == "message":
                parts = item.get("content") or []
                for part in parts:
                    if part.get("type") in {"output_text", "text"}:
                        return json.loads(part.get("text", ""))
    if "choices" in body:
        # Backward compatibility for any older chat-completions batch output.
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)
    raise ValueError("Could not parse OpenAI response JSON")


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
)
async def _responses_judge_one(client: Any, config: CleanerConfig, prompt: str) -> JudgeResult:
    body = _responses_request_body(config, prompt, JudgeResult.model_json_schema())
    response = await asyncio.to_thread(client.responses.create, **body)
    payload = json.loads(response.output_text)
    return JudgeResult.model_validate(payload)


async def run_responses_judge(config: CleanerConfig, feature_df: pd.DataFrame, profiles: dict[str, dict[str, Any]], stage_dir: Path) -> None:
    from openai import OpenAI

    client = OpenAI()
    judge_rows = feature_df[feature_df["route"] == "judge"].copy()
    expected_row_ids = [int(v) for v in judge_rows["row_id"].tolist()]
    if config.judge_granularity == "row":
        for _, row in judge_rows.iterrows():
            row_id = int(row["row_id"])
            for pass_idx in range(config.self_consistency_passes):
                out_path = row_result_path(stage_dir, row_id, pass_idx)
                if not out_path.exists():
                    prompt = build_judge_prompt(row, profiles, config)
                    result = await _responses_judge_one(client, config, prompt)
                    write_json(out_path, {"row_id": row_id, "pass_idx": pass_idx, "result": result.model_dump()})
                aggregate_incremental_judge_results(stage_dir, expected_row_ids)
    elif config.judge_granularity == "tag_batch":
        async def _judge_one(prompt: str) -> JudgeResult:
            return await _responses_judge_one(client, config, prompt)

        for packet_idx, packet in enumerate(build_judge_packets(feature_df, config)):
            packet_id = f"packet:{packet_idx:06d}"
            for pass_idx in range(config.self_consistency_passes):
                out_path = packet_result_path(stage_dir, packet_id, pass_idx)
                if not out_path.exists():
                    prompt = build_packet_prompt(packet_id, packet, profiles, config)
                    body = _responses_request_body(config, prompt, JudgePacketResult.model_json_schema())
                    response = await asyncio.to_thread(client.responses.create, **body)
                    packet_result = JudgePacketResult.model_validate(json.loads(response.output_text))
                    write_json(
                        out_path,
                        {
                            "packet_id": packet_id,
                            "pass_idx": pass_idx,
                            "row_ids": [int(v) for v in packet["row_id"].tolist()],
                            "result": packet_result.model_dump(),
                        },
                    )
                else:
                    payload = read_json(out_path)
                    packet_result = JudgePacketResult.model_validate(payload["result"])
                await _fill_packet_misses(config, stage_dir, packet, packet_result, pass_idx, profiles, _judge_one)
                aggregate_incremental_judge_results(stage_dir, expected_row_ids)
    else:
        raise ValueError(f"Unsupported judge_granularity: {config.judge_granularity}")
    aggregate_incremental_judge_results(stage_dir, expected_row_ids)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
)
async def _agents_judge_one(config: CleanerConfig, prompt: str) -> JudgeResult:
    from agents import Agent, ModelSettings, Runner
    from openai.types.shared import Reasoning

    agent = Agent(
        name="dataset cleaning judge",
        instructions="You are a strict dataset cleaning judge. Return only the structured result.",
        model=config.openai_model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=config.openai_reasoning_effort),
            verbosity="low",
        ),
        output_type=JudgeResult,
    )
    result = await asyncio.to_thread(Runner.run_sync, agent, prompt)
    return result.final_output


async def run_agents_judge(config: CleanerConfig, feature_df: pd.DataFrame, profiles: dict[str, dict[str, Any]], stage_dir: Path) -> None:
    judge_rows = feature_df[feature_df["route"] == "judge"].copy()
    expected_row_ids = [int(v) for v in judge_rows["row_id"].tolist()]
    if config.judge_granularity == "row":
        for _, row in judge_rows.iterrows():
            row_id = int(row["row_id"])
            for pass_idx in range(config.self_consistency_passes):
                out_path = row_result_path(stage_dir, row_id, pass_idx)
                if not out_path.exists():
                    prompt = build_judge_prompt(row, profiles, config)
                    result = await _agents_judge_one(config, prompt)
                    write_json(out_path, {"row_id": row_id, "pass_idx": pass_idx, "result": result.model_dump()})
                aggregate_incremental_judge_results(stage_dir, expected_row_ids)
    elif config.judge_granularity == "tag_batch":
        async def _judge_one(prompt: str) -> JudgeResult:
            return await _agents_judge_one(config, prompt)

        for packet_idx, packet in enumerate(build_judge_packets(feature_df, config)):
            packet_id = f"packet:{packet_idx:06d}"
            for pass_idx in range(config.self_consistency_passes):
                out_path = packet_result_path(stage_dir, packet_id, pass_idx)
                if not out_path.exists():
                    from agents import Agent, ModelSettings, Runner
                    from openai.types.shared import Reasoning

                    prompt = build_packet_prompt(packet_id, packet, profiles, config)
                    agent = Agent(
                        name="dataset packet cleaning judge",
                        instructions="You are a strict dataset cleaning judge. Return only the structured packet result.",
                        model=config.openai_model,
                        model_settings=ModelSettings(
                            reasoning=Reasoning(effort=config.openai_reasoning_effort),
                            verbosity="low",
                        ),
                        output_type=JudgePacketResult,
                    )
                    packet_result = await asyncio.to_thread(Runner.run_sync, agent, prompt)
                    parsed: JudgePacketResult = packet_result.final_output
                    write_json(
                        out_path,
                        {
                            "packet_id": packet_id,
                            "pass_idx": pass_idx,
                            "row_ids": [int(v) for v in packet["row_id"].tolist()],
                            "result": parsed.model_dump(),
                        },
                    )
                else:
                    payload = read_json(out_path)
                    parsed = JudgePacketResult.model_validate(payload["result"])
                await _fill_packet_misses(config, stage_dir, packet, parsed, pass_idx, profiles, _judge_one)
                aggregate_incremental_judge_results(stage_dir, expected_row_ids)
    else:
        raise ValueError(f"Unsupported judge_granularity: {config.judge_granularity}")
    aggregate_incremental_judge_results(stage_dir, expected_row_ids)


def _resolve_consistency(row_id: int, passes: list[JudgeResult]) -> dict[str, Any]:
    if not passes:
        return {
            "row_id": row_id,
            "decision": "jettison",
            "quality_score": 0,
            "ambiguity_score": 100,
            "context_dependent": False,
            "reason_code": "synthetic_artifact",
            "rationale": "No judge result was available.",
            "consistent": False,
        }
    first = passes[0]
    consistent = all(
        p.decision == first.decision
        and p.context_dependent == first.context_dependent
        and p.reason_code == first.reason_code
        for p in passes[1:]
    )
    if not consistent:
        return {
            "row_id": row_id,
            "decision": "jettison",
            "quality_score": 0,
            "ambiguity_score": 100,
            "context_dependent": any(p.context_dependent for p in passes),
            "reason_code": "sibling_collision",
            "rationale": "Judge passes disagreed; strict automation jettisoned the row.",
            "consistent": False,
        }
    payload = first.model_dump()
    payload["row_id"] = row_id
    payload["consistent"] = True
    return payload


def collect_batch_results(batch_output_path: Path, stage_dir: Path) -> None:
    grouped: defaultdict[int, list[JudgeResult]] = defaultdict(list)
    for line in read_jsonl(batch_output_path):
        custom_id = line.get("custom_id", "")
        if line.get("error"):
            continue
        body = (line.get("response") or {}).get("body") or {}
        try:
            payload = parse_responses_json_response(body)
            row_match = re.match(r"judge:(\d+):pass:(\d+)", custom_id)
            packet_match = re.match(r"packet:\d+:pass:(\d+)", custom_id)
            if row_match:
                row_id = int(row_match.group(1))
                grouped[row_id].append(JudgeResult.model_validate(payload))
            elif packet_match:
                parsed = JudgePacketResult.model_validate(payload)
                for decision in parsed.decisions:
                    grouped[decision.row_id].append(
                        JudgeResult(
                            decision=decision.decision,
                            quality_score=decision.quality_score,
                            ambiguity_score=decision.ambiguity_score,
                            context_dependent=decision.context_dependent,
                            reason_code=decision.reason_code,
                            rationale=decision.rationale,
                        )
                    )
        except (ValidationError, ValueError, json.JSONDecodeError):
            continue
    write_jsonl(stage_dir / "judge_results.jsonl", [_resolve_consistency(row_id, passes) for row_id, passes in grouped.items()])


def fetch_batch_output(stage_dir: Path) -> Path:
    """Fetch completed OpenAI Batch output recorded by a previous batch_submit run."""
    from openai import OpenAI

    batch_info = read_json(stage_dir / "batch.json")
    if not batch_info or not batch_info.get("id"):
        raise ValueError("No batch.json found. Provide --batch-output or run stage5 with judge_mode=batch_submit first.")

    client = OpenAI()
    batch = client.batches.retrieve(batch_info["id"])
    status = getattr(batch, "status", None)
    if status != "completed":
        raise RuntimeError(f"OpenAI batch {batch_info['id']} is not completed yet (status={status}).")
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(f"OpenAI batch {batch_info['id']} completed without output_file_id.")

    content = client.files.content(output_file_id)
    raw = content.read()
    if isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    else:
        raw_bytes = raw
    output_path = stage_dir / "judge_batch_output.jsonl"
    output_path.write_bytes(raw_bytes)
    return output_path


def build_audit_packets(feature_df: pd.DataFrame, config: CleanerConfig) -> list[pd.DataFrame]:
    """Group audit_buffer rows into per-tag packets of `audit_rows_per_packet`."""
    audit = feature_df[feature_df["audit_buffer"]].copy()
    if audit.empty:
        return []
    audit = audit.sort_values(
        ["canonical_tag", "composite_score", "row_id"], ascending=[True, False, True]
    )
    packets: list[pd.DataFrame] = []
    for _, group in audit.groupby("canonical_tag", sort=True):
        for start in range(0, len(group), config.audit_rows_per_packet):
            packets.append(group.iloc[start : start + config.audit_rows_per_packet].reset_index(drop=True))
    return packets


def build_audit_packet_prompt(
    packet_id: str,
    packet: pd.DataFrame,
    tag_to_rule: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    config: CleanerConfig,
) -> str:
    tag = packet["canonical_tag"].iloc[0]
    rule = tag_to_rule.get(tag, {})
    profile = profiles.get(tag, {})
    payload = {
        "task": (
            "For each target_row, return one decision: 'keep' if the row is a clean "
            "example of THIS tag, or 'flag' if it should be dropped from this tag. "
            "Do not relabel — this is a single-tag pass/reject audit."
        ),
        "decision_spec": {
            "keep": "Question is self-contained and clearly fits the tag's intent; "
                "no sibling-tag collision; not too generic; not synthetic.",
            "flag_reasons": {
                "wrong_intent": "Row's intent doesn't match this tag at all.",
                "sibling_collision": "Row fits a sibling/competing tag better.",
                "too_generic": "Row is generic enough to fit several tags.",
                "duplicate": "Row is a near-paraphrase of another row in the set.",
                "synthetic_artifact": "Awkward repetition, malformed, looks GPT-generated.",
                "context_dependent": "Row needs prior conversation context to make sense.",
            },
            "if_unsure": "Prefer flag.",
        },
        "packet_id": packet_id,
        "tag": tag,
        "tag_one_line_intent": rule.get("one_line_intent", ""),
        "must_have_concepts": rule.get("must_have", []),
        "must_avoid_concepts": rule.get("must_avoid", []),
        "tag_description": profile.get("description"),
        "tag_central_examples": _compact_examples(profile.get("central_questions"), config.central_examples),
        "target_rows": [
            {
                "row_id": int(row["row_id"]),
                "question_raw": row["question_raw"],
                "question_normalized": row["question_norm"],
            }
            for _, row in packet.iterrows()
        ],
        "output_contract": "Return ONE AuditRowDecision per target_row, in any order. "
            "Every target row_id must appear exactly once in decisions.",
        "required_json_schema": AuditPacketResult.model_json_schema(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
)
async def _responses_audit_one(client: Any, config: CleanerConfig, prompt: str) -> AuditPacketResult:
    body = _responses_request_body(config, prompt, AuditPacketResult.model_json_schema())
    response = await asyncio.to_thread(client.responses.create, **body)
    return AuditPacketResult.model_validate(json.loads(response.output_text))


def _build_audit_agent(config: CleanerConfig) -> Any:
    """Construct the audit Agent once per Stage 5 run; reuse across packets."""
    from agents import Agent, ModelSettings
    from openai.types.shared import Reasoning

    return Agent(
        name="tag-audit agent",
        instructions=(
            "You audit candidate rows for a single tag using its boundary "
            "rules. Return only schema-valid AuditPacketResult JSON."
        ),
        model=config.openai_model,
        model_settings=ModelSettings(
            reasoning=Reasoning(effort=config.openai_reasoning_effort),
            verbosity="low",
        ),
        output_type=AuditPacketResult,
    )


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
)
async def _agents_audit_one(agent: Any, prompt: str) -> AuditPacketResult:
    """Native async Runner.run; no asyncio.to_thread wrapper.

    Lets `asyncio.gather` actually run packets concurrently.
    """
    from agents import Runner

    result = await Runner.run(agent, prompt)
    return result.final_output


def aggregate_audit_results(stage_dir: Path, expected_row_ids: Iterable[int]) -> None:
    """Merge per-packet audit results into a flat per-row JSONL.

    Rows missing from any packet response default to keep with audit_status=missing —
    the deterministic ranker already endorsed them; absent audit signal is not evidence
    against the row.
    """
    expected = {int(r) for r in expected_row_ids}
    decisions: dict[int, dict[str, Any]] = {}
    for path in sorted((stage_dir / "audit_packets").glob("packet_*_pass_*.json")):
        payload = read_json(path)
        if not payload:
            continue
        result = AuditPacketResult.model_validate(payload["result"])
        for d in result.decisions:
            if d.row_id in expected and d.row_id not in decisions:
                decisions[d.row_id] = {**d.model_dump(), "audit_status": "audited"}
    rows = []
    for row_id in sorted(expected):
        if row_id in decisions:
            rows.append(decisions[row_id])
        else:
            rows.append({
                "row_id": row_id,
                "decision": "keep",
                "reason_code": "clean",
                "rationale": "No audit response; ranker-trusted default.",
                "audit_status": "missing",
            })
    write_jsonl(stage_dir / "audit_results.jsonl", rows)


async def _run_audit_loop(
    config: CleanerConfig,
    feature_df: pd.DataFrame,
    tag_to_rule: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    stage_dir: Path,
    audit_one: Any,
    trace_label: str | None = None,
) -> None:
    """Run audit packets concurrently with bounded semaphore.

    `audit_one` is an async callable taking a prompt and returning
    AuditPacketResult. `trace_label`, when set, wraps the gather in
    `agents.trace(label, group_id=run_id)` for observability.

    Cached packets (out_path exists) are skipped without an API call.
    Atomic write: tmp-file + rename.
    """
    audit = feature_df[feature_df["audit_buffer"]].copy()
    expected_row_ids = [int(v) for v in audit["row_id"].tolist()]
    packets = build_audit_packets(feature_df, config)
    (stage_dir / "audit_packets").mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max(1, int(config.concurrency)))
    state: dict[str, int] = {"ok": 0, "skipped": 0, "failed": 0}

    async def process_packet(packet_idx: int, packet: pd.DataFrame) -> None:
        packet_id = f"audit_packet:{packet_idx:06d}"
        out_path = stage_dir / "audit_packets" / f"packet_{packet_idx:06d}_pass_0.json"
        if out_path.exists():
            state["skipped"] += 1
            return
        prompt = build_audit_packet_prompt(packet_id, packet, tag_to_rule, profiles, config)
        async with semaphore:
            try:
                result = await audit_one(prompt)
            except Exception as exc:
                print(f"[stage5] audit packet {packet_id} failed after retries: {exc}")
                state["failed"] += 1
                return
        # Atomic write: tmp + rename so concurrent readers never see a partial file.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        write_json(
            tmp_path,
            {
                "packet_id": packet_id,
                "pass_idx": 0,
                "row_ids": [int(v) for v in packet["row_id"].tolist()],
                "result": result.model_dump(),
            },
        )
        tmp_path.replace(out_path)
        state["ok"] += 1
        # Show progress as packets land (order arbitrary under concurrency).
        completed = state["ok"] + state["skipped"] + state["failed"]
        print(f"[stage5] audit progress: {completed}/{len(packets)} packets")

    cm: Any = None
    if trace_label:
        try:
            from agents import trace  # type: ignore

            cm = trace(trace_label, group_id=config.resolved_run_id())
        except Exception:
            cm = None

    if cm is not None:
        with cm:
            await asyncio.gather(*(process_packet(i, p) for i, p in enumerate(packets)))
    else:
        await asyncio.gather(*(process_packet(i, p) for i, p in enumerate(packets)))

    print(
        f"[stage5] audit done: {state['ok']} ok, {state['skipped']} cached, "
        f"{state['failed']} failed; concurrency={config.concurrency}"
    )
    aggregate_audit_results(stage_dir, expected_row_ids)


def _heuristic_audit(feature_df: pd.DataFrame, stage_dir: Path) -> None:
    audit = feature_df[feature_df["audit_buffer"]].copy()
    rows = []
    for _, row in audit.iterrows():
        decision = "flag" if (row["cross_tag_duplicate"] or row["near_dup_count"] > 0 or row["artifact_score"] > 0.5) else "keep"
        rows.append({
            "row_id": int(row["row_id"]),
            "decision": decision,
            "reason_code": "clean" if decision == "keep" else ("duplicate" if row["near_dup_count"] > 0 else "synthetic_artifact"),
            "rationale": "Heuristic audit fallback (no GPT).",
            "audit_status": "heuristic",
        })
    write_jsonl(stage_dir / "audit_results.jsonl", rows)


def run_stage5(config: CleanerConfig, resume: bool = True, batch_output: Path | None = None) -> None:
    run_stage4(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage5"
    feature_df = pd.read_parquet(run_dir / "stage4" / "row_features.parquet")
    profiles = {
        row["tag"]: row
        for row in pd.read_parquet(run_dir / "stage2" / "tag_profile.parquet").to_dict(orient="records")
    }
    policy_records = read_jsonl(run_dir / "stage3" / "tag_boundary_policy.jsonl")
    tag_to_rule: dict[str, dict[str, Any]] = {}
    for cluster in policy_records:
        for rule in cluster.get("rules", []) or []:
            tag = rule.get("tag")
            if not tag:
                continue
            tag_to_rule[tag] = {
                "one_line_intent": rule.get("one_line_intent", ""),
                "must_have": [str(c) for c in rule.get("must_have_concepts") or []],
                "must_avoid": [str(c) for c in rule.get("must_avoid_concepts") or []],
            }

    input_hash = dataframe_hash(
        feature_df, ["row_id", "audit_buffer", "canonical_tag", "target_scope", "composite_score"],
    )
    if resume and stage_done(stage_dir, input_hash) and (stage_dir / "audit_results.jsonl").exists():
        print(f"[stage5] skip: {stage_dir}")
        return

    stage_dir.mkdir(parents=True, exist_ok=True)
    if config.judge_mode == "heuristic":
        _heuristic_audit(feature_df, stage_dir)
    elif config.judge_mode == "sync":
        from openai import OpenAI

        client = OpenAI()

        async def _audit_one(prompt: str) -> AuditPacketResult:
            return await _responses_audit_one(client, config, prompt)

        asyncio.run(_run_audit_loop(config, feature_df, tag_to_rule, profiles, stage_dir, _audit_one))
    elif config.judge_mode == "agents":
        agent = _build_audit_agent(config)

        async def _audit_one(prompt: str) -> AuditPacketResult:
            return await _agents_audit_one(agent, prompt)

        asyncio.run(
            _run_audit_loop(
                config, feature_df, tag_to_rule, profiles, stage_dir, _audit_one,
                trace_label="tagclean.audit",
            )
        )
    elif config.judge_mode in {"batch_prepare", "batch_submit", "batch_collect"}:
        raise NotImplementedError(
            "Audit-mode batch path is not yet wired. Use judge_mode=sync or agents for now."
        )
    else:
        raise ValueError(f"Unsupported judge_mode: {config.judge_mode}")

    finish_stage(stage_dir, config, input_hash, {"judge_mode": config.judge_mode, "audit_buffer_rows": int(feature_df["audit_buffer"].sum())})
    print(f"[stage5] wrote {stage_dir}")


def _load_judge_results(run_dir: Path) -> pd.DataFrame:
    rows = read_jsonl(run_dir / "stage5" / "judge_results.jsonl")
    if not rows:
        return pd.DataFrame(columns=["row_id", "decision", "quality_score", "ambiguity_score", "reason_code", "rationale"])
    return pd.DataFrame(rows)


def _load_stage_judge_results(stage_dir: Path) -> pd.DataFrame:
    rows = read_jsonl(stage_dir / "judge_results.jsonl")
    if not rows:
        return pd.DataFrame(columns=["row_id", "decision", "quality_score", "ambiguity_score", "reason_code", "rationale"])
    return pd.DataFrame(rows)


def _normalize_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def _mmr_order(group: pd.DataFrame, vectors: np.ndarray, lambda_: float = 0.7) -> dict[int, float]:
    ids = group.index.to_list()
    if not ids:
        return {}
    selected: list[int] = []
    scores: dict[int, float] = {}
    base = group["base_score"].to_dict()
    while ids:
        best_id = None
        best_score = -float("inf")
        for idx in ids:
            if selected:
                redundancy = max(float(vectors[idx] @ vectors[j]) for j in selected)
            else:
                redundancy = 0.0
            score = lambda_ * float(base[idx]) - (1 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_id = idx
        assert best_id is not None
        selected.append(best_id)
        ids.remove(best_id)
        scores[best_id] = max(0.0, best_score)
    if scores:
        vals = pd.Series(scores)
        return _normalize_series(vals).to_dict()
    return scores


def run_stage6(config: CleanerConfig, resume: bool = True) -> None:
    run_stage5(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage6"
    feature_df = pd.read_parquet(run_dir / "stage4" / "row_features.parquet")
    audit_rows = read_jsonl(run_dir / "stage5" / "audit_results.jsonl")
    audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame(
        columns=["row_id", "decision", "reason_code", "rationale", "audit_status"],
    )
    input_hash = dataframe_hash(
        feature_df, ["row_id", "audit_buffer", "canonical_tag", "target_scope", "composite_score"],
    )
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage6] skip: {stage_dir}")
        return

    df = feature_df.merge(
        audit_df[["row_id", "decision", "reason_code", "rationale", "audit_status"]] if not audit_df.empty else audit_df,
        on="row_id",
        how="left",
    )
    df = df[df["target_scope"]].copy()
    if "decision" not in df.columns:
        df["decision"] = pd.Series([None] * len(df), index=df.index, dtype="object")
        df["reason_code"] = ""
        df["rationale"] = ""
        df["audit_status"] = ""

    # Status: keep iff in audit buffer AND audit decision == "keep". Below-buffer
    # rows and audit-flagged rows are jettisoned.
    df["status"] = "jettison"
    df["status_reason"] = ""
    in_buffer = df["audit_buffer"]
    df.loc[in_buffer & (df["decision"] == "keep"), "status"] = "keep"
    df.loc[in_buffer & (df["decision"] == "keep"), "status_reason"] = "audited_pass"
    df.loc[in_buffer & (df["decision"] == "flag"), "status_reason"] = "audited_flag"
    df.loc[~in_buffer, "status_reason"] = "below_audit_buffer"

    kept = df[df["status"] == "keep"].copy()
    jettisoned = df[df["status"] == "jettison"].copy()

    # MMR-adjusted ranking inside each tag's kept set so 40 rows cover phrasing
    # diversity, not 40 near-paraphrases of the medoid.
    if not kept.empty:
        kept["base_score"] = kept["composite_score"]
        emb_e5 = np.load(run_dir / "stage1" / "emb_e5.npy")
        emb_rows = pd.read_parquet(run_dir / "stage1" / "embedding_rows.parquet")
        row_pos = {int(rid): i for i, rid in enumerate(emb_rows["row_id"])}
        kept["_emb_pos"] = kept["row_id"].astype(int).map(row_pos)
        mmr_scores: dict[int, float] = {}
        for _, group in kept.groupby("canonical_tag"):
            indexed = group.set_index("_emb_pos")
            mmr_scores.update(_mmr_order(indexed, emb_e5, lambda_=0.7))
        kept["mmr_score"] = kept["_emb_pos"].map(mmr_scores).fillna(0.0)
        kept["final_score"] = 0.85 * kept["composite_score"] + 0.15 * kept["mmr_score"]
        kept = kept.drop(columns=["_emb_pos"]).sort_values(
            ["canonical_tag", "final_score"], ascending=[True, False]
        )
        kept["rank"] = kept.groupby("canonical_tag").cumcount() + 1
        kept["production_recommended"] = kept["rank"] <= config.top_n
    else:
        kept["final_score"] = []
        kept["rank"] = []
        kept["production_recommended"] = []

    stage_dir.mkdir(parents=True, exist_ok=True)
    cleaned_out_cols = [
        "question_raw", "canonical_tag", "rank", "final_score", "composite_score",
        "production_recommended", "audit_status", "reason_code", "row_id", "tag",
        "e5_margin", "gemma_margin", "token_alignment", "artifact_score", "rationale",
    ]
    cleaned_out_cols = [c for c in cleaned_out_cols if c in kept.columns]
    kept[cleaned_out_cols].rename(
        columns={"question_raw": "question", "canonical_tag": "tag_clean", "tag": "original_tag"}
    ).to_csv(stage_dir / "question_tag.cleaned.csv", index=False)

    top = kept[kept["production_recommended"]] if "production_recommended" in kept.columns else kept.iloc[0:0]
    top[["question_raw", "canonical_tag"]].rename(
        columns={"question_raw": "question", "canonical_tag": "tag"}
    ).to_csv(stage_dir / "question_tag.top40.csv", index=False)

    jett_cols = [
        "question_raw", "canonical_tag", "row_id", "tag", "status_reason",
        "audit_status", "reason_code", "rationale", "composite_score",
        "audit_buffer", "e5_margin", "gemma_margin", "token_alignment", "artifact_score",
    ]
    jett_cols = [c for c in jett_cols if c in jettisoned.columns]
    jettisoned[jett_cols].to_csv(stage_dir / "jettisoned_rows.csv", index=False)
    finish_stage(
        stage_dir,
        config,
        input_hash,
        {
            "kept": len(kept),
            "jettisoned": len(jettisoned),
            "production_recommended": int(kept["production_recommended"].sum()) if "production_recommended" in kept.columns else 0,
        },
    )
    print(f"[stage6] wrote {stage_dir}")


def select_review_candidates(cleaned: pd.DataFrame, config: CleanerConfig) -> list[int]:
    if cleaned.empty or not config.review_enabled:
        return []
    selected: set[int] = set()
    for _, group in cleaned.groupby("tag_clean"):
        threshold = group["composite_score"].quantile(config.review_low_score_quantile)
        tail = group[group["composite_score"] <= threshold]
        tail = tail.nsmallest(config.review_rows_per_tag, "composite_score")
        selected.update(int(v) for v in tail["row_id"].tolist())
    return sorted(selected)


def apply_review_results(cleaned: pd.DataFrame, review_results: pd.DataFrame, config: CleanerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if review_results.empty:
        reviewed = cleaned.copy()
        reviewed["review_decision"] = "not_reviewed"
        reviewed["review_reason_code"] = ""
        reviewed["review_rationale"] = ""
        return reviewed, cleaned.iloc[0:0].copy()

    results = review_results[["row_id", "decision", "reason_code", "rationale", "quality_score", "ambiguity_score"]].rename(
        columns={
            "decision": "review_decision",
            "reason_code": "review_reason_code",
            "rationale": "review_rationale",
            "quality_score": "review_quality_score",
            "ambiguity_score": "review_ambiguity_score",
        }
    )
    merged = cleaned.merge(results, on="row_id", how="left")
    merged["review_decision"] = merged["review_decision"].fillna("not_reviewed")
    reviewed_keep = merged[merged["review_decision"].isin(["keep", "not_reviewed"])].copy()
    reviewed_drop = merged[~merged["review_decision"].isin(["keep", "not_reviewed"])].copy()

    reviewed_keep = reviewed_keep.sort_values(["tag_clean", "composite_score"], ascending=[True, False])
    reviewed_keep["rank"] = reviewed_keep.groupby("tag_clean").cumcount() + 1
    reviewed_keep["production_recommended"] = reviewed_keep["rank"] <= config.top_n
    return reviewed_keep, reviewed_drop


def run_stage7(config: CleanerConfig, resume: bool = True) -> None:
    """Second automated review pass over low-confidence kept rows."""
    run_stage6(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage7"
    cleaned_path = run_dir / "stage6" / "question_tag.cleaned.csv"
    input_hash = hashlib.sha256(
        file_sha256(cleaned_path).encode("utf-8")
        + json.dumps(
            {
                "review_enabled": config.review_enabled,
                "review_rows_per_tag": config.review_rows_per_tag,
                "review_low_score_quantile": config.review_low_score_quantile,
                "judge_mode": config.judge_mode,
                "judge_granularity": config.judge_granularity,
                "passes": config.self_consistency_passes,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage7] skip: {stage_dir}")
        return

    stage_dir.mkdir(parents=True, exist_ok=True)
    cleaned = pd.read_csv(cleaned_path)
    candidate_ids = select_review_candidates(cleaned, config)
    write_json(stage_dir / "review_candidates.json", {"row_ids": candidate_ids, "count": len(candidate_ids)})

    if not candidate_ids:
        reviewed = cleaned.copy()
        reviewed["review_decision"] = "not_reviewed"
        reviewed["review_reason_code"] = ""
        reviewed["review_rationale"] = ""
        reviewed_drop = cleaned.iloc[0:0].copy()
    else:
        feature_df = pd.read_parquet(run_dir / "stage4" / "row_features.parquet")
        review_features = feature_df[feature_df["row_id"].isin(candidate_ids)].copy()
        review_features["route"] = "judge"
        profiles = {
            row["tag"]: row
            for row in pd.read_parquet(run_dir / "stage2" / "tag_profile.parquet").to_dict(orient="records")
        }

        if config.judge_mode == "heuristic":
            rows = []
            for _, row in review_features.iterrows():
                decision = "jettison" if row["cross_tag_duplicate"] or row["e5_margin"] < 0 or row["gemma_margin"] < 0 else "keep"
                rows.append(
                    {
                        "row_id": int(row["row_id"]),
                        "decision": decision,
                        "quality_score": 70 if decision == "keep" else 0,
                        "ambiguity_score": 35 if decision == "keep" else 100,
                        "context_dependent": False,
                        "reason_code": "clean" if decision == "keep" else "sibling_collision",
                        "rationale": "Heuristic second-pass review fallback.",
                        "consistent": True,
                    }
                )
                write_jsonl(stage_dir / "judge_results.jsonl", rows)
        elif config.judge_mode == "sync":
            asyncio.run(run_responses_judge(config, review_features, profiles, stage_dir))
        elif config.judge_mode == "agents":
            asyncio.run(run_agents_judge(config, review_features, profiles, stage_dir))
        elif config.judge_mode in {"batch_prepare", "batch_submit", "batch_collect"}:
            request_path = prepare_judge_batch(config, review_features, profiles, stage_dir)
            raise RuntimeError(
                f"Second-pass review is configured for async batch mode. Prepared {request_path}; "
                "collect it, then rerun stage7 with judge_mode=batch_collect or switch to sync/agents for sequential execution."
            )
        else:
            raise ValueError(f"Unsupported judge_mode: {config.judge_mode}")

        review_df = _load_stage_judge_results(stage_dir)
        reviewed, reviewed_drop = apply_review_results(cleaned, review_df, config)

    reviewed.to_csv(stage_dir / "question_tag.reviewed.csv", index=False)
    reviewed[reviewed["production_recommended"]][["question", "tag_clean"]].rename(columns={"tag_clean": "tag"}).to_csv(
        stage_dir / "question_tag.top40.reviewed.csv", index=False
    )
    reviewed_drop.to_csv(stage_dir / "jettisoned_review_rows.csv", index=False)
    finish_stage(stage_dir, config, input_hash, {"reviewed": len(reviewed), "review_jettisoned": len(reviewed_drop)})
    print(f"[stage7] wrote {stage_dir}")


def _faiss_search(vectors: np.ndarray, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index.search(query_vectors.astype(np.float32), top_k)


def _loo_metrics(df: pd.DataFrame, vectors: np.ndarray, query_vectors: np.ndarray, top_k: int) -> dict[str, Any]:
    search_k = min(top_k + 2, len(df))
    _, indices = _faiss_search(vectors, query_vectors, search_k)
    tags = df["tag"].tolist()
    row_ids = df["row_id"].astype(int).tolist()
    top1_ok = 0
    topk_ok = 0
    confusions: Counter[tuple[str, str]] = Counter()
    for i, neighbors in enumerate(indices):
        filtered = [n for n in neighbors if n >= 0 and row_ids[n] != row_ids[i]]
        if not filtered:
            continue
        true_tag = tags[i]
        pred_tag = tags[filtered[0]]
        if pred_tag == true_tag:
            top1_ok += 1
        else:
            confusions[(true_tag, pred_tag)] += 1
        if any(tags[n] == true_tag for n in filtered[:top_k]):
            topk_ok += 1
    total = len(df)
    return {
        "rows": total,
        "top1_accuracy": top1_ok / total if total else 0,
        f"top{top_k}_accuracy": topk_ok / total if total else 0,
        "confusions": [
            {"true_tag": a, "predicted_tag": b, "count": c}
            for (a, b), c in confusions.most_common(100)
        ],
    }


def run_stage8(config: CleanerConfig, resume: bool = True) -> None:
    run_stage7(config, resume=resume)
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage8"
    reviewed_path = run_dir / "stage7" / "question_tag.reviewed.csv"
    reviewed_top40_path = run_dir / "stage7" / "question_tag.top40.reviewed.csv"
    cleaned_path = reviewed_path if reviewed_path.exists() else run_dir / "stage6" / "question_tag.cleaned.csv"
    top40_path = reviewed_top40_path if reviewed_top40_path.exists() else run_dir / "stage6" / "question_tag.top40.csv"
    input_hash = hashlib.sha256(file_sha256(cleaned_path).encode("utf-8") + file_sha256(top40_path).encode("utf-8")).hexdigest()
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage8] skip: {stage_dir}")
        return

    cleaned = pd.read_csv(cleaned_path).rename(columns={"tag_clean": "tag"})
    top40 = pd.read_csv(top40_path)
    emb_rows = pd.read_parquet(run_dir / "stage1" / "embedding_rows.parquet")
    emb_e5 = np.load(run_dir / "stage1" / "emb_e5.npy")
    emb_e5_query = np.load(run_dir / "stage1" / "emb_e5_query.npy")
    row_pos = {int(row_id): i for i, row_id in enumerate(emb_rows["row_id"])}

    def attach_vectors(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        frame = frame.copy()
        if "row_id" not in frame.columns:
            # Top40 intentionally contains only question/tag. Map back for validation.
            frame = frame.merge(cleaned[["question", "row_id"]], on="question", how="left")
        frame = frame.dropna(subset=["row_id"]).copy()
        positions = [row_pos[int(row_id)] for row_id in frame["row_id"]]
        return frame.reset_index(drop=True), emb_e5[positions], emb_e5_query[positions]

    cleaned_df, cleaned_vecs, cleaned_queries = attach_vectors(cleaned)
    top_df, top_vecs, top_queries = attach_vectors(top40)

    report = {
        "cleaned": _loo_metrics(cleaned_df, cleaned_vecs, cleaned_queries, config.validation_top_k),
        "top40": _loo_metrics(top_df, top_vecs, top_queries, config.validation_top_k),
        "low_support_tags": (
            cleaned_df.groupby("tag")
            .size()
            .reset_index(name="surviving_rows")
            .loc[lambda frame: frame["surviving_rows"] < config.low_support_threshold]
            .to_dict(orient="records")
        ),
    }

    confusion_rows = report["cleaned"]["confusions"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    write_json(stage_dir / "cleaning_report.json", report)
    pd.DataFrame(confusion_rows).to_parquet(stage_dir / "confusion_matrix.parquet", index=False)
    finish_stage(stage_dir, config, input_hash, {"cleaned_rows": len(cleaned_df), "top40_rows": len(top_df)})
    print(f"[stage8] wrote {stage_dir}")


def run_stage9(config: CleanerConfig, resume: bool = True) -> None:
    """E5-only production-risk audit. Runs leave-one-out top-K retrieval over
    the cleaned set using E5 alone (no Gemma) — this mimics production, where
    inference is E5-only. Reports per-row neighborhood and, by default, drops
    rows whose nearest non-self neighbor belongs to a different tag.

    Input is by default `stage6/question_tag.cleaned.csv`. Pass
    `config.stage9_input_csv` to audit an external CSV — typically the
    concatenation of top-40 sets across multiple family runs (the actual
    production candidate set).

    Stage 9 NEVER chains Stage 0–8: chaining would silently overwrite the
    cleaned set under a different config (e.g. running stage9 with a
    different judge_mode would re-run stage4–6 with that config, clobbering
    the original cleaned.csv). The user must run Stage 8 (or earlier)
    explicitly first; Stage 9 only audits what's already there.
    """
    run_dir = config.run_dir()
    stage_dir = run_dir / "stage9"
    if config.stage9_input_csv:
        input_csv = Path(config.stage9_input_csv).expanduser().resolve()
    else:
        input_csv = run_dir / "stage6" / "question_tag.cleaned.csv"
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Stage 9 input not found: {input_csv}\n"
            f"Run `tagclean stage6` (or stage8) first, or pass --e5-audit-input <csv>."
        )

    knobs = f"{config.e5_audit_top_k}|{int(config.e5_audit_drop_on_top1_mismatch)}|{config.e5_model}|{int(config.e5_use_prefixes)}"
    input_hash = hashlib.sha256((file_sha256(input_csv) + "|" + knobs).encode("utf-8")).hexdigest()
    if resume and stage_done(stage_dir, input_hash):
        print(f"[stage9] skip: {stage_dir}")
        return

    df = pd.read_csv(input_csv)
    if "tag" not in df.columns and "tag_clean" in df.columns:
        df = df.rename(columns={"tag_clean": "tag"})
    if "question" not in df.columns or "tag" not in df.columns:
        raise ValueError(f"Stage 9 input must have question,tag columns; got {list(df.columns)}")
    df = df[["question", "tag"]].dropna().reset_index(drop=True).copy()
    df["question_norm"] = df["question"].astype(str).map(normalize_question)
    df = df[df["question_norm"].str.len() > 0].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"Stage 9 needs >=2 rows; got {len(df)}")

    texts = df["question_norm"].tolist()
    e5_inputs = _format_e5_passages(texts, config)
    if config.embedding_backend == "hashing":
        vecs = _hashing_embeddings(e5_inputs, config.hashing_dim)
    else:
        vecs = _encode_sentence_transformer(config.e5_model, e5_inputs, config)

    # Pure-numpy top-K — avoids importing FAISS after sentence-transformers
    # in the same process (segfaults at shutdown on Python 3.14 / Mac).
    # Vectors are L2-normalized by sentence-transformers; matmul gives cosine.
    top_k = max(2, int(config.e5_audit_top_k))
    search_k = min(top_k + 1, len(df))
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)  # exclude self
    indices = np.argpartition(-sims, kth=search_k - 1, axis=1)[:, :search_k]
    # argpartition isn't sorted; re-sort the top slice by similarity
    row_idx = np.arange(len(df))[:, None]
    ordered = np.argsort(-sims[row_idx, indices], axis=1)
    indices = indices[row_idx, ordered]
    tags = df["tag"].tolist()
    questions = df["question"].tolist()

    top1_tag, top1_q, top1_correct, own_share, neighbor_dist = [], [], [], [], []
    for i, neighbors in enumerate(indices):
        # diagonal already masked out, but keep the self-guard for safety.
        filtered = [int(n) for n in neighbors if int(n) != i][:top_k]
        if not filtered:
            top1_tag.append("")
            top1_q.append("")
            top1_correct.append(False)
            own_share.append(0.0)
            neighbor_dist.append("{}")
            continue
        nbr_tags = [tags[n] for n in filtered]
        own = sum(1 for t in nbr_tags if t == tags[i]) / len(nbr_tags)
        top1_tag.append(nbr_tags[0])
        top1_q.append(questions[filtered[0]])
        top1_correct.append(nbr_tags[0] == tags[i])
        own_share.append(round(own, 3))
        neighbor_dist.append(json.dumps(dict(Counter(nbr_tags)), ensure_ascii=False))

    audit = df[["question", "tag"]].copy()
    audit["top1_neighbor_tag"] = top1_tag
    audit["top1_neighbor_question"] = top1_q
    audit["top1_correct"] = top1_correct
    audit["own_share_top_k"] = own_share
    audit["neighbor_tag_dist"] = neighbor_dist
    audit["audit_top_k"] = top_k

    drop_mask = (~audit["top1_correct"]) if config.e5_audit_drop_on_top1_mismatch else pd.Series([False] * len(audit))
    kept = audit.loc[~drop_mask, ["question", "tag"]].copy()
    dropped = audit.loc[drop_mask].copy()

    stage_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(stage_dir / "e5_neighbor_audit.csv", index=False)
    kept.to_csv(stage_dir / "production_filtered.csv", index=False)
    dropped.to_csv(stage_dir / "e5_dropped.csv", index=False)

    per_tag_stats = (
        audit.groupby("tag")
        .agg(
            input_rows=("question", "size"),
            top1_accuracy=("top1_correct", "mean"),
            median_own_share=("own_share_top_k", "median"),
            dropped=("top1_correct", lambda s: int((~s).sum())),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    report = {
        "input_csv": str(input_csv),
        "rows_in": int(len(audit)),
        "audit_top_k": top_k,
        "drop_on_top1_mismatch": bool(config.e5_audit_drop_on_top1_mismatch),
        "rows_kept": int(len(kept)),
        "rows_dropped": int(len(dropped)),
        "top1_accuracy_in": float(audit["top1_correct"].mean()),
        "median_own_share_top_k": float(audit["own_share_top_k"].median()),
        "per_tag": per_tag_stats,
        "top_drop_examples": dropped.head(10)[
            ["question", "tag", "top1_neighbor_tag", "top1_neighbor_question", "own_share_top_k", "neighbor_tag_dist"]
        ].to_dict(orient="records"),
    }
    write_json(stage_dir / "audit_report.json", report)
    finish_stage(stage_dir, config, input_hash, {"rows_in": len(audit), "rows_kept": len(kept), "rows_dropped": len(dropped)})
    print(
        f"[stage9] wrote {stage_dir}  in={len(audit)} kept={len(kept)} dropped={len(dropped)}  "
        f"top1_acc={report['top1_accuracy_in']:.4f}  drop_on_mismatch={config.e5_audit_drop_on_top1_mismatch}"
    )


STAGES = {
    "stage0": run_stage0,
    "stage1": run_stage1,
    "stage2": run_stage2,
    "stage3": run_stage3,
    "stage4": run_stage4,
    "stage5": run_stage5,
    "stage6": run_stage6,
    "stage7": run_stage7,
    "stage8": run_stage8,
    "stage9": run_stage9,
}


def _stable_family_id(target_tags: Iterable[str]) -> str:
    """Stable 8-char hash family ID from the sorted tag set.

    Same tags -> same ID, regardless of discovery run. Lets us re-run discover
    deterministically and resume completed family runs by directory name.
    """
    payload = "|".join(sorted(target_tags))
    return "fam_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def run_discover(
    config: CleanerConfig,
    *,
    centroids_from: str,
    threshold: float = 0.88,
    pair_threshold: float = 0.86,
    top_k: int = 8,
    max_family_size: int = 4,
    out_path: Path,
    report_path: Path | None = None,
    production_tags: set[str] | None = None,
) -> None:
    """Seed-centric reciprocal-NN family discovery for the close-tag scope.

    Replaces the broken union-find in `find_close_tag_clusters` for the
    scaling case. Algorithm (Codex-approved):
      1. Per tag, top-K neighbors by min(cos_e5, cos_gemma).
      2. Edge kept iff reciprocal AND min(E5,Gemma) >= threshold.
      3. Per seed, ego-family = seed + reciprocal neighbors, capped at
         max_family_size, requiring pairwise min_sim >= pair_threshold with
         every existing member.
      4. Score each candidate by (min_edge_sim, avg_edge_sim, size).
      5. Greedy selection by descending score; mark members covered.
      6. Uncovered tags become singletons.
    Writes a hand-editable `families.yaml` and an optional human-readable
    report. No transitive closure -> no mega-components at corpus scale.
    """
    centroids_dir = config.artifact_root / centroids_from / "stage2"
    if not (centroids_dir / "tag_index.json").exists():
        raise FileNotFoundError(
            f"discover needs cached stage2 outputs at {centroids_dir}; "
            f"run a stage2 (or stage8) pass first."
        )
    tag_index = read_json(centroids_dir / "tag_index.json")
    tags: list[str] = list(tag_index.get("tags", []))
    e5 = np.load(centroids_dir / "tag_centroids_e5.npy")
    gemma = np.load(centroids_dir / "tag_centroids_gemma.npy")

    # Renormalize defensively — Stage 2 centroids are means, not unit norm.
    e5 = e5 / np.clip(np.linalg.norm(e5, axis=1, keepdims=True), 1e-12, None)
    gemma = gemma / np.clip(np.linalg.norm(gemma, axis=1, keepdims=True), 1e-12, None)

    sim_e5 = (e5 @ e5.T).astype(np.float32)
    sim_gem = (gemma @ gemma.T).astype(np.float32)
    sim = np.minimum(sim_e5, sim_gem)
    np.fill_diagonal(sim, -np.inf)

    n = len(tags)
    allow = np.ones(n, dtype=bool)
    if production_tags is not None:
        tag_set = set(tags)
        unknown = sorted(t for t in production_tags if t not in tag_set)
        if unknown:
            # Hard-fail — at 990-tag scale, one allowlist typo silently drops
            # production coverage and we'd never notice. Refuse and surface.
            raise ValueError(
                f"--production-tags contains {len(unknown)} entries not in the "
                f"corpus tag_index: {unknown[:8]}{'...' if len(unknown) > 8 else ''}"
            )
        for i, t in enumerate(tags):
            if t not in production_tags:
                allow[i] = False
    if config.discover_exclude_pattern:
        rx = re.compile(config.discover_exclude_pattern)
        for i, t in enumerate(tags):
            if rx.search(t):
                allow[i] = False
    sim[~allow, :] = -np.inf
    sim[:, ~allow] = -np.inf

    # Top-K neighbors per tag (sorted by similarity desc).
    eff_k = min(int(top_k), max(1, n - 1))
    if eff_k <= 0:
        raise ValueError(f"top_k must be >=1; got {top_k}")
    nbr_partial = np.argpartition(-sim, eff_k - 1, axis=1)[:, :eff_k]
    nbr_idx = np.empty_like(nbr_partial)
    for i in range(n):
        order = np.argsort(-sim[i, nbr_partial[i]])
        nbr_idx[i] = nbr_partial[i][order]
    in_top_k = [set(int(j) for j in nbr_idx[i].tolist()) for i in range(n)]

    # Row counts from stage2/tag_profile.parquet (best-effort; binary may be absent
    # in a fresh checkout, so silently degrade).
    profile_path = centroids_dir / "tag_profile.parquet"
    row_count: dict[str, int] = {}
    if profile_path.exists():
        try:
            prof = pd.read_parquet(profile_path)
            row_count = dict(zip(prof["tag"], prof["row_count"].astype(int)))
        except Exception:
            row_count = {}

    # Build ego-family candidates per allowed seed.
    candidates: list[dict[str, Any]] = []
    for i in range(n):
        if not allow[i]:
            continue
        members: list[int] = [i]
        for j in nbr_idx[i].tolist():
            if int(j) == i or not allow[int(j)]:
                continue
            if sim[i, int(j)] < threshold:
                continue
            if i not in in_top_k[int(j)]:  # reciprocity check
                continue
            # Pairwise check against existing members.
            ok = True
            for m in members:
                if m == i:
                    continue
                if min(float(sim[m, int(j)]), float(sim[int(j), m])) < pair_threshold:
                    ok = False
                    break
            if not ok:
                continue
            members.append(int(j))
            if len(members) >= max_family_size:
                break

        if len(members) < 2:
            candidates.append(
                {"members": members, "is_singleton": True, "min_sim": 1.0, "avg_sim": 1.0}
            )
            continue
        edges = [
            float(sim[members[a], members[b]])
            for a in range(len(members))
            for b in range(a + 1, len(members))
        ]
        candidates.append(
            {
                "members": members,
                "is_singleton": False,
                "min_sim": float(min(edges)),
                "avg_sim": float(sum(edges) / len(edges)),
            }
        )

    # Greedy: take strongest non-singleton candidates first; mark covered.
    multi = sorted(
        [c for c in candidates if not c["is_singleton"]],
        key=lambda c: (-c["min_sim"], -c["avg_sim"], -len(c["members"])),
    )
    multi_covered: set[int] = set()  # only tags claimed by a SELECTED multi-family
    covered: set[int] = set()  # multi_covered + singleton fills (full coverage)
    selected: list[dict[str, Any]] = []
    for c in multi:
        if any(m in multi_covered for m in c["members"]):
            continue
        multi_covered.update(c["members"])
        covered.update(c["members"])
        selected.append(c)
    # Singletons fill the gaps so coverage is complete. Iterate ALL allowed
    # tags (not just `candidates`) — a tag whose only candidate was multi-tag
    # may get rejected by greedy overlap and would otherwise be orphaned.
    for i in range(n):
        if not allow[i] or i in covered:
            continue
        covered.add(i)
        selected.append({"members": [i], "is_singleton": True, "min_sim": 1.0, "avg_sim": 1.0})

    # Build manifest records with excluded-neighbor diagnostics.
    family_records: list[dict[str, Any]] = []
    for c in selected:
        member_set = set(c["members"])
        target_tags = sorted(tags[m] for m in c["members"])
        family_id = _stable_family_id(target_tags)
        excluded: dict[str, dict[str, Any]] = {}
        for m in c["members"]:
            for j in nbr_idx[m].tolist():
                jj = int(j)
                if jj in member_set or jj == m:
                    continue
                tag_j = tags[jj]
                s = float(sim[m, jj])
                if s < threshold:
                    reason = "below_threshold"
                elif m not in in_top_k[jj] or jj not in in_top_k[m]:
                    reason = "non_reciprocal"
                elif jj in multi_covered:
                    # Only true if neighbor was claimed by another selected
                    # multi-tag family (not by a singleton fill); otherwise
                    # the reason is "missed reciprocity / pair threshold".
                    reason = "covered_by_other_family"
                else:
                    reason = "below_pair_threshold"
                if tag_j not in excluded or s > excluded[tag_j]["min_sim"]:
                    excluded[tag_j] = {"tag": tag_j, "min_sim": round(s, 4), "reason": reason}
        excluded_list = sorted(excluded.values(), key=lambda e: -e["min_sim"])[:5]

        family_records.append(
            {
                "family_id": family_id,
                "status": "singleton" if c["is_singleton"] else "approved",
                "target_tags": target_tags,
                "score": {
                    "min_edge_sim": round(c["min_sim"], 4),
                    "avg_edge_sim": round(c["avg_sim"], 4),
                },
                "row_counts": {t: int(row_count.get(t, 0)) for t in target_tags},
                "excluded_neighbors": excluded_list,
                "notes": "",
            }
        )

    payload = {
        "schema_version": 1,
        "source": {
            "centroids_run_id": centroids_from,
            "tag_index": str(centroids_dir / "tag_index.json"),
            "e5_centroids": str(centroids_dir / "tag_centroids_e5.npy"),
            "gemma_centroids": str(centroids_dir / "tag_centroids_gemma.npy"),
        },
        "discover_config": {
            "threshold": float(threshold),
            "pair_threshold": float(pair_threshold),
            "top_k": int(top_k),
            "max_family_size": int(max_family_size),
            "production_tags_count": int(allow.sum()),
        },
        "families": family_records,
    }
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    multi_recs = [r for r in family_records if r["status"] == "approved"]
    singles = [r for r in family_records if r["status"] == "singleton"]
    size_hist = Counter(len(r["target_tags"]) for r in multi_recs)
    print(f"[discover] wrote {out_path}")
    print(
        f"[discover] {len(multi_recs)} multi-tag families "
        f"(sizes: {dict(sorted(size_hist.items()))}), {len(singles)} singletons; "
        f"coverage {len(covered)}/{n} tags."
    )

    if report_path is not None:
        report_path = Path(report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"# tagclean discover report\n\n")
            f.write(f"- centroids: `{centroids_dir}`\n")
            f.write(
                f"- thresholds: edge={threshold}, pair={pair_threshold}, top_k={top_k}, max_family_size={max_family_size}\n"
            )
            f.write(f"- multi-tag families: {len(multi_recs)}\n")
            f.write(f"- singletons: {len(singles)}\n")
            f.write(f"- coverage: {len(covered)}/{n}\n\n")
            f.write("## Multi-tag families (sorted by min_edge_sim desc)\n\n")
            for fam in sorted(multi_recs, key=lambda r: -r["score"]["min_edge_sim"]):
                f.write(
                    f"### `{fam['family_id']}`  min={fam['score']['min_edge_sim']:.3f}  avg={fam['score']['avg_edge_sim']:.3f}\n"
                )
                f.write(f"- tags: {', '.join(fam['target_tags'])}\n")
                f.write(f"- row counts: {fam['row_counts']}\n")
                if fam["excluded_neighbors"]:
                    nearest = fam["excluded_neighbors"][:3]
                    f.write(f"- nearest excluded: {nearest}\n")
                f.write("\n")
        print(f"[discover] report at {report_path}")


def _symlink_shared_stages(
    artifact_root: Path,
    family_run_id: str,
    centroids_run_id: str,
    stages: Iterable[str] = ("stage0", "stage1", "stage2"),
) -> None:
    """Symlink stage0/1/2 from the centroids run into the family run dir.

    Stages 0–2 are corpus-wide and identical across families that share the
    same input CSV; symlinking saves ~30-60 min Mac MPS embedding per family.
    If a destination already exists, verify it points at the expected source
    — a stale family dir from a different centroids run mixes geometries and
    silently corrupts downstream stages. Refuse rather than guess.
    """
    src_root = (artifact_root / centroids_run_id).resolve()
    dst_root = artifact_root / family_run_id
    dst_root.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        src = src_root / stage
        dst = dst_root / stage
        if not src.exists():
            continue
        if dst.is_symlink():
            try:
                existing = (dst_root / os.readlink(dst)).resolve()
            except OSError:
                existing = None
            if existing != src:
                raise RuntimeError(
                    f"{dst} is a symlink to {existing}, not the manifest's "
                    f"centroids_run_id={centroids_run_id} ({src}). Refusing to "
                    "mix geometries; remove the stale family dir and retry."
                )
            continue
        if dst.exists():
            # A real directory at this path — assume the user intentionally
            # populated it (e.g. a separate stage0/1/2 run with matching hash).
            # We don't validate hash equality here; cross-run mismatch would
            # surface as stage-manifest hash failures downstream.
            continue
        rel = os.path.relpath(src, dst_root)
        dst.symlink_to(rel)


def run_families_manifest(
    config: CleanerConfig,
    manifest_path: Path,
    skip_completed: bool = True,
    include_singletons: bool = False,
    force_rerun: bool = False,
) -> None:
    """Run stage8 for every approved family in the manifest.

    Each family becomes a `tagclean stage8 --target-tags <tags> --run-id <family_id>`
    call, executed in-process. Stage 0–2 outputs are symlinked from the
    manifest's `source.centroids_run_id` so each family skips the redundant
    full-corpus embedding (huge scaling win at 264+ families).
    """
    import copy

    manifest_path = Path(manifest_path).expanduser().resolve()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    families = payload.get("families", [])
    centroids_run_id = (payload.get("source") or {}).get("centroids_run_id")

    todo: list[dict[str, Any]] = []
    for fam in families:
        status = fam.get("status")
        tags = fam.get("target_tags", [])
        if status == "approved" and len(tags) >= 2:
            todo.append(fam)
        elif include_singletons and status == "singleton" and len(tags) == 1:
            todo.append(fam)
    if skip_completed:
        todo = [
            fam
            for fam in todo
            if not (
                config.artifact_root / fam["family_id"] / "stage8" / "cleaning_report.json"
            ).exists()
        ]

    print(f"[run-families] {len(todo)} families to run")
    if centroids_run_id and not (config.artifact_root / centroids_run_id / "stage2").exists():
        print(
            f"[run-families] WARNING: centroids_run_id={centroids_run_id} stage2 missing; "
            "each family will redo stage0-2 from scratch (slow)."
        )
        centroids_run_id = None

    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    for i, fam in enumerate(todo, 1):
        run_id = fam["family_id"]
        target_tags = list(fam["target_tags"])
        print(f"[run-families] {i}/{len(todo)}  {run_id}  tags={target_tags}")
        if centroids_run_id and run_id != centroids_run_id:
            _symlink_shared_stages(config.artifact_root, run_id, centroids_run_id)
        family_config = copy.copy(config)
        family_config.run_id = run_id
        family_config.target_tags = target_tags
        try:
            # force_rerun bypasses every per-stage cache so a manual override
            # actually re-runs (resume=True would silently skip-on-hash).
            run_stage8(family_config, resume=not force_rerun)
            successes.append(run_id)
        except Exception as exc:
            print(f"[run-families] FAILED {run_id}: {exc}")
            failures.append((run_id, str(exc)))
    print(f"[run-families] done: {len(successes)} ok, {len(failures)} failed")
    if failures:
        for run_id, msg in failures:
            print(f"[run-families]   FAILED {run_id}: {msg[:160]}")


def run_compose(
    config: CleanerConfig,
    from_runs: list[str],
    source: str,
    out_path: Path,
) -> None:
    """Concatenate per-family cleaning outputs into a single production CSV.

    Each entry in `from_runs` contributes its `stage6/question_tag.<source>.csv`
    (default `top40`). Rows are deduplicated on (question, tag) and emitted in
    deterministic (tag, question) order. The result is the corpus you feed to
    `tagclean stage9 --e5-audit-input <path>` for the cross-family E5 audit.

    Compose from `top40.csv`, not from per-family `production_filtered.csv` —
    a row that fails LOO inside a 3-tag family can be safely separable in the
    9+-tag production union (its in-family competitor is no longer a peer).
    Filter once, globally, after composition.
    """
    if source not in {"top40", "cleaned"}:
        raise ValueError(f"--compose-source must be 'top40' or 'cleaned'; got {source!r}")
    file_name = (
        "question_tag.top40.csv" if source == "top40" else "question_tag.cleaned.csv"
    )

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    per_run: dict[str, tuple[int, int]] = {}
    for run_id in from_runs:
        src = config.artifact_root / run_id / "stage6" / file_name
        if not src.exists():
            raise FileNotFoundError(f"Source CSV not found for run {run_id!r}: {src}")
        frame = pd.read_csv(src)
        if "tag" not in frame.columns and "tag_clean" in frame.columns:
            frame = frame.rename(columns={"tag_clean": "tag"})
        if "question" not in frame.columns or "tag" not in frame.columns:
            raise ValueError(
                f"{src} must have question,tag columns; got {list(frame.columns)}"
            )
        frame = frame[["question", "tag"]].dropna()
        added = 0
        for _, r in frame.iterrows():
            key = (str(r["question"]), str(r["tag"]))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append({"question": key[0], "tag": key[1]})
            added += 1
        per_run[run_id] = (len(frame), added)

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed = pd.DataFrame(rows)
    composed = composed.sort_values(["tag", "question"]).reset_index(drop=True)
    composed.to_csv(out_path, index=False)

    print(f"[compose] wrote {out_path}")
    print(
        f"[compose] total rows: {len(composed)}, "
        f"unique tags: {composed['tag'].nunique()}, "
        f"duplicates skipped: {duplicates}"
    )
    for run_id, (in_rows, added) in per_run.items():
        print(f"[compose]   {run_id}: {in_rows} {source} rows -> {added} added")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tagclean", description="LLM-assisted FAQ dataset cleaner.")
    parser.add_argument(
        "stage", choices=[*STAGES.keys(), "all", "compose", "discover", "run-families"]
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None, help="Path to question_tag.csv (overrides config)")
    parser.add_argument("--tag-answer", type=Path, default=None, help="Path to tag_answer.json (overrides config)")
    parser.add_argument("--artifact-root", type=Path, default=None, help="Output dir for artifacts (overrides config)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--judge-mode", choices=["batch_prepare", "batch_submit", "batch_collect", "sync", "agents", "heuristic"], default=None)
    parser.add_argument("--openai-model", default=None, help="Override config.openai_model (e.g. gpt-5.4, gpt-5.5)")
    parser.add_argument("--language", choices=["bn", "none"], default=None, help="Text-normalization language (default: bn)")
    parser.add_argument("--embedding-backend", choices=["sentence-transformers", "hashing"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-tags", type=int, default=None)
    parser.add_argument("--max-rows-per-tag", type=int, default=None)
    parser.add_argument("--target-tags", default=None, help="Comma-separated tags to clean while using the full corpus as evidence")
    parser.add_argument("--target-max-tags", type=int, default=None, help="Clean only the first N tags while keeping full corpus evidence")
    parser.add_argument("--seed-tag", default=None, help="Resolve the close-tag cluster containing this tag and clean its members")
    parser.add_argument("--self-consistency-passes", type=int, default=None)
    parser.add_argument("--tags-per-judge-call", type=int, default=None)
    parser.add_argument("--rows-per-tag-per-judge-call", type=int, default=None)
    parser.add_argument("--batch-output", type=Path, default=None)
    parser.add_argument(
        "--e5-audit-input",
        type=Path,
        default=None,
        help="Stage 9: external CSV (question,tag) to audit instead of <run>/stage6/question_tag.cleaned.csv. "
        "Use when auditing a unioned production set across multiple family runs.",
    )
    parser.add_argument(
        "--e5-audit-k",
        type=int,
        default=None,
        help="Stage 9: top-K neighborhood size for own-share severity (default 10).",
    )
    parser.add_argument(
        "--no-e5-drop",
        action="store_true",
        help="Stage 9: report only — do not drop top-1 mismatches into a filtered set.",
    )
    parser.add_argument(
        "--from-runs",
        default=None,
        help="compose: comma-separated list of run_ids to combine into a single production CSV.",
    )
    parser.add_argument(
        "--compose-source",
        choices=["top40", "cleaned"],
        default="top40",
        help="compose: which per-run file to pull from (default: top40).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="compose: output CSV path. Default: <artifact_root>/<run_id>/composed_<source>.csv.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="compose / run-families: families.yaml manifest path.",
    )
    parser.add_argument(
        "--centroids-from",
        default=None,
        help="discover: existing run_id whose stage2 outputs supply the centroids.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.88,
        help="discover: min(cos_e5, cos_gemma) edge threshold (default 0.88).",
    )
    parser.add_argument(
        "--pair-threshold",
        type=float,
        default=0.86,
        help="discover: required pairwise similarity between every family member (default 0.86).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="discover: candidate-pool size per tag before reciprocity check (default 8).",
    )
    parser.add_argument(
        "--max-family-size",
        type=int,
        default=4,
        help="discover: cap members per family (default 4; 2-4 recommended).",
    )
    parser.add_argument(
        "--production-tags",
        type=Path,
        default=None,
        help="discover: optional newline-delimited file of tags to restrict discovery to.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="discover: optional Markdown report path.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="run-families: re-include families that already have a stage8 cleaning_report.json. "
        "Per-stage caches still apply unless --force-rerun is set.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="run-families: force resume=False on each family — every stage re-runs from scratch.",
    )
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="run-families: also clean singleton tags (default: skip; multi-tag only).",
    )
    parser.add_argument(
        "--exclude-tag-pattern",
        default=None,
        help="discover: regex of tag names to exclude from family discovery "
        "(e.g. '_followup_[a-d]$' to skip dialog-turn artifacts).",
    )
    return parser.parse_args()


def resolve_seed_cluster(config: CleanerConfig, seed_tag: str) -> list[str]:
    """Run stages 0-2 (resume-friendly), then return the close-tag cluster
    containing `seed_tag`. Falls back to [seed_tag] if it has no close siblings.

    Uses the same find_close_tag_clusters logic Stage 3 relies on, so the
    cluster the user gets is exactly the one that will receive a boundary policy.
    """
    run_stage2(config, resume=True)
    run_dir = config.run_dir()
    tag_index = read_json(run_dir / "stage2" / "tag_index.json")
    tags: list[str] = list(tag_index.get("tags", []))
    if seed_tag not in tags:
        raise ValueError(
            f"Seed tag {seed_tag!r} not found in dataset (have {len(tags)} tags). "
            f"Sample: {tags[:5]}"
        )
    e5_cents = np.load(run_dir / "stage2" / "tag_centroids_e5.npy")
    gemma_cents = np.load(run_dir / "stage2" / "tag_centroids_gemma.npy")
    sim_e5 = e5_cents @ e5_cents.T
    sim_gemma = gemma_cents @ gemma_cents.T
    clusters = find_close_tag_clusters(
        tags,
        np.asarray(sim_e5, dtype=np.float32),
        np.asarray(sim_gemma, dtype=np.float32),
        threshold=config.boundary_policy_threshold,
        max_cluster_size=config.boundary_policy_max_cluster_size,
    )
    cluster_for_seed = [seed_tag]
    for cluster in clusters:
        if seed_tag in cluster:
            cluster_for_seed = list(cluster)
            break

    # Diagnostics: top-5 nearest tags NOT in the cluster so the user sees why
    # siblings didn't make the cut and can lower the threshold or pass
    # --target-tags explicitly.
    seed_idx = tags.index(seed_tag)
    in_cluster = set(cluster_for_seed)
    excluded = [
        (
            tags[j],
            float(sim_e5[seed_idx, j]),
            float(sim_gemma[seed_idx, j]),
            min(float(sim_e5[seed_idx, j]), float(sim_gemma[seed_idx, j])),
        )
        for j in range(len(tags))
        if tags[j] != seed_tag and tags[j] not in in_cluster
    ]
    excluded.sort(key=lambda r: r[3], reverse=True)
    if excluded:
        print(
            f"[seed] threshold={config.boundary_policy_threshold:.2f} "
            f"(min of E5/Gemma must clear). Nearest excluded tags:"
        )
        for tag, e5, gemma, mn in excluded[:5]:
            hint = " ← sibling, just below threshold" if mn >= 0.75 else ""
            print(f"        {tag:<40s}  E5={e5:.3f}  Gemma={gemma:.3f}  min={mn:.3f}{hint}")
    return cluster_for_seed


_SECRET_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key|secret|token|password|bearer)")


def _redact_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    """Defensive: if a future config field looks like a secret, don't dump it."""
    return {k: ("***REDACTED***" if _SECRET_KEY_PATTERN.search(k) else v) for k, v in payload.items()}


def write_run_manifest(config: CleanerConfig, stage: str) -> Path:
    """Top-level manifest distinct from per-stage manifests; identifies a run."""
    run_dir = config.run_dir()
    payload = {
        "tagclean_version": "0.1.0",
        "stage_invoked": stage,
        "run_id": config.resolved_run_id(),
        "seed_tag": getattr(config, "_seed_tag", None),
        "input_csv": str(config.input_csv),
        "input_csv_sha256": file_sha256(config.input_csv) if config.input_csv.exists() else None,
        "tag_answer_json": str(config.tag_answer_json),
        "tag_answer_sha256": file_sha256(config.tag_answer_json) if config.tag_answer_json.exists() else None,
        "models": {
            "e5": config.e5_model,
            "gemma": config.gemma_model,
            "openai": config.openai_model,
        },
        "judge_mode": config.judge_mode,
        "language": config.language,
        "config": _redact_secrets({
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()
        }),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = run_dir / "run_manifest.json"
    write_json(path, payload)
    return path


def main() -> None:
    global _NORMALIZATION_LANGUAGE
    args = parse_args()
    config = load_config(args.config)
    if args.input:
        config.input_csv = args.input.resolve()
    if args.tag_answer:
        config.tag_answer_json = args.tag_answer.resolve()
    if args.artifact_root:
        config.artifact_root = args.artifact_root.resolve()
    if args.run_id:
        config.run_id = args.run_id
    if args.judge_mode:
        config.judge_mode = args.judge_mode
    if args.openai_model:
        config.openai_model = args.openai_model
    if args.language:
        config.language = args.language
    if args.embedding_backend:
        config.embedding_backend = args.embedding_backend
    if args.device:
        config.device = None if args.device == "auto" else args.device
    if args.max_rows is not None:
        config.max_rows = args.max_rows
    if args.max_tags is not None:
        config.max_tags = args.max_tags
    if args.max_rows_per_tag is not None:
        config.max_rows_per_tag = args.max_rows_per_tag
    if args.target_tags:
        config.target_tags = [tag.strip() for tag in args.target_tags.split(",") if tag.strip()]
    if args.target_max_tags is not None:
        config.target_max_tags = args.target_max_tags
    if args.seed_tag:
        if config.target_tags:
            print(f"[seed] --seed-tag overrides --target-tags ({config.target_tags})")
        cluster = resolve_seed_cluster(config, args.seed_tag)
        print(f"[seed] resolved cluster from '{args.seed_tag}': {cluster}")
        config.target_tags = cluster
        config._seed_tag = args.seed_tag  # surfaced in run_manifest.json
    if args.self_consistency_passes is not None:
        config.self_consistency_passes = args.self_consistency_passes
    if args.tags_per_judge_call is not None:
        config.tags_per_judge_call = args.tags_per_judge_call
    if args.rows_per_tag_per_judge_call is not None:
        config.rows_per_tag_per_judge_call = args.rows_per_tag_per_judge_call
    if args.e5_audit_input is not None:
        config.stage9_input_csv = str(args.e5_audit_input.resolve())
    if args.e5_audit_k is not None:
        config.e5_audit_top_k = args.e5_audit_k
    if args.no_e5_drop:
        config.e5_audit_drop_on_top1_mismatch = False

    _NORMALIZATION_LANGUAGE = config.language

    resume = args.resume or not args.no_resume
    if args.stage == "discover":
        if not args.centroids_from:
            raise SystemExit(
                "discover requires --centroids-from RUN_ID (a run with cached stage2/ output)."
            )
        if not args.out:
            raise SystemExit("discover requires --out PATH (where to write families.yaml).")
        production_tags: set[str] | None = None
        if args.production_tags is not None:
            production_tags = {
                line.strip()
                for line in args.production_tags.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            }
        if args.exclude_tag_pattern is not None:
            config.discover_exclude_pattern = args.exclude_tag_pattern
        run_discover(
            config,
            centroids_from=args.centroids_from,
            threshold=args.threshold,
            pair_threshold=args.pair_threshold,
            top_k=args.top_k,
            max_family_size=args.max_family_size,
            out_path=args.out,
            report_path=args.report,
            production_tags=production_tags,
        )
        return
    if args.stage == "run-families":
        if not args.manifest:
            raise SystemExit("run-families requires --manifest families.yaml")
        run_families_manifest(
            config,
            manifest_path=args.manifest,
            skip_completed=not args.no_skip_completed,
            include_singletons=args.include_singletons,
            force_rerun=args.force_rerun,
        )
        return
    if args.stage == "compose":
        if args.manifest:
            payload = yaml.safe_load(
                Path(args.manifest).expanduser().read_text(encoding="utf-8")
            )
            # Include singletons too — if run-families --include-singletons
            # was used, those single-tag runs produce stage6 outputs that
            # belong in the production union. Filter only on status, not size.
            from_runs = [
                fam["family_id"]
                for fam in payload.get("families", [])
                if fam.get("status") in {"approved", "singleton"}
                and len(fam.get("target_tags", [])) >= 1
            ]
            # Drop runs whose stage6/cleaned.csv (or top40.csv) doesn't exist —
            # singletons that weren't run via --include-singletons fall here.
            file_name = (
                "question_tag.top40.csv"
                if args.compose_source == "top40"
                else "question_tag.cleaned.csv"
            )
            from_runs = [
                fid
                for fid in from_runs
                if (config.artifact_root / fid / "stage6" / file_name).exists()
            ]
            if not from_runs:
                raise SystemExit(
                    "manifest has no families with cleaned stage6 outputs to compose"
                )
        elif args.from_runs:
            from_runs = [r.strip() for r in args.from_runs.split(",") if r.strip()]
        else:
            raise SystemExit(
                "compose requires either --from-runs run1,run2,... or --manifest families.yaml"
            )
        out_path = (
            args.out
            if args.out
            else config.run_dir() / f"composed_{args.compose_source}.csv"
        )
        run_compose(config, from_runs, args.compose_source, out_path)
        return
    if args.stage == "all":
        if config.judge_mode in {"batch_prepare", "batch_submit"}:
            run_stage5(config, resume=resume, batch_output=args.batch_output)
            print(
                "[all] stopped after stage5 because judge_mode is asynchronous. "
                "Collect the batch output, then rerun with judge_mode=batch_collect, sync, or agents."
            )
            return
        run_stage8(config, resume=resume)
    elif args.stage == "stage5":
        run_stage5(config, resume=resume, batch_output=args.batch_output)
    else:
        STAGES[args.stage](config, resume=resume)

    manifest_path = write_run_manifest(config, args.stage)
    print(f"[run] wrote {manifest_path}")


if __name__ == "__main__":
    main()
