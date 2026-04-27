"""State container for the deterministic geometric repair loop.

Pure data + light helpers. All decision logic lives in repair.py.

Geometry-only design (per user constraint): no tag-name patterns, no
tag_answer.json. Tag identity is an integer index; tag_names is just a
display map maintained alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# Per-row final status values (written to final_assignment.parquet).
STATUS_KEPT = "kept"
STATUS_DROPPED = "dropped"
STATUS_REASSIGNED = "reassigned"
STATUS_MERGED_INTO = "merged_into"


@dataclass
class RepairState:
    """Mutable state of the repair loop.

    Tag identity is an int index into `tag_names`. Names are display-only —
    no decisions look at them. Indices are stable: a merged tag stays in the
    arrays but its `alive_tags[idx]` flips to False; downstream code skips
    dead tags via the alive mask rather than re-indexing (which would
    invalidate every other index in flight).
    """

    # ---- immutable inputs (set at iter 0, never change) ----
    embeddings: np.ndarray              # (N, D), L2-normalized float32
    row_ids: np.ndarray                 # (N,) int64 — original CSV row_id
    question_raw: list[str]             # (N,) — for the final output CSV
    original_assignment: np.ndarray     # (N,) int32 — what the input CSV said

    # ---- mutable per-iteration state ----
    assignment: np.ndarray              # (N,) int32 — current tag index
    moves_remaining: np.ndarray         # (N,) int8  — per-row reassignment budget
    prior_tag: np.ndarray               # (N,) int32 — last tag this row CAME from (-1 = never moved)
    status: np.ndarray                  # (N,) object — STATUS_* (terminal: dropped, merged_into; row state otherwise STATUS_KEPT)

    tag_names: list[str]                # (T,) display names, stable indexing
    alive_tags: np.ndarray              # (T,) bool — False once dissolved or merged into another
    canonical_of: np.ndarray            # (T,) int32 — if merged, points to canonical tag's index; else == self

    current_centroids: np.ndarray       # (T, D) — recomputed each iter (with damping)
    previous_centroids: np.ndarray      # (T, D) — for damping anchor
    frozen_centroids: np.ndarray        # (T, D) — iter-0 centroids; for absorber override + merge hysteresis
    frozen_pairwise_cosine: np.ndarray  # (T, T) — iter-0 pairwise centroid cosines; merge hysteresis predicate

    untrusted_tags: set[int] = field(default_factory=set)
    absorber_history: dict[int, list[float]] = field(default_factory=dict)  # tag_idx -> [intake_frac per iter]

    iter: int = 0

    # ---------- factory ----------
    @classmethod
    def from_inputs(
        cls,
        embeddings: np.ndarray,
        row_ids: np.ndarray,
        question_raw: list[str],
        tags_per_row: list[str],
        move_budget: int = 3,
    ) -> "RepairState":
        """Build initial state. Embeddings must already be L2-normalized.

        `tags_per_row[i]` is the tag string for row i. We assign an integer
        index to each unique tag (sorted alphabetically for determinism).
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        n, _d = embeddings.shape
        if len(row_ids) != n or len(question_raw) != n or len(tags_per_row) != n:
            raise ValueError("row_ids, question_raw, tags_per_row must all match embeddings length")

        unique_tags = sorted(set(tags_per_row))
        tag_to_idx = {t: i for i, t in enumerate(unique_tags)}
        T = len(unique_tags)

        assignment = np.array([tag_to_idx[t] for t in tags_per_row], dtype=np.int32)
        return cls(
            embeddings=embeddings.astype(np.float32, copy=False),
            row_ids=row_ids.astype(np.int64, copy=False),
            question_raw=list(question_raw),
            original_assignment=assignment.copy(),
            assignment=assignment.copy(),
            moves_remaining=np.full(n, move_budget, dtype=np.int8),
            prior_tag=np.full(n, -1, dtype=np.int32),
            status=np.full(n, STATUS_KEPT, dtype=object),
            tag_names=list(unique_tags),
            alive_tags=np.ones(T, dtype=bool),
            canonical_of=np.arange(T, dtype=np.int32),
            current_centroids=np.zeros((T, embeddings.shape[1]), dtype=np.float32),
            previous_centroids=np.zeros((T, embeddings.shape[1]), dtype=np.float32),
            frozen_centroids=np.zeros((T, embeddings.shape[1]), dtype=np.float32),
            frozen_pairwise_cosine=np.zeros((T, T), dtype=np.float32),
        )

    # ---------- views ----------
    @property
    def n_rows(self) -> int:
        return len(self.assignment)

    @property
    def n_tags(self) -> int:
        return len(self.tag_names)

    @property
    def d(self) -> int:
        return self.embeddings.shape[1]

    def alive_indices(self) -> np.ndarray:
        """Indices of tags currently alive (not merged or dissolved)."""
        return np.where(self.alive_tags)[0]

    def kept_row_mask(self) -> np.ndarray:
        """True for rows that are still in play (not dropped / merged_into)."""
        return self.status == STATUS_KEPT

    def rows_in_tag(self, tag_idx: int) -> np.ndarray:
        """Indices of CURRENTLY KEPT rows assigned to this tag."""
        mask = (self.assignment == tag_idx) & self.kept_row_mask()
        return np.where(mask)[0]

    # ---------- pickle helpers ----------
    def to_dict_minimal(self) -> dict:
        """For per-iter ops.jsonl logging — small scalar summary, no big arrays."""
        kept = self.kept_row_mask()
        return {
            "iter": self.iter,
            "n_rows_alive": int(kept.sum()),
            "n_rows_dropped": int((self.status == STATUS_DROPPED).sum()),
            "n_tags_alive": int(self.alive_tags.sum()),
            "n_tags_dissolved": int((~self.alive_tags).sum()),
            "n_untrusted": len(self.untrusted_tags),
        }

    def final_assignment_frame(self) -> pd.DataFrame:
        """Build the per-row output frame (final_assignment.parquet shape).

        Resolves merged tags through the canonical_of chain so the displayed
        repaired_tag is always the surviving canonical.
        """
        repaired_idx = np.array([self._resolve_canonical(int(t)) for t in self.assignment], dtype=np.int32)
        original_names = [self.tag_names[i] for i in self.original_assignment]
        repaired_names = [self.tag_names[i] for i in repaired_idx]
        return pd.DataFrame({
            "row_id": self.row_ids,
            "question": self.question_raw,
            "original_tag": original_names,
            "repaired_tag": repaired_names,
            "status": self.status,
        })

    def _resolve_canonical(self, idx: int) -> int:
        """Walk canonical_of until fixed point (with cycle guard)."""
        seen = set()
        cur = idx
        while self.canonical_of[cur] != cur:
            if cur in seen:
                raise RuntimeError(f"canonical_of cycle detected at tag idx {cur}")
            seen.add(cur)
            cur = int(self.canonical_of[cur])
        return cur
