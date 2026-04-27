"""Deterministic geometric repair loop (E5-only).

Goal: drive LOO top-1 = own_tag for every surviving row, via drop / reassign /
merge moves driven entirely by E5 embedding geometry. No tag-name patterns,
no tag_answer.json — pure geometry.

See /home/synesis/.claude/plans/zazzy-sleeping-sketch.md for the full design.
"""

from __future__ import annotations

# Resource caps — must be set BEFORE any thread-using lib imports.
# 4 CPU threads + 8 GB RAM ceiling per user constraint.
import os as _os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "4")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Cap FAISS thread pool (separate from BLAS).
try:
    import faiss as _faiss_threadcfg
    _faiss_threadcfg.omp_set_num_threads(4)
except Exception:  # pragma: no cover  — faiss optional in some test envs
    pass

from .cleaner import (
    centroid as _centroid_trimmed_input,  # operates on already-selected vector subset
    l2_normalize,
    _trimmed_indices,
)
from .repair_state import (
    RepairState,
    STATUS_DROPPED,
    STATUS_KEPT,
    STATUS_MERGED_INTO,
    STATUS_REASSIGNED,
)


# -------------------- threshold defaults (Bengali NID + E5-large-instruct) --------------------

@dataclass
class RepairConfig:
    # Reassignment
    reassign_thresh: float = -0.03         # margin below this triggers eligibility
    delta_move: float = 0.05               # min margin gap to move forward
    delta_hyst: float = 0.02               # extra gap required to move BACK to prior_tag
    knn_top_k: int = 10                    # K for kNN tag-share vector
    intake_cap_per_tag: float = 0.10       # max reassigns IN per iter / tag size

    # Drop
    cross_tag_dup_thresh: float = 0.985    # cosine for cross-tag dup detection
    hard_drop_thresh: float = -0.10        # hard-margin drop
    drop_cap_per_tag: float = 0.20         # max drops per iter / tag size
    drop_cap_strikes: int = 3              # consecutive cap hits → abort

    # Merge (Phase C)
    merge_cosine_thresh: float = 0.92
    merge_knn_overlap_thresh: float = 0.50
    mutual_confusion_thresh: float = 0.50
    merges_allowed_through_iter: int = 2   # no merges from iter 3+

    # Cold start
    medoid_diversity_trigger: float = 0.40  # e5_diversity > this → outlier-heavy
    medoid_centroid_disagreement_trigger: float = 0.15

    # Absorber detection
    absorber_intake_thresh: float = 0.30
    absorber_strikes_to_flag: int = 2

    # Loop control
    max_iter: int = 8
    convergence_change_frac: float = 0.001  # < this fraction of rows changed → converged
    convergence_required_consec: int = 2
    move_budget: int = 3

    # Damping schedule (alpha applied to OLD centroid; new = (1-alpha)*fresh + alpha*old)
    damping_schedule: tuple[float, ...] = (0.5, 0.5, 0.3, 0.3, 0.15, 0.15, 0.0, 0.0)

    # Tag floor
    min_tag_rows: int = 3


# -------------------- Geometry primitives --------------------

def compute_trimmed_centroid(
    vectors: np.ndarray,
    trim_fraction: float = 0.05,
) -> np.ndarray:
    """Outlier-trimmed L2-normalized centroid.

    Two-pass per Stage 2: rough mean centroid, drop top/bottom `trim_fraction`
    by sim-to-rough-centroid, then mean of survivors.

    Returns a (D,) float32 unit vector.
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute centroid for empty vectors")
    rough = _centroid_trimmed_input(vectors)  # mean + L2-normalize
    if len(vectors) < 10 or trim_fraction <= 0:
        return rough
    sims = vectors @ rough
    keep = _trimmed_indices(sims, trim_fraction)
    if len(keep) == 0:
        return rough
    return _centroid_trimmed_input(vectors[keep])


def compute_medoid_index(vectors: np.ndarray) -> int:
    """Index of the row whose mean cosine to all other rows is highest.

    For cold-start of outlier-heavy tags: medoid is more robust than the
    trimmed mean when the tag is contaminated.
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute medoid for empty vectors")
    if len(vectors) == 1:
        return 0
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, 0.0)
    mean_sims = sim.mean(axis=1)
    return int(np.argmax(mean_sims))


def medoid_centered_centroid(
    vectors: np.ndarray,
    core_threshold: float = 0.85,
) -> np.ndarray:
    """Cold-start centroid: medoid + its core (rows with cos > threshold to medoid).

    Used when a tag is outlier-heavy; the trimmed mean would chase contamination,
    but the medoid finds the densest local cluster and we average only its core.

    Returns L2-normalized (D,) float32. Falls back to trimmed mean if core is too small.
    """
    if len(vectors) < 5:
        return compute_trimmed_centroid(vectors)
    medoid_idx = compute_medoid_index(vectors)
    medoid_vec = vectors[medoid_idx]
    sims_to_medoid = vectors @ medoid_vec
    core_mask = sims_to_medoid >= core_threshold
    if core_mask.sum() < 3:
        return compute_trimmed_centroid(vectors)
    return _centroid_trimmed_input(vectors[core_mask])


def is_outlier_heavy(
    vectors: np.ndarray,
    config: RepairConfig,
) -> bool:
    """Check whether a tag should use cold-start medoid bootstrapping.

    Two triggers (either fires):
    - e5_diversity (1 - mean cos to centroid) > config.medoid_diversity_trigger
    - cos(medoid, trimmed_centroid) < 1 - config.medoid_centroid_disagreement_trigger
    """
    if len(vectors) < 5:
        return False
    trimmed = compute_trimmed_centroid(vectors)
    diversity = float(1.0 - (vectors @ trimmed).mean())
    if diversity > config.medoid_diversity_trigger:
        return True
    medoid_idx = compute_medoid_index(vectors)
    medoid_vec = vectors[medoid_idx]
    disagreement = float(1.0 - (medoid_vec @ trimmed))
    return disagreement > config.medoid_centroid_disagreement_trigger


def recompute_centroids(
    state: RepairState,
    config: RepairConfig,
    only_tag_indices: list[int] | None = None,
) -> None:
    """Recompute centroids in-place for either all alive tags or a subset.

    Cold-start path is used at iter 0 (when previous_centroids is all zeros).
    Subsequent iterations use trimmed mean (cold-start makes outlier-heavy
    tags trust their initial core; once stable, switch to trimmed mean).

    Damping: at iter k, new = (1-alpha)*fresh + alpha*previous, then re-normalize.
    Skipped if previous is all zeros (iter 0).
    """
    schedule = config.damping_schedule
    alpha = schedule[min(state.iter, len(schedule) - 1)] if state.iter > 0 else 0.0

    targets = (
        np.asarray(only_tag_indices, dtype=np.int64)
        if only_tag_indices is not None
        else state.alive_indices()
    )
    if len(targets) == 0:
        return

    state.previous_centroids[targets] = state.current_centroids[targets]

    cold_start = state.iter == 0
    for tag_idx in targets:
        rows = state.rows_in_tag(int(tag_idx))
        if len(rows) == 0:
            # Empty alive tag — keep the previous centroid (will dissolve later).
            continue
        vecs = state.embeddings[rows]
        if cold_start and is_outlier_heavy(vecs, config):
            fresh = medoid_centered_centroid(vecs)
        else:
            fresh = compute_trimmed_centroid(vecs)
        if alpha > 0.0 and state.previous_centroids[tag_idx].any():
            blended = (1.0 - alpha) * fresh + alpha * state.previous_centroids[tag_idx]
            blended = blended.astype(np.float32)
            norm = float(np.linalg.norm(blended))
            if norm > 0:
                blended = blended / norm
            state.current_centroids[tag_idx] = blended
        else:
            state.current_centroids[tag_idx] = fresh


def compute_geometry(state: RepairState) -> dict[str, np.ndarray]:
    """Vectorized per-row geometry over kept rows.

    Returns dict with keys:
      own_sim:     (N,) cosine of row to centroid of its current tag (NaN for dropped)
      best_other:  (N,) tag index of nearest OTHER alive tag (-1 if no alternative)
      best_other_sim: (N,) cosine to that best-other centroid (-inf if none)
      margin:      (N,) own_sim - best_other_sim (NaN if no best-other)

    Dead/dropped rows get NaN/-1 placeholders so callers must mask via state.kept_row_mask().
    """
    n = state.n_rows
    kept = state.kept_row_mask()

    own_sim = np.full(n, np.nan, dtype=np.float32)
    best_other = np.full(n, -1, dtype=np.int32)
    best_other_sim = np.full(n, -np.inf, dtype=np.float32)
    margin = np.full(n, np.nan, dtype=np.float32)

    alive_idx = state.alive_indices()
    if len(alive_idx) == 0:
        return {"own_sim": own_sim, "best_other": best_other, "best_other_sim": best_other_sim, "margin": margin}

    alive_centroids = state.current_centroids[alive_idx]
    # (N, T_alive) cosine matrix; we only need rows that are kept, but compute all for vectorization simplicity.
    scores = state.embeddings @ alive_centroids.T  # (N, T_alive)

    # Map alive position back to global tag index.
    alive_pos_of: dict[int, int] = {int(t): i for i, t in enumerate(alive_idx)}

    # own_sim
    own_pos = np.array(
        [alive_pos_of.get(int(state.assignment[i]), -1) for i in range(n)],
        dtype=np.int64,
    )
    has_alive_own = own_pos >= 0
    rows_with_alive_own = np.where(has_alive_own & kept)[0]
    if len(rows_with_alive_own) > 0:
        own_sim[rows_with_alive_own] = scores[rows_with_alive_own, own_pos[rows_with_alive_own]]

    # best_other: argmax over scores with own column masked out
    if len(alive_idx) > 1:
        masked = scores.copy()
        for i in rows_with_alive_own:
            masked[i, own_pos[i]] = -np.inf
        best_pos = np.argmax(masked, axis=1)
        best_other[rows_with_alive_own] = alive_idx[best_pos[rows_with_alive_own]].astype(np.int32)
        best_other_sim[rows_with_alive_own] = masked[rows_with_alive_own, best_pos[rows_with_alive_own]]
        # finite where own AND best both exist
        finite_both = rows_with_alive_own[
            np.isfinite(best_other_sim[rows_with_alive_own])
            & np.isfinite(own_sim[rows_with_alive_own])
        ]
        margin[finite_both] = own_sim[finite_both] - best_other_sim[finite_both]

    return {
        "own_sim": own_sim,
        "best_other": best_other,
        "best_other_sim": best_other_sim,
        "margin": margin,
    }


# -------------------- Per-row kNN tag-share (step 3) --------------------

def compute_knn_tag_share(
    state: RepairState,
    top_k: int = 10,
) -> np.ndarray:
    """For each (kept) row r, the distribution over current tags of r's top-K
    nearest neighbors (excluding self).

    Uses FAISS IndexFlatIP for exact top-K search — much faster + lower
    memory than chunked numpy at corpus scale.

    Returns (N, T) float32 — sums to 1.0 along axis 1 for kept rows; all
    zeros for dropped rows. Merged tags are resolved to canonical so a row
    whose neighbor lives in a "merged-into" tag attributes share to the
    canonical.
    """
    import faiss

    n = state.n_rows
    t = state.n_tags
    share = np.zeros((n, t), dtype=np.float32)

    kept = state.kept_row_mask()
    kept_idx = np.where(kept)[0]
    if len(kept_idx) < 2:
        return share

    kept_emb = np.ascontiguousarray(state.embeddings[kept_idx], dtype=np.float32)
    kept_assign = state.assignment[kept_idx]
    canon_per_kept = np.fromiter(
        (state._resolve_canonical(int(tg)) for tg in kept_assign),
        dtype=np.int32,
        count=len(kept_assign),
    )

    # Search top_k+1 to leave room to drop self. For unit-norm vectors the
    # self hit is always at position 0 (cos(x,x)=1 strictly maximal among unit
    # vectors), so we just take columns 1..top_k+1. Stage 0 dedup removed
    # within-tag exact duplicates, so this assumption holds for our corpus.
    actual_k = min(top_k + 1, len(kept_idx))
    if actual_k <= 1:
        return share
    index = faiss.IndexFlatIP(kept_emb.shape[1])
    index.add(kept_emb)
    _, neighbor_kept_pos = index.search(kept_emb, actual_k)
    cleaned = neighbor_kept_pos[:, 1:actual_k]  # (kept, top_k) — self stripped
    effective_k = cleaned.shape[1]
    if effective_k == 0:
        return share

    # Vectorized scatter: kept_idx[i] receives 1/K share for each tag in
    # canon_per_kept[cleaned[i]]
    inc = np.float32(1.0 / effective_k)
    neighbor_canon = canon_per_kept[cleaned]  # (kept, K)
    rows_flat = np.repeat(kept_idx, effective_k)
    cols_flat = neighbor_canon.reshape(-1)
    np.add.at(share, (rows_flat, cols_flat), inc)
    return share


# -------------------- Initialization helpers (used by the loop driver) --------------------

def initialize_state(
    state: RepairState,
    config: RepairConfig,
) -> None:
    """Iter 0 setup: cold-start centroids + freeze them + flag outlier-heavy tags.

    After this returns:
      - state.current_centroids and state.frozen_centroids both populated
      - state.frozen_pairwise_cosine populated
      - state.untrusted_tags contains outlier-heavy tag indices
      - state.iter still == 0 (caller increments before iter 1)
    """
    state.iter = 0
    recompute_centroids(state, config)
    state.frozen_centroids = state.current_centroids.copy()
    alive = state.alive_indices()
    if len(alive) > 0:
        # Pairwise cosine matrix over alive tags only (others stay at 0.0)
        ac = state.frozen_centroids[alive]
        state.frozen_pairwise_cosine = np.zeros((state.n_tags, state.n_tags), dtype=np.float32)
        sub = ac @ ac.T
        for i, ti in enumerate(alive):
            for j, tj in enumerate(alive):
                state.frozen_pairwise_cosine[ti, tj] = sub[i, j]

    # Flag outlier-heavy tags as untrusted for iter 1
    for tag_idx in state.alive_indices():
        rows = state.rows_in_tag(int(tag_idx))
        if len(rows) >= 5 and is_outlier_heavy(state.embeddings[rows], config):
            state.untrusted_tags.add(int(tag_idx))


# -------------------- Phase A: Reassignment (step 4) --------------------

def _knn_share_for_subset(
    state: RepairState,
    candidate_row_indices: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Compute kNN tag-share for ONLY a subset of rows (the move-candidate set).

    Returns (len(candidates), T) — share aggregated by canonical tag.

    Cheap: at corpus scale this scales linearly with candidate count, not N²
    like the all-rows variant. Used inside propose_reassignments so we only
    pay for rows that have already passed the cheaper margin/gap gates.
    """
    import faiss

    t = state.n_tags
    out = np.zeros((len(candidate_row_indices), t), dtype=np.float32)
    if len(candidate_row_indices) == 0:
        return out
    kept = state.kept_row_mask()
    kept_idx = np.where(kept)[0]
    if len(kept_idx) < 2:
        return out

    kept_emb = np.ascontiguousarray(state.embeddings[kept_idx], dtype=np.float32)
    kept_assign = state.assignment[kept_idx]
    canon_per_kept = np.fromiter(
        (state._resolve_canonical(int(tg)) for tg in kept_assign),
        dtype=np.int32,
        count=len(kept_assign),
    )

    actual_k = min(top_k + 1, len(kept_idx))
    if actual_k <= 1:
        return out

    index = faiss.IndexFlatIP(kept_emb.shape[1])
    index.add(kept_emb)

    # Build query matrix from candidate global indices
    cand_emb = np.ascontiguousarray(state.embeddings[candidate_row_indices], dtype=np.float32)
    _, neighbor_kept_pos = index.search(cand_emb, actual_k)

    # Each candidate's first hit is itself iff it's also in kept_idx (it should
    # be — candidates are kept rows). For unit norm vectors self has the
    # highest IP, so column 0 is self.
    cleaned = neighbor_kept_pos[:, 1:actual_k]  # (n_candidates, top_k)
    effective_k = cleaned.shape[1]
    if effective_k == 0:
        return out
    inc = np.float32(1.0 / effective_k)
    neighbor_canon = canon_per_kept[cleaned]  # (n_candidates, K)
    cand_rows_flat = np.repeat(np.arange(len(candidate_row_indices)), effective_k)
    cols_flat = neighbor_canon.reshape(-1)
    np.add.at(out, (cand_rows_flat, cols_flat), inc)
    return out


def propose_reassignments(
    state: RepairState,
    geom: dict[str, np.ndarray],
    config: RepairConfig,
    knn_share: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Build the candidate move list for this iteration.

    Returns list of (row_idx, from_tag, to_tag). Pure-geometry gates only.

    Two-stage filtering for performance:
      1. Cheap gates: margin, gap, move-budget, hysteresis, untrusted target
      2. Expensive gate (kNN-majority): computed only for candidates that
         passed stage 1, via FAISS search restricted to the candidate set.

    `knn_share` may be passed in (precomputed all-rows tag-share) — used by
    unit tests and by simpler synthetic flows. At corpus scale, leave it
    as None and the function will compute per-candidate shares lazily.
    """
    n = state.n_rows
    kept = state.kept_row_mask()
    margin = geom["margin"]
    best_other = geom["best_other"]
    best_other_sim = geom["best_other_sim"]
    own_sim = geom["own_sim"]

    # Stage 1: cheap gates → produce a filtered candidate list
    cheap_candidates: list[tuple[int, int, int, float]] = []  # row_idx, from, to, gap
    for row_idx in range(n):
        if not kept[row_idx]:
            continue
        if state.moves_remaining[row_idx] <= 0:
            continue
        cur = int(state.assignment[row_idx])
        bo = int(best_other[row_idx])
        if bo < 0 or bo == cur:
            continue
        if not np.isfinite(margin[row_idx]):
            continue
        if margin[row_idx] >= config.reassign_thresh:
            continue
        gap = float(best_other_sim[row_idx] - own_sim[row_idx])
        required_gap = config.delta_move
        if int(state.prior_tag[row_idx]) == bo:
            required_gap += config.delta_hyst
        if gap < required_gap:
            continue
        if bo in state.untrusted_tags:
            continue
        cheap_candidates.append((row_idx, cur, bo, gap))

    if not cheap_candidates:
        return []

    # Stage 2: expensive kNN-majority gate, only on rows that passed stage 1
    if knn_share is None:
        cand_row_indices = np.array([c[0] for c in cheap_candidates], dtype=np.int64)
        per_cand_share = _knn_share_for_subset(state, cand_row_indices, top_k=config.knn_top_k)
    else:
        per_cand_share = None  # use the passed-in all-rows share

    candidates: list[tuple[int, int, int, float]] = []
    for cand_pos, (row_idx, cur, bo, gap) in enumerate(cheap_candidates):
        bo_canon = state._resolve_canonical(bo)
        cur_canon = state._resolve_canonical(cur)
        if knn_share is not None:
            share_at_bo = float(knn_share[row_idx, bo_canon])
            share_at_cur = float(knn_share[row_idx, cur_canon])
        else:
            share_at_bo = float(per_cand_share[cand_pos, bo_canon])
            share_at_cur = float(per_cand_share[cand_pos, cur_canon])
        if share_at_bo <= share_at_cur:
            continue
        candidates.append((row_idx, cur, bo, gap))

    # Sort candidates by gap descending so the strongest moves get priority
    candidates.sort(key=lambda x: -x[3])

    # Apply intake caps per destination tag
    intake_so_far: dict[int, int] = {}
    moves: list[tuple[int, int, int]] = []
    for row_idx, cur, bo, _gap in candidates:
        bo_size = len(state.rows_in_tag(bo))
        if bo_size == 0:
            cap = max(1, config.min_tag_rows)
        else:
            cap = max(1, int(np.ceil(bo_size * config.intake_cap_per_tag)))
        if intake_so_far.get(bo, 0) >= cap:
            continue
        intake_so_far[bo] = intake_so_far.get(bo, 0) + 1
        moves.append((row_idx, cur, bo))
    return moves


def apply_reassignments(
    state: RepairState,
    moves: list[tuple[int, int, int]],
    config: RepairConfig,
) -> dict[str, Any]:
    """Apply reassignment moves and recompute affected centroids.

    Returns telemetry: counts, per-tag intake, list of (row_id, from, to) for ops.jsonl.
    """
    if not moves:
        return {"reassigned": 0, "ops": []}

    affected_tags: set[int] = set()
    ops_log = []
    for row_idx, cur, bo in moves:
        state.prior_tag[row_idx] = cur
        state.assignment[row_idx] = bo
        state.moves_remaining[row_idx] -= 1
        # Note: status stays STATUS_KEPT until terminal (drop or merged_into).
        # The status doesn't track reassignment in the canonical view; that's
        # captured by `assignment != original_assignment` in the final frame.
        affected_tags.add(cur)
        affected_tags.add(bo)
        ops_log.append({
            "op": "reassign",
            "row_id": int(state.row_ids[row_idx]),
            "from_tag": state.tag_names[cur],
            "to_tag": state.tag_names[bo],
        })

    # Recompute centroids for affected tags only (with damping per state.iter)
    recompute_centroids(state, config, only_tag_indices=list(affected_tags))

    return {"reassigned": len(moves), "affected_tags": list(affected_tags), "ops": ops_log}


# -------------------- Phase B: Drop / triage (steps 5-6) --------------------

def find_cross_tag_near_dups(
    state: RepairState,
    config: RepairConfig,
) -> list[tuple[int, int]]:
    """Detect (row_a, row_b) pairs in different tags with cosine > threshold.

    Uses FAISS range_search (find all neighbors above a similarity threshold)
    for memory efficiency at corpus scale. Each pair is reported once
    (the lower kept-position as `row_a`).
    """
    import faiss

    kept = state.kept_row_mask()
    kept_idx = np.where(kept)[0]
    if len(kept_idx) < 2:
        return []
    kept_emb = np.ascontiguousarray(state.embeddings[kept_idx], dtype=np.float32)
    kept_assign = state.assignment[kept_idx]
    canon_per_kept = np.fromiter(
        (state._resolve_canonical(int(tg)) for tg in kept_assign),
        dtype=np.int32,
        count=len(kept_assign),
    )
    thresh = config.cross_tag_dup_thresh

    index = faiss.IndexFlatIP(kept_emb.shape[1])
    index.add(kept_emb)
    # range_search returns CSR-style (lims, dists, ids): for each query i,
    # neighbors with IP > thresh are at positions lims[i]..lims[i+1].
    lims, dists, ids = index.range_search(kept_emb, thresh)
    pairs: list[tuple[int, int]] = []
    n_kept = len(kept_idx)
    for i in range(n_kept):
        start = int(lims[i])
        stop = int(lims[i + 1])
        if stop <= start:
            continue
        for off in range(start, stop):
            j = int(ids[off])
            if j <= i:  # self or already-reported pair
                continue
            if canon_per_kept[i] == canon_per_kept[j]:
                continue
            pairs.append((int(kept_idx[i]), int(kept_idx[j])))
    return pairs


def triage_cross_tag_pair(
    state: RepairState,
    geom: dict[str, np.ndarray],
    row_a: int,
    row_b: int,
    config: RepairConfig,
) -> tuple[str, list[int]]:
    """Decide what to do with a cross-tag near-duplicate pair.

    Returns (case_label, rows_to_drop).
      Case 1 (winner_take): one row has clearly higher own_sim → drop the loser.
      Case 2 (drop_both): no clear winner → drop both (last resort).

    Phase C (merges) runs BEFORE Phase B in the loop, so by the time we
    triage a cross-tag pair, any merge that was going to happen already
    happened — and the pair's tags would have collapsed into one canonical
    (in which case `find_cross_tag_near_dups` would not have flagged it
    this iter, since it filters by canonical-tag identity).

    Therefore any pair that reaches triage has tags that did NOT merge.
    No more "merge_defer" — winner-take or drop-both, decisively.
    """
    own_a = geom["own_sim"][row_a]
    own_b = geom["own_sim"][row_b]
    if np.isfinite(own_a) and np.isfinite(own_b):
        diff = abs(float(own_a - own_b))
        if diff > 0.05:
            loser = row_a if own_a < own_b else row_b
            return "winner_take", [loser]
    return "drop_both", [row_a, row_b]


def apply_drops_phase_b(
    state: RepairState,
    geom: dict[str, np.ndarray],
    config: RepairConfig,
    cap_strikes: dict[int, int],
) -> dict[str, Any]:
    """Phase B: cross-tag dup triage + hard-margin drops + per-tag drop caps.

    Returns telemetry dict with counts and ops.

    Caps: per-tag drop count limited to drop_cap_per_tag * tag_size per iter.
    Updates cap_strikes (passed in) — 3 consecutive cap-hits on a tag aborts.
    Strikes accumulate across iterations; reset to 0 on any iter where the
    tag did NOT hit cap.
    """
    ops: list[dict] = []
    drops_per_tag: dict[int, int] = {}
    drop_set: set[int] = set()

    # 1. Cross-tag near-dup triage — Phase C (merges) ran first; remaining
    #    cross-tag pairs are decisively winner-take or drop-both.
    cross_dup_pairs = find_cross_tag_near_dups(state, config)
    for row_a, row_b in cross_dup_pairs:
        if row_a in drop_set or row_b in drop_set:
            continue
        case, to_drop = triage_cross_tag_pair(state, geom, row_a, row_b, config)
        for r in to_drop:
            drop_set.add(r)
            ops.append({
                "op": "drop",
                "reason": f"cross_tag_dup_{case}",
                "row_id": int(state.row_ids[r]),
                "tag": state.tag_names[int(state.assignment[r])],
            })

    # 2. Hard-margin drops
    margin = geom["margin"]
    kept = state.kept_row_mask()
    for r in range(state.n_rows):
        if not kept[r] or r in drop_set:
            continue
        if not np.isfinite(margin[r]):
            continue
        if margin[r] < config.hard_drop_thresh:
            drop_set.add(r)
            ops.append({
                "op": "drop",
                "reason": "hard_margin",
                "row_id": int(state.row_ids[r]),
                "tag": state.tag_names[int(state.assignment[r])],
                "margin": float(margin[r]),
            })

    # 3. Apply per-tag drop caps; defer excess
    final_drops: list[int] = []
    deferred_for_cap: list[tuple[int, dict]] = []
    # Group drops by tag and sort within tag by margin (most negative first)
    by_tag: dict[int, list[int]] = {}
    for r in drop_set:
        tag = int(state.assignment[r])
        by_tag.setdefault(tag, []).append(r)
    for tag, rs in by_tag.items():
        tag_size = len(state.rows_in_tag(tag)) + len([x for x in rs if state.assignment[x] == tag])
        cap = max(1, int(np.ceil(tag_size * config.drop_cap_per_tag)))
        # sort drops by margin ascending (most negative first)
        rs_sorted = sorted(rs, key=lambda r: margin[r] if np.isfinite(margin[r]) else 0.0)
        kept_count = min(len(rs_sorted), cap)
        final_drops.extend(rs_sorted[:kept_count])
        if len(rs_sorted) > cap:
            for r in rs_sorted[cap:]:
                deferred_for_cap.append((r, {"op": "drop_deferred", "reason": "drop_cap", "tag": state.tag_names[tag]}))
            drops_per_tag[tag] = kept_count
            cap_strikes[tag] = cap_strikes.get(tag, 0) + 1
        else:
            cap_strikes.pop(tag, None)
            drops_per_tag[tag] = kept_count

    # Apply final drops
    for r in final_drops:
        state.status[r] = STATUS_DROPPED

    # Record deferred-cap ops for telemetry (not actually dropped this iter)
    for r, op in deferred_for_cap:
        op["row_id"] = int(state.row_ids[r])
        ops.append(op)

    # Strikes: any tag with cap_strikes >= drop_cap_strikes is the abort signal
    aborting_tags = [t for t, n in cap_strikes.items() if n >= config.drop_cap_strikes]
    if aborting_tags:
        return {
            "dropped": len(final_drops),
            "deferred_for_cap": len(deferred_for_cap),
            "ops": ops,
            "abort_tags": [state.tag_names[t] for t in aborting_tags],
        }

    return {
        "dropped": len(final_drops),
        "deferred_for_cap": len(deferred_for_cap),
        "ops": ops,
    }


# -------------------- Phase C: Merge (steps 9-10) --------------------

def compute_mutual_directed_confusion(
    state: RepairState,
) -> np.ndarray:
    """For each pair of alive tags (A, B): the fraction of A's rows whose
    nearest OTHER-TAG centroid is B.

    Returns a (T, T) float32 matrix where row A column B = fraction of A's
    rows whose argmax-other-centroid lands on B. Diagonal is 0 (a tag's
    rows are never "confused with themselves" in this metric).
    """
    T = state.n_tags
    confusion = np.zeros((T, T), dtype=np.float32)
    alive_idx = state.alive_indices()
    if len(alive_idx) < 2:
        return confusion
    alive_centroids = state.current_centroids[alive_idx]
    alive_pos_of = {int(t): i for i, t in enumerate(alive_idx)}
    kept = state.kept_row_mask()
    for tag_idx in alive_idx:
        rows = state.rows_in_tag(int(tag_idx))
        if len(rows) == 0:
            continue
        scores = state.embeddings[rows] @ alive_centroids.T  # (n_rows_in_tag, T_alive)
        # Mask own column
        own_pos = alive_pos_of[int(tag_idx)]
        scores[:, own_pos] = -np.inf
        best_pos = np.argmax(scores, axis=1)
        unique, counts = np.unique(best_pos, return_counts=True)
        for pos, count in zip(unique, counts):
            other_tag = int(alive_idx[pos])
            confusion[int(tag_idx), other_tag] = count / len(rows)
    return confusion


def compute_merge_knn_overlap(
    state: RepairState,
    tag_a: int,
    tag_b: int,
    sample_cap: int = 50,
    k: int = 10,
) -> float:
    """Cross-tag kNN overlap: fraction of (A's top-k neighbors that live in B)
    + symmetrically, averaged.

    Mirrors cleaner.py:_knn_overlap shape but operates on RepairState (which
    handles the kept-row + canonical-tag bookkeeping).
    """
    rows_a = state.rows_in_tag(tag_a)
    rows_b = state.rows_in_tag(tag_b)
    if len(rows_a) == 0 or len(rows_b) == 0:
        return 0.0
    rng = np.random.RandomState(0)
    if len(rows_a) > sample_cap:
        rows_a = rng.choice(rows_a, size=sample_cap, replace=False)
    if len(rows_b) > sample_cap:
        rows_b = rng.choice(rows_b, size=sample_cap, replace=False)
    # Combined indices into embeddings
    all_idx = np.concatenate([rows_a, rows_b])
    a_set = set(int(x) for x in rows_a)
    combined_emb = state.embeddings[all_idx]
    sims = combined_emb @ combined_emb.T
    np.fill_diagonal(sims, -np.inf)
    # For each row in A, count how many of its top-k are in B (and vice versa)
    actual_k = min(k, len(all_idx) - 1)
    if actual_k <= 0:
        return 0.0
    top = np.argpartition(-sims, kth=actual_k - 1, axis=1)[:, :actual_k]
    overlap_count = 0
    total = 0
    for local_i, global_row in enumerate(all_idx):
        is_in_a = int(global_row) in a_set
        for nbr_pos in top[local_i]:
            nbr_global = int(all_idx[nbr_pos])
            nbr_in_a = nbr_global in a_set
            # Cross-tag neighbor counts
            if is_in_a != nbr_in_a:
                overlap_count += 1
            total += 1
    return overlap_count / total if total > 0 else 0.0


def propose_merges(
    state: RepairState,
    config: RepairConfig,
    deferred_pairs: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Find tag pairs that meet ALL merge gates.

    Gates (pure geometry, no answers):
      - cos(centroid_a, centroid_b) > merge_cosine_thresh
      - knn_overlap(rows_a, rows_b) > merge_knn_overlap_thresh
      - mutual_confusion (A→B AND B→A) > mutual_confusion_thresh
      - hysteresis: iter-0 frozen pairwise A↔B cos also exceeded threshold

    Returns list of (tag_a, tag_b) where tag_a is the canonical-of-merge.
    """
    if state.iter > config.merges_allowed_through_iter:
        return []
    alive = state.alive_indices()
    if len(alive) < 2:
        return []
    confusion = compute_mutual_directed_confusion(state)
    # Iterate over alive tag pairs
    proposals: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, ti in enumerate(alive):
        for tj in alive[i + 1:]:
            ti, tj = int(ti), int(tj)
            pair = (min(ti, tj), max(ti, tj))
            if pair in seen:
                continue
            seen.add(pair)
            cc = float(state.current_centroids[ti] @ state.current_centroids[tj])
            if cc < config.merge_cosine_thresh:
                continue
            # Hysteresis predicate: iter-0 frozen pairwise also exceeded threshold
            if state.frozen_pairwise_cosine[ti, tj] < config.merge_cosine_thresh:
                continue
            # Mutual directed confusion
            if confusion[ti, tj] < config.mutual_confusion_thresh:
                continue
            if confusion[tj, ti] < config.mutual_confusion_thresh:
                continue
            # kNN overlap
            knn_overlap = compute_merge_knn_overlap(state, ti, tj)
            if knn_overlap < config.merge_knn_overlap_thresh:
                continue
            # Pick canonical via geometric composite
            canonical, other = _pick_canonical(state, ti, tj)
            proposals.append((canonical, other))
    return proposals


def _pick_canonical(state: RepairState, tag_a: int, tag_b: int) -> tuple[int, int]:
    """Return (canonical, other) per geometric composite score.

    score = row_count + 100*neighborhood_support - 50*medoid_centroid_disagreement
    Tie-break: lower index wins (deterministic).
    """
    def score(t: int) -> tuple[float, int]:
        rows = state.rows_in_tag(t)
        n = len(rows)
        if n == 0:
            return (-1.0, t)
        vecs = state.embeddings[rows]
        centroid_v = state.current_centroids[t]
        sims = vecs @ centroid_v
        # neighborhood_support: mean cos in top quartile
        cutoff = max(1, n // 4)
        top_sims = np.partition(sims, max(0, n - cutoff))[-cutoff:]
        nbr_support = float(top_sims.mean())
        # medoid disagreement
        medoid_idx = int(rows[compute_medoid_index(vecs)])
        medoid_v = state.embeddings[medoid_idx]
        disagreement = float(1.0 - (medoid_v @ centroid_v))
        composite = n + 100.0 * nbr_support - 50.0 * disagreement
        return (composite, -t)  # negate t for stable tiebreak (lower idx = larger -t = wins ties)

    s_a, t_a_neg = score(tag_a)
    s_b, t_b_neg = score(tag_b)
    if (s_a, t_a_neg) >= (s_b, t_b_neg):
        return tag_a, tag_b
    return tag_b, tag_a


def apply_merges(
    state: RepairState,
    proposals: list[tuple[int, int]],
    config: RepairConfig,
) -> dict[str, Any]:
    """Execute merges: reassign all rows of `other` to `canonical`, mark
    `other` as dead in alive_tags, and set canonical_of[other] = canonical.

    Recomputes the canonical's centroid after rows arrive.
    """
    if not proposals:
        return {"merged": 0, "ops": []}
    ops = []
    affected = set()
    for canonical, other in proposals:
        # In case a chain forms (we proposed (A,B) and (A,C) in the same iter),
        # resolve via canonical_of so we don't double-apply.
        canonical = state._resolve_canonical(canonical)
        other = state._resolve_canonical(other)
        if canonical == other:
            continue
        # Move all rows in `other` to `canonical`
        rows_in_other = state.rows_in_tag(other)
        for r in rows_in_other:
            state.assignment[r] = canonical
        state.canonical_of[other] = canonical
        state.alive_tags[other] = False
        affected.add(canonical)
        ops.append({
            "op": "merge",
            "from_tag": state.tag_names[other],
            "into_tag": state.tag_names[canonical],
            "rows_moved": int(len(rows_in_other)),
        })
    # Recompute affected canonical centroids
    if affected:
        recompute_centroids(state, config, only_tag_indices=list(affected))
    return {"merged": len(ops), "ops": ops}


# -------------------- Phase D: Absorber detection (step 11) --------------------

def update_absorber_flags(
    state: RepairState,
    intake_per_tag: dict[int, int],
    config: RepairConfig,
) -> list[int]:
    """Update absorber_history; flag tags as untrusted_absorber when they
    receive >= absorber_intake_thresh of their size for absorber_strikes_to_flag
    consecutive iters.

    Returns list of newly-flagged tag indices.
    """
    newly_flagged: list[int] = []
    for tag_idx in state.alive_indices():
        rows = state.rows_in_tag(int(tag_idx))
        size = len(rows)
        if size == 0:
            continue
        intake = intake_per_tag.get(int(tag_idx), 0)
        intake_frac = intake / size if size > 0 else 0.0
        history = state.absorber_history.setdefault(int(tag_idx), [])
        history.append(intake_frac)
        # Trim history to last N iterations
        if len(history) > config.absorber_strikes_to_flag + 2:
            history.pop(0)
        # Check last N iters for consistent high intake
        recent = history[-config.absorber_strikes_to_flag:]
        if (
            len(recent) >= config.absorber_strikes_to_flag
            and all(x >= config.absorber_intake_thresh for x in recent)
        ):
            if int(tag_idx) not in state.untrusted_tags:
                state.untrusted_tags.add(int(tag_idx))
                # Replace centroid with iter-0 frozen centroid
                state.current_centroids[tag_idx] = state.frozen_centroids[tag_idx].copy()
                newly_flagged.append(int(tag_idx))
        elif int(tag_idx) in state.untrusted_tags:
            # Recovery: if recent intake dropped below threshold, allow tag back as target
            if len(recent) >= 1 and recent[-1] < (config.absorber_intake_thresh / 3):
                state.untrusted_tags.discard(int(tag_idx))
    return newly_flagged


# -------------------- Loop driver (step 12) --------------------

def run_repair_loop(
    state: RepairState,
    config: RepairConfig,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Main loop. Mutates state in place. Writes per-iter ops + metrics if
    artifact_dir is provided.

    Returns final telemetry dict.
    """
    initialize_state(state, config)

    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = artifact_dir / "iter_metrics.jsonl"
        metrics_path.unlink(missing_ok=True)

    cap_strikes: dict[int, int] = {}
    consec_low_change = 0
    consec_neg_margin_rise = 0
    prev_pct_neg_margin = None
    final_status = "running"

    for it in range(1, config.max_iter + 1):
        state.iter = it
        geom = compute_geometry(state)

        # Phase A: Reassign — propose_reassignments computes per-candidate
        # kNN-share lazily (only for rows that passed margin/gap gates).
        moves = propose_reassignments(state, geom, config)
        intake_per_tag: dict[int, int] = {}
        for _, _, bo in moves:
            intake_per_tag[bo] = intake_per_tag.get(bo, 0) + 1
        reassign_telemetry = apply_reassignments(state, moves, config)

        # Phase C: Merge tags FIRST — so cross-tag near-dups whose tags merge
        # disappear from the pair-set Phase B sees. Without this ordering,
        # most pairs would get stuck deferred to merge that never executes.
        merge_telemetry = {"merged": 0, "ops": []}
        if it <= config.merges_allowed_through_iter:
            geom = compute_geometry(state)
            merge_proposals = propose_merges(state, config)
            merge_telemetry = apply_merges(state, merge_proposals, config)

        # Re-fetch geometry after reassignments + merges so Phase B sees
        # the latest tag layout.
        geom = compute_geometry(state)

        # Phase B: Triage cross-tag dups + hard-margin drops
        drop_telemetry = apply_drops_phase_b(state, geom, config, cap_strikes)
        if "abort_tags" in drop_telemetry:
            final_status = "abort_drop_cap"
            _emit_metrics(state, artifact_dir, it, reassign_telemetry, drop_telemetry, merge_telemetry, {}, prev_pct_neg_margin)
            break

        # Phase D: Absorber detection + convergence
        newly_flagged = update_absorber_flags(state, intake_per_tag, config)
        absorber_telemetry = {"newly_flagged": [state.tag_names[t] for t in newly_flagged]}

        # Compute convergence signals
        geom = compute_geometry(state)
        kept = state.kept_row_mask()
        margin = geom["margin"]
        finite_margin = margin[kept & np.isfinite(margin)]
        pct_neg = float((finite_margin < 0).mean()) if len(finite_margin) > 0 else 0.0
        change_count = (
            reassign_telemetry["reassigned"]
            + drop_telemetry["dropped"]
            + merge_telemetry["merged"]
        )
        change_frac = change_count / max(1, state.n_rows)

        if change_frac < config.convergence_change_frac:
            consec_low_change += 1
        else:
            consec_low_change = 0

        if prev_pct_neg_margin is not None and pct_neg > prev_pct_neg_margin:
            consec_neg_margin_rise += 1
        else:
            consec_neg_margin_rise = 0

        _emit_metrics(state, artifact_dir, it, reassign_telemetry, drop_telemetry, merge_telemetry, absorber_telemetry, pct_neg)

        if consec_neg_margin_rise >= 2:
            final_status = "abort_oscillation"
            break

        if consec_low_change >= config.convergence_required_consec:
            final_status = "converged"
            break

        prev_pct_neg_margin = pct_neg

    if final_status == "running":
        final_status = "max_iter"

    # Dissolve tiny tags (below min_tag_rows) — set their rows' status to dropped
    # (only at the very end, to avoid mid-loop churn)
    for tag_idx in state.alive_indices():
        rows = state.rows_in_tag(int(tag_idx))
        if len(rows) > 0 and len(rows) < config.min_tag_rows:
            for r in rows:
                state.status[r] = STATUS_DROPPED
            state.alive_tags[tag_idx] = False

    return {
        "final_status": final_status,
        "iters_run": state.iter,
    }


def _emit_metrics(
    state: RepairState,
    artifact_dir: Path | None,
    it: int,
    reassign_t: dict,
    drop_t: dict,
    merge_t: dict,
    absorber_t: dict,
    pct_neg_margin: float | None,
) -> None:
    if artifact_dir is None:
        return
    record = {
        "iter": it,
        "n_rows_alive": int(state.kept_row_mask().sum()),
        "n_tags_alive": int(state.alive_tags.sum()),
        "reassigned": reassign_t.get("reassigned", 0),
        "dropped": drop_t.get("dropped", 0),
        "deferred_for_cap": drop_t.get("deferred_for_cap", 0),
        "merged": merge_t.get("merged", 0),
        "newly_flagged_absorbers": absorber_t.get("newly_flagged", []),
        "pct_neg_margin": pct_neg_margin,
        "n_untrusted": len(state.untrusted_tags),
    }
    with (artifact_dir / "iter_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Per-iter ops jsonl
    ops_path = artifact_dir / f"iter_{it:02d}_ops.jsonl"
    with ops_path.open("w", encoding="utf-8") as f:
        for op in reassign_t.get("ops", []):
            f.write(json.dumps(op, ensure_ascii=False) + "\n")
        for op in drop_t.get("ops", []):
            f.write(json.dumps(op, ensure_ascii=False) + "\n")
        for op in merge_t.get("ops", []):
            f.write(json.dumps(op, ensure_ascii=False) + "\n")


# -------------------- CLI entry point (step 13) --------------------

def run_repair(config: Any, resume: bool = True) -> None:
    """Entry point invoked via `tagclean repair --config ... --run-id ...`.

    Reads the cached stage1 embeddings + initial assignment from the run dir
    (or chains stage0/1 if missing), builds RepairState, runs the loop, and
    writes the deliverable to repair/xray_cleanup.csv.

    `config` is a CleanerConfig (typed via Any to avoid circular import at
    module load time).
    """
    from .cleaner import (  # local import to avoid circularity
        run_stage0,
        run_stage1,
        write_run_manifest,
    )

    run_dir = config.run_dir()
    repair_dir = run_dir / "repair"

    # Make sure stage0 + stage1 are done
    run_stage1(config, resume=resume)

    # Load embeddings + initial assignment
    emb_e5 = np.load(run_dir / "stage1" / "emb_e5.npy")
    emb_rows = pd.read_parquet(run_dir / "stage1" / "embedding_rows.parquet")
    # Filter to kept rows only (Stage 0 may have flagged some as not-keep)
    intake = pd.read_parquet(run_dir / "stage0" / "intake.parquet")
    # Stage 1's emb_e5 was built from intake[status=='keep']; embedding_rows.parquet
    # has those row_ids in order. We trust stage1's parquet directly.
    row_ids = emb_rows["row_id"].astype(np.int64).to_numpy()
    tags_per_row = emb_rows["tag"].astype(str).tolist()
    question_raw = emb_rows["question_raw"].astype(str).tolist()

    # Embeddings are already L2-normalized by stage1
    print(f"[repair] loaded {len(emb_e5)} rows × {emb_e5.shape[1]}d, {len(set(tags_per_row))} unique tags")

    repair_config = RepairConfig()
    state = RepairState.from_inputs(
        embeddings=emb_e5,
        row_ids=row_ids,
        question_raw=question_raw,
        tags_per_row=tags_per_row,
        move_budget=repair_config.move_budget,
    )

    print(f"[repair] starting loop (max_iter={repair_config.max_iter})")
    result = run_repair_loop(state, repair_config, artifact_dir=repair_dir)
    print(f"[repair] loop ended: status={result['final_status']}, iters_run={result['iters_run']}")

    # Write final artifacts
    repair_dir.mkdir(parents=True, exist_ok=True)
    final_frame = state.final_assignment_frame()
    final_frame.to_parquet(repair_dir / "final_assignment.parquet", index=False)

    # The deliverable: kept rows as 2-col question,tag CSV
    kept_frame = final_frame[final_frame["status"] == STATUS_KEPT].copy()
    out = kept_frame[["question", "repaired_tag"]].rename(columns={"repaired_tag": "tag"})
    out.to_csv(repair_dir / "xray_cleanup.csv", index=False)

    # Summary report
    n_in = len(final_frame)
    n_kept = len(kept_frame)
    n_dropped = int((final_frame["status"] == STATUS_DROPPED).sum())
    n_reassigned = int((final_frame["original_tag"] != final_frame["repaired_tag"]).sum())
    n_alive_tags = int(state.alive_tags.sum())
    n_input_tags = int(len(set(tags_per_row)))
    report = {
        "rows_in": n_in,
        "rows_kept": n_kept,
        "rows_dropped": n_dropped,
        "rows_reassigned_or_merged": n_reassigned,
        "tags_in": n_input_tags,
        "tags_alive_final": n_alive_tags,
        "loop_status": result["final_status"],
        "iters_run": result["iters_run"],
    }
    with (repair_dir / "repair_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[repair] {n_kept}/{n_in} rows kept; {n_alive_tags}/{n_input_tags} tags alive")
    print(f"[repair] wrote {repair_dir / 'xray_cleanup.csv'}")
    write_run_manifest(config, "repair")
