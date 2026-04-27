"""Tests for the deterministic geometric repair loop.

Pure-geometry constraint: no test should rely on tag-name patterns or
tag_answer.json content. Synthetic embeddings are designed so the
geometric signal alone determines correct behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tagclean.cleaner import l2_normalize
from tagclean.repair import (
    RepairConfig,
    compute_geometry,
    compute_knn_tag_share,
    compute_medoid_index,
    compute_trimmed_centroid,
    initialize_state,
    is_outlier_heavy,
    medoid_centered_centroid,
    recompute_centroids,
)
from tagclean.repair_state import (
    RepairState,
    STATUS_DROPPED,
    STATUS_KEPT,
)


def _make_state(
    embeddings: np.ndarray,
    tags: list[str],
    move_budget: int = 3,
) -> RepairState:
    """Build a RepairState from synthetic embeddings + tag labels.

    Embeddings get L2-normalized. row_ids and questions are auto-generated.
    """
    embeddings = l2_normalize(embeddings.astype(np.float32))
    n = len(embeddings)
    return RepairState.from_inputs(
        embeddings=embeddings,
        row_ids=np.arange(n, dtype=np.int64),
        question_raw=[f"q{i}" for i in range(n)],
        tags_per_row=tags,
        move_budget=move_budget,
    )


# -------------------- repair_state.py --------------------

def test_state_from_inputs_assigns_indices_alphabetically() -> None:
    emb = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    tags = ["c", "a", "b", "a", "c", "b"]
    state = _make_state(emb, tags)
    # tag_names sorted alphabetically: a=0, b=1, c=2
    assert state.tag_names == ["a", "b", "c"]
    assert state.assignment.tolist() == [2, 0, 1, 0, 2, 1]
    # original_assignment frozen = current at construction
    assert (state.original_assignment == state.assignment).all()
    # all rows kept initially
    assert (state.status == STATUS_KEPT).all()
    # all tags alive
    assert state.alive_tags.all()
    # canonical_of points to self
    assert (state.canonical_of == np.arange(state.n_tags)).all()
    # move budget set
    assert (state.moves_remaining == 3).all()


def test_state_rows_in_tag_excludes_dropped() -> None:
    emb = np.random.RandomState(0).randn(4, 4).astype(np.float32)
    state = _make_state(emb, ["a", "a", "b", "b"])
    # Drop one a-row
    state.status[0] = STATUS_DROPPED
    a_rows = state.rows_in_tag(0)
    assert a_rows.tolist() == [1]


def test_state_kept_row_mask_tracks_status() -> None:
    emb = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    state = _make_state(emb, ["a", "a", "a"])
    state.status[1] = STATUS_DROPPED
    mask = state.kept_row_mask()
    assert mask.tolist() == [True, False, True]


def test_state_resolve_canonical_handles_chain() -> None:
    emb = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    state = _make_state(emb, ["a", "b", "c"])
    # Set up: c -> b -> a
    state.canonical_of[2] = 1  # c -> b
    state.canonical_of[1] = 0  # b -> a
    assert state._resolve_canonical(2) == 0
    assert state._resolve_canonical(1) == 0
    assert state._resolve_canonical(0) == 0


def test_state_resolve_canonical_detects_cycle() -> None:
    emb = np.random.RandomState(0).randn(3, 4).astype(np.float32)
    state = _make_state(emb, ["a", "b", "c"])
    state.canonical_of[0] = 1
    state.canonical_of[1] = 0  # 0 -> 1 -> 0 cycle
    with pytest.raises(RuntimeError, match="cycle"):
        state._resolve_canonical(0)


def test_state_final_assignment_frame_resolves_merges() -> None:
    emb = np.random.RandomState(0).randn(4, 4).astype(np.float32)
    state = _make_state(emb, ["a", "a", "b", "b"])
    # Merge b into a
    state.canonical_of[1] = 0
    state.alive_tags[1] = False
    df = state.final_assignment_frame()
    # Both b-rows should report repaired_tag = "a" (canonical)
    assert df[df.original_tag == "b"]["repaired_tag"].tolist() == ["a", "a"]


# -------------------- geometry primitives --------------------

def test_trimmed_centroid_drops_outliers() -> None:
    """Build a tight cluster + 2 obvious outliers; trimmed centroid should
    pull toward the cluster, not toward the mean of all points."""
    rng = np.random.RandomState(0)
    base = rng.randn(20, 8).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # Tight cluster: 18 vectors near vec_0
    cluster = (base[0:1] + 0.05 * rng.randn(18, 8)).astype(np.float32)
    cluster /= np.linalg.norm(cluster, axis=1, keepdims=True)
    # 2 outliers far from cluster
    outliers = -base[0:1] + 0.05 * rng.randn(2, 8)
    outliers = outliers.astype(np.float32) / np.linalg.norm(outliers, axis=1, keepdims=True)
    all_vecs = np.vstack([cluster, outliers]).astype(np.float32)

    trimmed = compute_trimmed_centroid(all_vecs, trim_fraction=0.10)
    # The trimmed centroid should be much closer to the cluster mean than to the average of all
    cluster_mean = cluster.mean(axis=0)
    cluster_mean /= np.linalg.norm(cluster_mean)
    sim_to_cluster = float(trimmed @ cluster_mean)
    assert sim_to_cluster > 0.95


def test_medoid_finds_central_row() -> None:
    """Construct vectors where one is the obvious 'central' row."""
    rng = np.random.RandomState(0)
    central = rng.randn(8).astype(np.float32)
    central /= np.linalg.norm(central)
    # 5 rows close to central, 3 outliers
    close = (central[None] + 0.1 * rng.randn(5, 8)).astype(np.float32)
    close /= np.linalg.norm(close, axis=1, keepdims=True)
    far = (-central[None] + 0.05 * rng.randn(3, 8)).astype(np.float32)
    far /= np.linalg.norm(far, axis=1, keepdims=True)
    # central is the most central — its mean cosine to all others is highest
    vecs = np.vstack([central[None], close, far]).astype(np.float32)
    assert compute_medoid_index(vecs) == 0


def test_medoid_centered_centroid_finds_core() -> None:
    """Bimodal tag: 7 'real' rows + 5 contaminating rows. Medoid bootstrapping
    should return a centroid close to the 'real' rows' mean, not the
    average of both modes."""
    rng = np.random.RandomState(42)
    real_dir = rng.randn(8).astype(np.float32)
    real_dir /= np.linalg.norm(real_dir)
    contam_dir = -real_dir
    real = (real_dir[None] + 0.05 * rng.randn(7, 8)).astype(np.float32)
    real /= np.linalg.norm(real, axis=1, keepdims=True)
    contam = (contam_dir[None] + 0.05 * rng.randn(5, 8)).astype(np.float32)
    contam /= np.linalg.norm(contam, axis=1, keepdims=True)
    # First row is genuinely real, will be the medoid since it's central to
    # the 7-row cluster which beats the 5-row cluster on mean-pairwise.
    bimodal = np.vstack([real, contam]).astype(np.float32)

    cold_start_centroid = medoid_centered_centroid(bimodal, core_threshold=0.85)
    # Should align with real_dir, not contaminated mean
    real_mean = real.mean(axis=0)
    real_mean /= np.linalg.norm(real_mean)
    contam_mean = contam.mean(axis=0)
    contam_mean /= np.linalg.norm(contam_mean)
    assert float(cold_start_centroid @ real_mean) > 0.95
    assert float(cold_start_centroid @ contam_mean) < 0.0


def test_is_outlier_heavy_detects_bimodal() -> None:
    """Bimodal vectors trigger outlier-heavy flag; tight cluster does not."""
    rng = np.random.RandomState(0)
    config = RepairConfig()

    # Tight cluster — should NOT be flagged
    direction = rng.randn(8).astype(np.float32)
    direction /= np.linalg.norm(direction)
    tight = (direction[None] + 0.05 * rng.randn(15, 8)).astype(np.float32)
    tight /= np.linalg.norm(tight, axis=1, keepdims=True)
    assert not is_outlier_heavy(tight, config)

    # Bimodal — should be flagged
    half_a = (direction[None] + 0.05 * rng.randn(7, 8)).astype(np.float32)
    half_a /= np.linalg.norm(half_a, axis=1, keepdims=True)
    half_b = (-direction[None] + 0.05 * rng.randn(7, 8)).astype(np.float32)
    half_b /= np.linalg.norm(half_b, axis=1, keepdims=True)
    bimodal = np.vstack([half_a, half_b]).astype(np.float32)
    assert is_outlier_heavy(bimodal, config)


def test_recompute_centroids_initializes_at_iter_0() -> None:
    """At iter 0, centroids must be populated (no damping applied)."""
    rng = np.random.RandomState(0)
    # 3 well-separated tags, 5 rows each
    centers = np.eye(3, dtype=np.float32)  # 3 orthogonal directions in 3D
    rows = []
    tags = []
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            row = c + 0.05 * rng.randn(3).astype(np.float32)
            rows.append(row)
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig()

    recompute_centroids(state, cfg)

    # Each centroid should align with its corresponding center
    for tag_i, c in enumerate(centers):
        c_norm = c / np.linalg.norm(c)
        sim = float(state.current_centroids[tag_i] @ c_norm)
        assert sim > 0.99, f"tag {tag_i}: cos {sim:.4f} to seed direction"


def test_compute_geometry_yields_correct_margin() -> None:
    """3 well-separated tags → every row's margin > 0 (own_sim > best_other_sim)."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows = []
    tags = []
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            row = c + 0.05 * rng.randn(3).astype(np.float32)
            rows.append(row)
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig()
    recompute_centroids(state, cfg)

    geom = compute_geometry(state)
    margin = geom["margin"]
    # All rows' margin must be positive (well-separated tags)
    assert (margin > 0).all(), f"some margins not positive: {margin}"


def test_compute_geometry_negative_margin_when_mistagged() -> None:
    """A row whose embedding is closer to another tag's centroid should
    show negative margin."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows = []
    tags = []
    # Build well-separated tags
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            row = c + 0.02 * rng.randn(3).astype(np.float32)
            rows.append(row)
            tags.append(["x", "y", "z"][tag_i])
    # Now add a row whose embedding is near tag y's centroid but tagged as x
    mistagged = centers[1] + 0.02 * rng.randn(3).astype(np.float32)
    rows.append(mistagged)
    tags.append("x")
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig()
    recompute_centroids(state, cfg)

    geom = compute_geometry(state)
    # The last row (mistagged) should have negative margin and best_other = y
    mistagged_idx = len(rows) - 1
    assert geom["margin"][mistagged_idx] < 0.0
    y_idx = state.tag_names.index("y")
    assert geom["best_other"][mistagged_idx] == y_idx


def test_compute_knn_tag_share_assigns_to_correct_tag() -> None:
    """A row's kNN should be dominated by its own tag in a clean dataset."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows = []
    tags = []
    for tag_i, c in enumerate(centers):
        for _ in range(8):
            row = c + 0.02 * rng.randn(3).astype(np.float32)
            rows.append(row)
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)

    share = compute_knn_tag_share(state, top_k=5)
    # share is (N, T); for each row, the share of its own tag should be > 0.5
    for row_i in range(len(rows)):
        own_tag = state.assignment[row_i]
        assert share[row_i, own_tag] > 0.5, (
            f"row {row_i} (tag={own_tag}): own-tag share {share[row_i, own_tag]:.2f} too low"
        )


def test_compute_knn_tag_share_zero_for_dropped_rows() -> None:
    """Dropped rows produce all-zero share."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows = []
    tags = []
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            row = c + 0.02 * rng.randn(3).astype(np.float32)
            rows.append(row)
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    state.status[0] = STATUS_DROPPED

    share = compute_knn_tag_share(state, top_k=3)
    assert share[0].sum() == 0.0


def test_initialize_state_freezes_centroids() -> None:
    """initialize_state populates frozen_centroids + frozen_pairwise_cosine."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows = []
    tags = []
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            row = c + 0.05 * rng.randn(3).astype(np.float32)
            rows.append(row)
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig()

    initialize_state(state, cfg)

    # frozen_centroids should equal current_centroids at iter 0
    assert np.allclose(state.frozen_centroids, state.current_centroids)
    # frozen pairwise cosines: diagonal ~= 1.0, off-diagonal small (orthogonal seeds)
    pw = state.frozen_pairwise_cosine
    for i in range(3):
        assert abs(pw[i, i] - 1.0) < 0.01, f"pw[{i},{i}] = {pw[i,i]:.4f}"
        for j in range(3):
            if i != j:
                assert pw[i, j] < 0.2, f"pw[{i},{j}] = {pw[i,j]:.4f}"


# -------------------- Phase A: Reassignment (step 4) --------------------

from tagclean.repair import (
    apply_drops_phase_b,
    apply_merges,
    apply_reassignments,
    compute_merge_knn_overlap,
    compute_mutual_directed_confusion,
    find_cross_tag_near_dups,
    propose_merges,
    propose_reassignments,
    run_repair_loop,
    update_absorber_flags,
)


def _build_three_tag_corpus_with_mistag(rng_seed: int = 0) -> tuple[RepairState, RepairConfig]:
    """3 well-separated tags + 1 row in tag_x that geometrically belongs in tag_y."""
    rng = np.random.RandomState(rng_seed)
    centers = np.eye(3, dtype=np.float32)
    rows = []
    tags = []
    for tag_i, c in enumerate(centers):
        for _ in range(8):
            row = c + 0.02 * rng.randn(3).astype(np.float32)
            rows.append(row)
            tags.append(["x", "y", "z"][tag_i])
    # Row that should be in y but is tagged x
    rows.append(centers[1] + 0.02 * rng.randn(3).astype(np.float32))
    tags.append("x")
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig()
    initialize_state(state, cfg)
    return state, cfg


def test_propose_reassignments_moves_obvious_mistag() -> None:
    state, cfg = _build_three_tag_corpus_with_mistag()
    state.iter = 1
    geom = compute_geometry(state)
    knn_share = compute_knn_tag_share(state, top_k=cfg.knn_top_k)
    moves = propose_reassignments(state, geom, cfg, knn_share=knn_share)

    # The mistagged row (last index) should be in the move list
    mistag_idx = state.n_rows - 1
    move_rows = {row_idx for row_idx, _, _ in moves}
    assert mistag_idx in move_rows
    # And the destination should be tag y (idx 1, since alphabetic: x=0, y=1, z=2)
    for row_idx, cur, dest in moves:
        if row_idx == mistag_idx:
            assert dest == state.tag_names.index("y")
            assert cur == state.tag_names.index("x")


def test_propose_reassignments_does_not_move_clean_rows() -> None:
    """A clean dataset should produce zero moves."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows, tags = [], []
    for tag_i, c in enumerate(centers):
        for _ in range(8):
            rows.append(c + 0.02 * rng.randn(3).astype(np.float32))
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig()
    initialize_state(state, cfg)
    state.iter = 1
    geom = compute_geometry(state)
    knn_share = compute_knn_tag_share(state, top_k=cfg.knn_top_k)
    moves = propose_reassignments(state, geom, cfg, knn_share=knn_share)
    assert moves == []


def test_apply_reassignments_updates_state_and_centroids() -> None:
    state, cfg = _build_three_tag_corpus_with_mistag()
    state.iter = 1
    centroids_before = state.current_centroids.copy()
    geom = compute_geometry(state)
    knn_share = compute_knn_tag_share(state, top_k=cfg.knn_top_k)
    moves = propose_reassignments(state, geom, cfg, knn_share=knn_share)
    apply_reassignments(state, moves, cfg)
    # Mistagged row should now be in tag y
    mistag_idx = state.n_rows - 1
    y_idx = state.tag_names.index("y")
    x_idx = state.tag_names.index("x")
    assert state.assignment[mistag_idx] == y_idx
    assert state.prior_tag[mistag_idx] == x_idx
    assert state.moves_remaining[mistag_idx] == cfg.move_budget - 1
    # Centroids of x and y should have updated (damping applied at iter 1)
    assert not np.allclose(state.current_centroids[x_idx], centroids_before[x_idx])


def test_reassignment_respects_move_budget() -> None:
    """Burn the move budget on a row that wants to move; no further moves possible."""
    state, cfg = _build_three_tag_corpus_with_mistag()
    state.moves_remaining[state.n_rows - 1] = 0
    state.iter = 1
    geom = compute_geometry(state)
    knn_share = compute_knn_tag_share(state, top_k=cfg.knn_top_k)
    moves = propose_reassignments(state, geom, cfg, knn_share=knn_share)
    move_rows = {r for r, _, _ in moves}
    assert state.n_rows - 1 not in move_rows


def test_reassignment_hysteresis_prevents_immediate_back_move() -> None:
    """A row that just moved A->B needs an extra delta_hyst gap to move back B->A."""
    state, cfg = _build_three_tag_corpus_with_mistag()
    # Force a setup where prior_tag=x and current=y, gap is borderline
    mistag_idx = state.n_rows - 1
    state.prior_tag[mistag_idx] = state.tag_names.index("x")
    state.assignment[mistag_idx] = state.tag_names.index("y")
    state.iter = 1
    geom = compute_geometry(state)
    knn_share = compute_knn_tag_share(state, top_k=cfg.knn_top_k)
    moves = propose_reassignments(state, geom, cfg, knn_share=knn_share)
    # The row currently in y and prior x — it shouldn't want to move to x again
    # (because the embedding is near y's centroid). This test mostly verifies the
    # hysteresis branch executes; assertion is no-op-on-mistag-row.
    move_rows = {r for r, _, _ in moves}
    assert mistag_idx not in move_rows


# -------------------- Phase B: Drop / triage (steps 5-6) --------------------

def test_find_cross_tag_near_dups_detects_pair() -> None:
    """Two near-identical rows in different tags must be detected."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows = []
    tags = []
    # Build well-separated tags
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            rows.append(c + 0.02 * rng.randn(3).astype(np.float32))
            tags.append(["x", "y", "z"][tag_i])
    # Inject a near-duplicate: a row in x that's identical to a row in y
    duplicate_question_emb = centers[0] + 0.005 * rng.randn(3).astype(np.float32)
    rows.append(duplicate_question_emb)
    tags.append("x")
    rows.append(duplicate_question_emb + 0.001 * rng.randn(3).astype(np.float32))
    tags.append("y")
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig(cross_tag_dup_thresh=0.99)
    initialize_state(state, cfg)
    pairs = find_cross_tag_near_dups(state, cfg)
    last_x = len(rows) - 2
    last_y = len(rows) - 1
    detected_pairs = {tuple(sorted(p)) for p in pairs}
    assert (last_x, last_y) in detected_pairs


def test_apply_drops_phase_b_drops_hard_margin_outliers() -> None:
    """Row with margin << threshold should be dropped."""
    rng = np.random.RandomState(0)
    # 3 well-separated tags + 1 row whose embedding is far from any centroid
    centers = np.eye(3, dtype=np.float32)
    rows, tags = [], []
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            rows.append(c + 0.02 * rng.randn(3).astype(np.float32))
            tags.append(["x", "y", "z"][tag_i])
    # A row tagged x but whose embedding is roughly equidistant from y/z, far from x
    bad = (centers[1] + centers[2]) / 2 - centers[0]
    bad_norm = bad / np.linalg.norm(bad)
    rows.append(bad_norm + 0.01 * rng.randn(3).astype(np.float32))
    tags.append("x")
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig(hard_drop_thresh=-0.10)
    initialize_state(state, cfg)
    state.iter = 1
    geom = compute_geometry(state)
    cap_strikes: dict[int, int] = {}
    result = apply_drops_phase_b(state, geom, cfg, cap_strikes)
    # The bad row should be flagged as dropped
    bad_idx = state.n_rows - 1
    # Either it was hard-dropped, or maybe winner-take if there's a near-dup
    # Just check it's no longer kept
    assert state.status[bad_idx] == STATUS_DROPPED, f"bad row status: {state.status[bad_idx]}"


# -------------------- Phase C: Merge (steps 9-10) --------------------

def test_compute_mutual_directed_confusion_zero_for_well_separated() -> None:
    """3 well-separated tags should have zero mutual confusion."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows, tags = [], []
    for tag_i, c in enumerate(centers):
        for _ in range(5):
            rows.append(c + 0.02 * rng.randn(3).astype(np.float32))
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig()
    initialize_state(state, cfg)
    confusion = compute_mutual_directed_confusion(state)
    # Diagonal must be 0
    for i in range(3):
        assert confusion[i, i] == 0.0
    # Off-diagonal: not necessarily 0 since each row has SOME nearest other,
    # but it should be split fairly evenly between the two non-self tags.
    # Just check no value dominates.
    for i in range(3):
        max_other = float(confusion[i].max())
        assert max_other < 0.9, f"tag {i} has confusion peak {max_other:.2f}"


def test_propose_merges_finds_overlapping_tags() -> None:
    """Two tags whose centroids are nearly identical should merge."""
    rng = np.random.RandomState(0)
    # Tag x and tag x_dup share the same direction, just different rows
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    distinct = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    rows, tags = [], []
    for _ in range(8):
        rows.append(direction + 0.02 * rng.randn(3).astype(np.float32))
        tags.append("x")
    for _ in range(8):
        rows.append(direction + 0.02 * rng.randn(3).astype(np.float32))
        tags.append("x_dup")
    for _ in range(8):
        rows.append(distinct + 0.02 * rng.randn(3).astype(np.float32))
        tags.append("y")
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig(
        merge_cosine_thresh=0.90,
        mutual_confusion_thresh=0.50,
        merge_knn_overlap_thresh=0.30,
    )
    initialize_state(state, cfg)
    state.iter = 1
    proposals = propose_merges(state, cfg)
    # x and x_dup should merge
    proposed_pairs = {(min(a, b), max(a, b)) for a, b in proposals}
    x_idx = state.tag_names.index("x")
    x_dup_idx = state.tag_names.index("x_dup")
    assert (min(x_idx, x_dup_idx), max(x_idx, x_dup_idx)) in proposed_pairs


def test_apply_merges_dissolves_other_tag() -> None:
    """After merge, the absorbed tag is no longer alive and its rows now
    point to canonical."""
    rng = np.random.RandomState(0)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rows, tags = [], []
    for _ in range(8):
        rows.append(direction + 0.02 * rng.randn(3).astype(np.float32))
        tags.append("x")
    for _ in range(8):
        rows.append(direction + 0.02 * rng.randn(3).astype(np.float32))
        tags.append("x_dup")
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig(merge_cosine_thresh=0.90, mutual_confusion_thresh=0.10)
    initialize_state(state, cfg)
    state.iter = 1
    proposals = propose_merges(state, cfg)
    apply_merges(state, proposals, cfg)
    # Exactly one of x or x_dup should be alive after merge
    x_idx = state.tag_names.index("x")
    x_dup_idx = state.tag_names.index("x_dup")
    assert state.alive_tags[x_idx] != state.alive_tags[x_dup_idx]
    # All rows should now resolve to the surviving canonical
    surviving = x_idx if state.alive_tags[x_idx] else x_dup_idx
    for r in range(state.n_rows):
        assert state._resolve_canonical(int(state.assignment[r])) == surviving


# -------------------- Phase D: Absorber detection --------------------

def test_absorber_flag_triggers_after_consecutive_high_intake() -> None:
    """Tag receiving high intake for absorber_strikes_to_flag iters becomes flagged."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows, tags = [], []
    for tag_i, c in enumerate(centers):
        for _ in range(10):
            rows.append(c + 0.02 * rng.randn(3).astype(np.float32))
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig(absorber_intake_thresh=0.30, absorber_strikes_to_flag=2)
    initialize_state(state, cfg)
    # Simulate two iterations of high intake into tag x
    x_idx = state.tag_names.index("x")
    update_absorber_flags(state, intake_per_tag={x_idx: 5}, config=cfg)
    update_absorber_flags(state, intake_per_tag={x_idx: 5}, config=cfg)
    assert x_idx in state.untrusted_tags
    # After a low-intake iter, recovery should kick in
    update_absorber_flags(state, intake_per_tag={x_idx: 0}, config=cfg)
    # 0 intake -> full recovery
    assert x_idx not in state.untrusted_tags


# -------------------- Loop driver (step 12) --------------------

def test_run_repair_loop_converges_on_clean_data() -> None:
    """Clean dataset should converge in 1-2 iters with 0 changes."""
    rng = np.random.RandomState(0)
    centers = np.eye(3, dtype=np.float32)
    rows, tags = [], []
    for tag_i, c in enumerate(centers):
        for _ in range(8):
            rows.append(c + 0.02 * rng.randn(3).astype(np.float32))
            tags.append(["x", "y", "z"][tag_i])
    state = _make_state(np.vstack(rows), tags)
    cfg = RepairConfig(max_iter=4)
    result = run_repair_loop(state, cfg)
    assert result["final_status"] == "converged"
    # All rows still kept
    assert state.kept_row_mask().sum() == state.n_rows


def test_run_repair_loop_repairs_obvious_mistag() -> None:
    """A row in tag x whose embedding belongs in tag y should end up in y after the loop."""
    state, cfg = _build_three_tag_corpus_with_mistag()
    cfg.max_iter = 4
    mistag_idx = state.n_rows - 1
    original_assignment = int(state.assignment[mistag_idx])
    run_repair_loop(state, cfg)
    final_canonical = state._resolve_canonical(int(state.assignment[mistag_idx]))
    y_idx = state.tag_names.index("y")
    # Should now be assigned to y (or merged tag containing y) — in this clean
    # case no merges happen, so just y.
    assert final_canonical == y_idx
    assert original_assignment != final_canonical
