# Repair Loop — System Design

This document explains *what* the geometric repair loop is and *how* it makes decisions. For *why* the design constraints exist (pure geometry, no LLM, no answer/name heuristics), see the README. For tuning knobs, see [`repair_thresholds.md`](repair_thresholds.md).

## Goal

Take a tagged FAQ corpus and produce a cleaned subset where, when each row is used as a query against the rest of the corpus (leave-one-out), the **top-1 nearest E5 neighbor lives in the same tag**. That's the property production retrieval needs.

Anything the loop drops, reassigns, or merges is a decision driven by frozen E5 embedding geometry — no LLM, no `tag_answer.json`, no tag-name heuristics.

## High-level pipeline

```
                    ┌────────────────────────────────────┐
                    │  data/question_tag.csv             │
                    │  78,990 rows × 1,394 tags (raw)    │
                    └──────────────────┬─────────────────┘
                                       │
                                       ▼
                    ┌────────────────────────────────────┐
                    │  Stage 0: intake                   │
                    │  • normalize (Bengali Unicode)      │
                    │  • dedup within tag                 │
                    │  • cross-tag duplicate log          │
                    └──────────────────┬─────────────────┘
                                       │
                                       ▼
                    ┌────────────────────────────────────┐
                    │  Stage 1: E5 embed                 │
                    │  • multilingual-e5-large-instruct, │
                    │    1024d, L2-normalized            │
                    │  • FAISS IndexFlatIP built         │
                    └──────────────────┬─────────────────┘
                                       │
                                       ▼
                    ┌────────────────────────────────────┐
                    │       REPAIR LOOP                  │
                    │       (Phases A → C → B → D)       │
                    └──────────────────┬─────────────────┘
                                       │
                                       ▼
                    ┌────────────────────────────────────┐
                    │  xray_cleanup.csv                  │
                    │  37,355 rows × 1,334 tags          │
                    │  pct_neg_margin = 0.026%           │
                    └────────────────────────────────────┘
```

## Loop overview

```
                       ITER 0 (cold start, runs once)
                       ──────────────────────────────
                ┌─────────────────────────────────────┐
                │  Detect outlier-heavy tags          │
                │  (e5_diversity > 0.4)               │
                │      ↓                              │
                │  Bootstrap centroid:                │
                │   • medoid + core for outlier-heavy │
                │   • trimmed mean for the rest       │
                │      ↓                              │
                │  FREEZE iter-0 centroids +          │
                │  pairwise cosines (used as          │
                │  hysteresis predicate later)        │
                └────────────────┬────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────┐
        │  ITER k = 1..max_iter (default 8)          │
        │                                             │
        │  ┌──────────────────────────────────────┐   │
        │  │  PHASE A — REASSIGN                  │   │
        │  │  Move rows that geometrically belong │   │
        │  │  elsewhere. Gates: margin +          │   │
        │  │  delta_move + kNN-majority +         │   │
        │  │  hysteresis + budget.                │   │
        │  └──────────────┬───────────────────────┘   │
        │                 │                            │
        │                 ▼                            │
        │  ┌──────────────────────────────────────┐   │
        │  │  PHASE C — MERGE  (only iter ≤ 2)    │   │
        │  │  Collapse tag pairs that are E5-     │   │
        │  │  indistinguishable AND mutually      │   │
        │  │  confused. Iter-0 frozen cosine      │   │
        │  │  hysteresis predicate.               │   │
        │  │  Note: C runs BEFORE B so cross-tag  │   │
        │  │  pairs whose tags merge disappear.   │   │
        │  └──────────────┬───────────────────────┘   │
        │                 │                            │
        │                 ▼                            │
        │  ┌──────────────────────────────────────┐   │
        │  │  PHASE B — TRIAGE / DROP             │   │
        │  │  Cross-tag dup pairs: winner-take    │   │
        │  │   or drop-both.                      │   │
        │  │  Hard-margin: drop rows below        │   │
        │  │   hard_drop_thresh.                  │   │
        │  │  Per-tag drop cap; abort after N     │   │
        │  │   consecutive cap-strikes.           │   │
        │  └──────────────┬───────────────────────┘   │
        │                 │                            │
        │                 ▼                            │
        │  ┌──────────────────────────────────────┐   │
        │  │  PHASE D — ABSORBER + CONVERGE       │   │
        │  │  Flag tags with >30% intake/iter for │   │
        │  │   2+ iters → freeze their centroid.  │   │
        │  │  Check: (changes/N) < 0.001 for 2    │   │
        │  │   consec iters → CONVERGED.          │   │
        │  │  Check: pct_neg_margin rising for 2  │   │
        │  │   consec iters → ABORT_OSCILLATION.  │   │
        │  └──────────────┬───────────────────────┘   │
        │                 │                            │
        │  ┌──────────────┴───────────────────┐        │
        │  │ continue OR exit (converged /   │        │
        │  │ abort_drop_cap / max_iter)      │        │
        │  └──────────────┬───────────────────┘        │
        └─────────────────┼────────────────────────────┘
                          │
                          ▼
                ┌─────────────────────────────────────┐
                │  POST-LOOP                          │
                │  • Dissolve tags with < 3 rows      │
                │  • Emit final_assignment.parquet,   │
                │    iter_metrics.jsonl, repair_      │
                │    report.json, xray_cleanup.csv    │
                └─────────────────────────────────────┘
```

## Per-row reassignment decision (Phase A gates)

```
   Input: row r in tag A, with embedding e(r)
        │
        ▼
   ┌──────────────────────────────────┐
   │  Compute geometry:               │
   │   own_sim    = cos(e(r), C[A])   │
   │   best_other = argmax cos(e(r),  │
   │                C[t]) for t ≠ A   │
   │   margin     = own_sim −         │
   │                best_other_sim    │
   └─────────────────┬────────────────┘
                     │
                     ▼
   ┌──────────────────────────────────┐
   │  margin < reassign_thresh        │   NO
   │  (-0.03)?                        │ ──────► STAY
   └─────────────────┬────────────────┘
                     │ YES
                     ▼
   ┌──────────────────────────────────┐
   │  best_other_sim − own_sim        │   NO
   │  > delta_move (0.05)?            │ ──────► STAY
   │  (+ delta_hyst if best_other     │
   │   == prior_tag)                  │
   └─────────────────┬────────────────┘
                     │ YES
                     ▼
   ┌──────────────────────────────────┐
   │  moves_remaining[r] > 0?         │   NO
   │                                  │ ──────► FROZEN (drop candidate)
   └─────────────────┬────────────────┘
                     │ YES
                     ▼
   ┌──────────────────────────────────┐
   │  best_other ∉ untrusted_tags?    │   NO
   │  (absorber-flag protection)      │ ──────► STAY
   └─────────────────┬────────────────┘
                     │ YES
                     ▼
   ┌──────────────────────────────────┐
   │  Compute kNN tag-share for r     │
   │  (FAISS top-10, lazy)            │
   │  knn_share[best_other] >         │   NO
   │  knn_share[current]?             │ ──────► STAY (centroid agrees but
   │                                  │         neighborhood disagrees)
   └─────────────────┬────────────────┘
                     │ YES
                     ▼
   ┌──────────────────────────────────┐
   │  Per-tag intake cap check        │   NO
   │  (≤ 10% of best_other size)?     │ ──────► DEFER (try next iter)
   └─────────────────┬────────────────┘
                     │ YES (rank by gap, top-N apply)
                     ▼
                  REASSIGN
        A(r) := best_other,
        prior_tag[r] := A,
        moves_remaining[r] −= 1

## Inputs / outputs

```
INPUT
  question_tag.csv  →  Stage 0 (normalize, dedup)
                    →  Stage 1 (E5 embeddings, FAISS index built)
                    →  Repair loop (this document)

OUTPUT
  runs/<run_id>/repair/
    xray_cleanup.csv        ← cleaned (question, tag), the deliverable
    final_assignment.parquet ← per-row trail
    iter_metrics.jsonl       ← per-iter scalars (drops, merges, margin dist)
    iter_NN_ops.jsonl        ← per-op log
    repair_report.json       ← summary
```

The repair loop assumes the corpus has already been embedded (Stage 0/1 cached). Re-runs in the same `run_id` reuse those embeddings.

## State

The loop carries a single `RepairState` (`src/tagclean/repair_state.py`):

| Field | Shape | Meaning |
|---|---|---|
| `embeddings` | (N, 1024) float32 | L2-normalized E5 vectors |
| `assignment` | (N,) int32 | Current tag index per row (mutates) |
| `original_assignment` | (N,) int32 | Frozen at iter 0; for diff |
| `current_centroids` | (T, 1024) | Recomputed per iter with damping |
| `frozen_centroids` | (T, 1024) | Iter-0 snapshot for hysteresis + absorber rescue |
| `frozen_pairwise_cosine` | (T, T) | Iter-0 tag-pair cosines for merge hysteresis |
| `alive_tags` | (T,) bool | False once dissolved or merged into another |
| `canonical_of` | (T,) int32 | If merged, points to the surviving canonical's index |
| `moves_remaining` | (N,) int8 | Per-row reassignment budget (default 3) |
| `prior_tag` | (N,) int32 | Tag this row most recently came FROM (-1 if never moved) |
| `status` | (N,) object | `kept` / `dropped` / `reassigned` / `merged_into` |
| `untrusted_tags` | set[int] | Outlier-heavy at iter 0 OR flagged absorber later |
| `absorber_history` | dict | Per-tag intake fraction history for absorber detection |

## The four phases

The loop is `INIT → for k in 1..max_iter: A → C → B → D`. Phase order matters — Phase C (merges) runs **before** Phase B (drops) so cross-tag near-duplicate pairs whose tags merge disappear before triage sees them.

### INIT — iter 0 (cold start)

- Load embeddings + initial assignment from the input.
- Detect outlier-heavy tags. A tag is outlier-heavy iff:
  - `e5_diversity = 1 - mean(cos(row, centroid))` > **0.40**, OR
  - the trimmed-mean centroid disagrees with the medoid by > **0.15** cosine.
- For outlier-heavy tags: bootstrap centroid from medoid + core (rows with cos > 0.85 to medoid). Otherwise: trimmed mean (drop top/bottom 5% by sim-to-rough-centroid, then average).
- Snapshot `frozen_centroids` and `frozen_pairwise_cosine` from this state — never modified again.
- Mark outlier-heavy tags as untrusted (excluded as reassignment targets in iter 1).

### Phase A — REASSIGN (every iter)

For each kept row r, compute:
- `own_sim[r]` = cos(emb(r), centroid[A(r)])
- `best_other[r]` = argmax cos(emb(r), centroid[t]) for t ≠ A(r) and t alive
- `margin[r]` = `own_sim[r] - best_other_sim[r]`

Two-stage filtering:

**Cheap gates** (vectorized over all rows):
- `kept` and `assignment` is alive
- `margin < reassign_thresh` (default -0.03)
- `gap = best_other_sim - own_sim > delta_move` (default 0.05)
- If `best_other == prior_tag`: gap must exceed `delta_move + delta_hyst` (extra 0.02) — hysteresis
- `moves_remaining > 0`
- `best_other ∉ untrusted_tags`

**Expensive gate** (lazy, only on cheap-pass candidates):
- For each candidate row, FAISS top-K (default 10) nearest neighbors. Tag the share each tag gets in those neighbors.
- Require: `kNN_share[best_other] > kNN_share[current]` — the row's local neighborhood agrees with the centroid signal.

Apply the moves, decrement `moves_remaining`, set `prior_tag = current`. Recompute centroids of affected tags only, with annealed damping:

| Iter | Old-centroid weight |
|---|---:|
| 1-2 | 0.50 |
| 3-4 | 0.30 |
| 5-7 | 0.15 |
| 8 | 0.00 |

Per-tag intake cap: at most `intake_cap_per_tag * tag_size` reassignments INTO any tag per iter (default 10%). Strongest-margin candidates win when over the cap.

### Phase C — MERGE (only iter ≤ 2)

For each pair of alive tags (a, b), merge if **all four gates** pass:

1. `cos(centroid[a], centroid[b]) > merge_cosine_thresh` (default 0.92)
2. `frozen_pairwise_cosine[a, b] > merge_cosine_thresh` — **hysteresis predicate**: pair must have been close at iter 0, not just become close after reassignments dragged centroids together
3. **Mutual directed confusion** > `mutual_confusion_thresh` (default 0.50) in BOTH directions:
   - fraction of A's rows whose nearest non-A centroid is B
   - fraction of B's rows whose nearest non-B centroid is A
4. **kNN overlap** > `merge_knn_overlap_thresh` (default 0.50): in a sampled mixed neighborhood of A's and B's rows, fraction of cross-tag neighbors

Pick the canonical via geometric composite score (NO names):
```
score(t) = row_count(t)
          + 100 * mean_cos_to_medoid_in_top_quartile(t)    # neighborhood support
          - 50 * (1 - cos(medoid, centroid))               # outlier penalty
```
Higher score wins. The "loser" tag's rows reassign to the winner; loser is marked `alive_tags[loser] = False`; `canonical_of[loser] = winner`.

Merges only fire in iter 1-2 to prevent cascade drift (AB merging with C because AB's centroid ≠ A's). After iter 2, no new merges.

### Phase B — TRIAGE (every iter)

Find cross-tag near-duplicate pairs via FAISS `range_search` at threshold 0.985 (default `cross_tag_dup_thresh`). For each pair (r_a in tag_A, r_b in tag_B), apply pure-geometry triage:

- **Case 1 — winner-take**: if `|own_sim(r_a) - own_sim(r_b)| > 0.05`, drop the loser, keep the winner.
- **Case 2 — drop both** (last resort): no clear winner — drop both rows. Phase C ran first, so any pair whose tags would merge has already been resolved by merging; remaining pairs reaching this step are genuinely undecidable.

Plus a separate hard-margin filter:
- Any kept row with `margin < hard_drop_thresh` (default -0.10) is dropped.

Per-tag drop cap: at most `drop_cap_per_tag * tag_size` drops per iter (default 20%). Excess deferred to next iter; consecutive cap hits accumulate `cap_strikes` per tag. After `drop_cap_strikes` consecutive cap hits (default 3), the loop **aborts** with `loop_status = abort_drop_cap`.

### Phase D — ABSORBER DETECTION + CONVERGENCE

A tag receiving > `absorber_intake_thresh` (default 0.30) of its size as reassignments-IN per iter, for `absorber_strikes_to_flag` (default 2) consecutive iters, is flagged "untrusted absorber":
- Its centroid is replaced with the iter-0 frozen version.
- It's excluded as a reassignment target until intake drops below ⅓ of the threshold.

Convergence:
- If `(reassigned + dropped + merged) / N < convergence_change_frac` (default 0.001) for `convergence_required_consec` (default 2) consecutive iters → `loop_status = converged`.
- If `pct_neg_margin` (fraction of rows with negative margin) **rises** for 2 consecutive iters → `loop_status = abort_oscillation` (a signal the gates aren't producing monotone improvement).
- If `iter == max_iter` (default 8) without converging → `loop_status = max_iter`.

After the loop, tags below `min_tag_rows` (default 3) are dissolved — their rows get `status = dropped`.

## Why this order: A → C → B → D

Originally I had A → B → C. That broke. Cross-tag near-duplicate pairs whose tags should merge would land in Phase B's "Case 1 merge_defer" branch, which deferred them to Phase C — but Phase C's merge gates are stricter than the simple centroid-cosine check the deferral used, so most deferrals never merged and never got dropped. Same pairs surfaced every iter and the loop converged on stuck-deferred state with no actual cleaning.

Moving Phase C **before** Phase B means: by the time Phase B sees cross-tag dup pairs, any pair whose tags were going to merge has already merged. The remaining pairs are decisively winner-take or drop-both — no deferral, no stuck state.

## Why iter-0 frozen centroids matter

Two places use the iter-0 snapshot:

1. **Merge hysteresis** (Phase C gate 2): a pair (a, b) can only merge if their cosine was already above threshold *at iter 0*. Without this, reassignments could drag two unrelated tags' centroids together, then Phase C would merge them — a cascade artifact, not real evidence.

2. **Absorber rescue** (Phase D): when a tag is flagged as a runaway absorber (sucking in too many reassignments per iter), its centroid is reset to the iter-0 version to break the feedback loop. Without rescue, a noisy broad tag could attract every uncertain row and dissolve all its competitors.

## Why annealed damping

Iteration k blends new centroid with previous: `new = (1-α) * fresh + α * previous`, where α is read from a schedule (0.5 → 0.5 → 0.3 → 0.3 → 0.15 → 0.15 → 0 → 0).

Heavier damping early prevents centroid swing when many reassignments happen at once. Lighter damping later lets the loop settle.

## Why pure geometry (and what it costs)

Every gate above uses only:
- E5 cosines between rows
- E5 cosines between row and tag centroid
- Row's top-K neighbors by E5 cosine
- Tag-pair centroid cosine and mutual neighbor confusion

No tag names. No `tag_answer.json`. No LLM.

Cost of that constraint: the loop **cannot prevent merging two tags whose canonical answers materially differ** when E5 says they're indistinguishable. If `password_reset` and `password_change` happen to be E5-close (they share heavy boilerplate), the loop will collapse them — even though their canonical answers differ. This is a deliberate trade-off; see README.

## Wall-time + memory

On the Bengali NID corpus (78,990 rows × 1,394 tags, 1024d embeddings):

| Phase | Time per iter | Memory peak |
|---|---|---|
| Geometry recompute | ~2s | 440 MB scores matrix |
| Phase A (lazy kNN) | ~0.5s | minimal — only candidate rows |
| Phase C (merge) | ~3s | ~10 MB pairwise + kNN samples |
| Phase B (cross-tag dup detect via FAISS range_search) | ~5 min | ~640 MB FAISS index |
| Phase D | <0.1s | trivial |
| **Total per iter** | ~5 min | ~1.5 GB peak working set |

Resource caps locked in `repair.py`: 4 OMP/MKL/OpenBLAS/FAISS threads, peak working set well under 2 GB.

The 5-minute cross-tag dup detection dominates. It's an exhaustive O(N²) range search (87k pairs above threshold on this corpus). Acceptable at 79k scale; would need approximate methods (HNSW) at 500k+.

## Convergence in practice

On the Bengali NID corpus at default thresholds:
- V1 (raw → first repair pass): 8,545 dropped, 50 merges, aborts at iter 3 on drop-cap (some tag couldn't lose 20% per iter for 3 consecutive iters)
- V2 (repair on V1): 273 dropped, 58 merges, converges in 3 iters
- V3 (repair on V2): 0 dropped, 8 merges, converges in 2 iters

`pct_neg_margin` floor: ~16%. To push lower, loosen `hard_drop_thresh` toward 0 or `merge_cosine_thresh` toward 0.88. See [`repair_thresholds.md`](repair_thresholds.md) for trade-offs.

## What v1 deliberately doesn't do

- **Tag splitting** (one tag → two when bimodal). Without this, the loop cannot drive `pct_neg_margin` to 0% in pathological cases (a tag whose rows form two well-separated sub-clusters in E5 space). Splitting requires either k=2 GMM + automated naming or human-in-the-loop. Defer.
- **New tag creation for orphan rows.** A row that drops out of A and finds no good fit anywhere might genuinely belong in tag X-that-doesn't-exist. v1 just drops it.
- **Streaming / incremental.** Full-corpus batch only. Adding new rows means re-running.
- **Cross-language threshold tables.** Defaults are calibrated for Bengali NID + E5-multilingual-large-instruct. Other languages may need recalibration.

## File map

| File | What |
|---|---|
| `src/tagclean/repair.py` | Loop driver, all four phases, geometry primitives, CLI handler |
| `src/tagclean/repair_state.py` | `RepairState` dataclass — pure data |
| `src/tagclean/cleaner.py` | Stage 0/1 (the embedding pass), `STAGES` registry that wires `repair` into argparse |
| `tests/test_repair.py` | 29 tests on synthetic + corpus-shape fixtures |
| `configs/bn_full.yaml` | Bengali-tuned config |
| `docs/repair_thresholds.md` | Every knob with worked examples |
