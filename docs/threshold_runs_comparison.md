# Three Threshold Runs, Three Outcomes

A juxtaposition of the three concrete repair runs on the Bengali NID corpus. Same input (78,990 rows × 1,394 tags), different hyperparameter choices, very different cleaned outputs. The middle run was a misfire — kept here as a documented failure mode.

## Side-by-side configuration (the levers I changed)

| Hyperparameter | Run A: Conservative chain | Run B: Aggressive overshoot | Run C: Aggressive tuned (shipped default) |
|---|---:|---:|---:|
| `hard_drop_thresh` | **-0.10** | +0.007 | **+0.004** |
| `cross_tag_dup_thresh` | 0.985 | 0.95 | **0.97** |
| `merge_cosine_thresh` | 0.92 | 0.85 | **0.90** |
| `mutual_confusion_thresh` | 0.50 | 0.40 | 0.50 |
| `merge_knn_overlap_thresh` | 0.50 | 0.40 | 0.50 |
| `merges_allowed_through_iter` | 2 | 4 | 2 |
| `drop_cap_per_tag` | 0.20 | 0.50 | **0.30** |
| `drop_cap_strikes` | 3 | 6 | 4 |

## Complete hyperparameter inventory across all three runs

Everything in `RepairConfig`, including the knobs that were held constant. Bold values were tuned across runs.

### Phase A — Reassign

| Knob | Run A | Run B | Run C |
|---|---:|---:|---:|
| `reassign_thresh` | -0.03 | -0.03 | -0.03 |
| `delta_move` | 0.05 | 0.05 | 0.05 |
| `delta_hyst` | 0.02 | 0.02 | 0.02 |
| `knn_top_k` | 10 | 10 | 10 |
| `intake_cap_per_tag` | 0.10 | 0.10 | 0.10 |
| `move_budget` | 3 | 3 | 3 |

### Phase B — Drop / triage

| Knob | Run A | Run B | Run C |
|---|---:|---:|---:|
| **`cross_tag_dup_thresh`** | 0.985 | 0.95 | **0.97** |
| **`hard_drop_thresh`** | -0.10 | +0.007 | **+0.004** |
| **`drop_cap_per_tag`** | 0.20 | 0.50 | **0.30** |
| **`drop_cap_strikes`** | 3 | 6 | 4 |

### Phase C — Merge

| Knob | Run A | Run B | Run C |
|---|---:|---:|---:|
| **`merge_cosine_thresh`** | 0.92 | 0.85 | **0.90** |
| `merge_knn_overlap_thresh` | 0.50 | 0.40 | 0.50 |
| `mutual_confusion_thresh` | 0.50 | 0.40 | 0.50 |
| **`merges_allowed_through_iter`** | 2 | 4 | 2 |

### Phase D — Absorber + convergence

| Knob | Run A | Run B | Run C |
|---|---:|---:|---:|
| `absorber_intake_thresh` | 0.30 | 0.30 | 0.30 |
| `absorber_strikes_to_flag` | 2 | 2 | 2 |
| `convergence_change_frac` | 0.001 | 0.001 | 0.001 |
| `convergence_required_consec` | 2 | 2 | 2 |

### Cold start (iter 0)

| Knob | Run A | Run B | Run C |
|---|---:|---:|---:|
| `medoid_diversity_trigger` | 0.40 | 0.40 | 0.40 |
| `medoid_centroid_disagreement_trigger` | 0.15 | 0.15 | 0.15 |

### Loop control

| Knob | Run A | Run B | Run C |
|---|---:|---:|---:|
| `max_iter` | 8 | 8 | 8 |
| `min_tag_rows` | 3 | 3 | 3 |
| `damping_schedule` | (0.5, 0.5, 0.3, 0.3, 0.15, 0.15, 0.0, 0.0) | same | same |

## Hidden magic numbers (not in `RepairConfig` — hardcoded in functions)

These exist as default arguments or literal constants and are **identical across all three runs** (none of my tuning touched them). Listed here so you know they're tunable if you edit `repair.py`:

| Where | Value | What it controls |
|---|---:|---|
| `compute_trimmed_centroid(... trim_fraction=0.05)` (line 99) | **0.05** | Drop top/bottom 5% of rows by sim-to-rough-centroid before computing the trimmed mean. Smaller value = less outlier-trimming. |
| `medoid_centered_centroid(... core_threshold=0.85)` (line 138) | **0.85** | Cold-start core inclusion threshold. After bootstrapping from the medoid, only rows with `cos(row, medoid) > 0.85` count as the core. Lower → more rows included → centroid drifts; higher → tiny core, may fall back to trimmed mean. |
| `compute_merge_knn_overlap(... sample_cap=50, k=10)` (line 801) | **50, 10** | When checking kNN overlap for a tag-pair merge gate, sample up to 50 rows from each tag and count cross-tag presence in their top-10 neighbors. Larger sample → more accurate but slower. |
| `triage_cross_tag_pair`, line 652 | **0.05** | Winner-take own-sim diff threshold. Two cross-tag near-dup rows with `|own_sim_a − own_sim_b| > 0.05` → keep the higher, drop the lower. Tighter means more "drop both" outcomes. |
| `_pick_canonical` score formula (line ~870) | **100, 50** | Composite tiebreak when two tags merge: `score = row_count + 100*neighborhood_support − 50*medoid_centroid_disagreement`. The `100` weight makes neighborhood density much more decisive than tag size differences; `50` penalizes outlier-heavy tags from winning canonicalization. |

Of these, `core_threshold = 0.85` is the one most likely to matter for a different corpus — Bengali NID's tag clusters are tight enough that 0.85 captures real cores. A more diffuse corpus might need 0.75 or even 0.70.

## Outcomes

| Metric | Run A | Run B | Run C |
|---|---:|---:|---:|
| **Strategy** | 3 chained passes (V1→V2→V3) | 1 pass | 1 pass |
| Rows in | 78,990 | 78,990 | 78,990 |
| Rows out | **70,172** | **7,137** | **37,355** |
| Drop rate | 11.2% | 91.0% | 52.7% |
| Tags in | 1,394 | 1,394 | 1,394 |
| Tags out | 1,278 | 1,291 | 1,334 |
| Tag merges total | 116 | 96 | 60 |
| Loop status | converged (V3) | converged | abort_drop_cap |
| Iters total | 8 (across V1+V2+V3) | 3 | 4 |
| Wall-time total | ~34 min | 6.3 min | 10.6 min |
| `pct_neg_margin` final | 16.1% | 0.0% | 0.026% |

## Per-iteration trajectories

### Run A — conservative chain

V1 (78,990 input):
| iter | reassigned | dropped | merged | rows alive | pct_neg_margin |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 6,437 | 40 | 72,553 | 17.7% |
| 2 | 1 | 1,519 | 9 | 71,034 | 16.7% |
| 3 | 0 | 587 | 0 | 70,447 | 16.7% |
| **end (V1)** | **abort_drop_cap, 70,445 final** |

V2 (70,445 input):
| iter | reassigned | dropped | merged | rows alive | pct_neg_margin |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 174 | 14 | 70,271 | 16.1% |
| 2 | 0 | 32 | 8 | 70,239 | 16.1% |
| 3 | 0 | 8 | 0 | 70,231 | 16.1% |
| **end (V2)** | **converged, 70,172 final** |

V3 (70,172 input):
| iter | reassigned | dropped | merged | rows alive | pct_neg_margin |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 3 | 70,172 | 16.1% |
| 2 | 0 | 0 | 1 | 70,172 | 16.1% |
| **end (V3)** | **converged, 70,172 final** |

### Run B — aggressive overshoot

| iter | reassigned | dropped | merged | rows alive | pct_neg_margin |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | **71,650** | 96 | 7,340 | 0.0% |
| 2 | 0 | 7 | 7 | 7,333 | 0.0% |
| 3 | 0 | 0 | 0 | 7,333 | 0.0% |
| **end** | **converged, 7,137 final (post-tag-dissolve)** |

### Run C — aggressive tuned (current default)

| iter | reassigned | dropped | merged | rows alive | pct_neg_margin |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 32,614 | 40 | 46,376 | 3.1% |
| 2 | 0 | 7,064 | 20 | 39,312 | 0.6% |
| 3 | 0 | 1,452 | 0 | 37,860 | 0.026% |
| 4 | 0 | 345 | 0 | 37,515 | 0.026% |
| **end** | **abort_drop_cap, 37,355 final (post-tag-dissolve)** |

## What each run teaches

### Run A — "iteration alone, no threshold change, hits a floor"

Default conservative thresholds, three chained passes. By V3 the loop drops zero rows — the gates simply don't fire on the remaining 70k. `pct_neg_margin` plateaus at 16.1% because there are 11k+ rows whose own-tag is barely losing to a competing tag (margin between -0.03 and 0), but the `hard_drop_thresh = -0.10` filter doesn't see them. Iteration alone cannot push past this floor; the gates have to change.

### Run B — "all dials cranked, cascade collapse"

`drop_cap_per_tag` raised from 0.20 to **0.50** combined with `cross_tag_dup_thresh` lowered from 0.985 to **0.95** lets iter 1 drop **71,650 rows in one shot**. The 0.50 cap multiplied the per-tag damage; the 0.95 cross-tag threshold turned every Bengali paraphrase pair into a triage candidate. The loop then converges quickly because almost nothing's left.

Lesson: thresholds aren't independent. Loosening cross-tag dup detection AND raising drop cap simultaneously creates an avalanche.

### Run C — "single dominant lever + safety rails"

Targeted at the floor that Run A revealed: raise `hard_drop_thresh` from -0.10 to **+0.004** so the rows in the negative-to-near-zero margin band get dropped in one pass. Keep cross-tag dup detection at a moderate **0.97** (slightly more aggressive than 0.985, not catastrophically so). Raise `drop_cap_per_tag` to **0.30** so iter 1's expected ~33k drops can land in 2-3 iters instead of 5+. Drop cap kicks in at iter 4, which is fine — `pct_neg_margin` is already 0.026% by then.

The trick: **`hard_drop_thresh` is the dominant lever**. Cross-tag and merge thresholds have huge cascading effects when changed too far; the per-iter caps are safety rails that keep the cascade tame.

## Why Run C is the shipped default

Run C produces a tight dataset (`pct_neg_margin = 0.026%` — almost every surviving row has its own-tag as the clear E5 winner) at 37,355 rows. This is what production retrieval can route confidently.

Run A produces a larger dataset (70k) but with `pct_neg_margin = 16.1%` — production E5 will misroute about 1-in-6 queries even on this "clean" set. Acceptable for some applications; not what we want as the default.

Run B is the "more is better" failure mode — useful as a documented warning that aggressive defaults need careful per-lever calibration, not bulk increases.

## Tuning to a target row count

To reproduce or modify these results, override the relevant defaults in `RepairConfig`:

```python
from tagclean.repair import RepairConfig
cfg = RepairConfig()
cfg.hard_drop_thresh = -0.10  # Run A: keep ~70k
# OR
cfg.hard_drop_thresh = +0.004  # Run C: keep ~37k (current default)
# OR pick a value in between via the histogram in repair_thresholds.md
```

The histogram on the V3 (70k) corpus showed:
- `hard_drop_thresh = 0.0` → ~58k surviving
- `hard_drop_thresh = +0.005` → ~46k surviving
- `hard_drop_thresh = +0.008` → ~37k surviving

Interpolate to your target. If the target is far below 30k, expect Run-B-style cascade behavior — back off cross-tag dup and merge thresholds simultaneously.
