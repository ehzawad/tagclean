# Repair Thresholds — Reference + Worked Examples

Every knob in `RepairConfig` (`src/tagclean/repair.py`), grouped by phase, with a concrete Bengali-NID example for each.

All thresholds are pure-geometry: they compare cosines of L2-normalized E5 vectors. No tag names, no `tag_answer.json`.

---

## Phase A — Reassignment

**The question this phase answers:** "Should this row move from its current tag to a different tag?"

### `reassign_thresh = -0.03`

A row is **eligible** to be considered for reassignment only if its margin (own-tag cosine minus best-other-tag cosine) is below this value.

**What it means:** "the row is closer to some other tag than its own."

**Example:**
```
Row R: "অ্যাকাউন্ট লক হয়ে গেছে কী করব"  (assigned to: account_locked)

  cos(R, centroid[account_locked])           = 0.86  ← own_sim
  cos(R, centroid[account_locked_followup_a]) = 0.91  ← best_other_sim

  margin = 0.86 - 0.91 = -0.05

  -0.05 < -0.03  →  ELIGIBLE for reassignment
```

If margin were e.g. -0.02, the row stays — too small a gap to act on; could be embedding noise.

### `delta_move = 0.05`

A row's best-other tag must beat its current tag's cosine by at least this amount. Same number as the eligibility margin in this case (the gap from own to best-other), but expressed as a positive direction.

**What it means:** "don't move on a tiny lead — wait for an actually-clear winner."

**Example:**
```
Row R, currently in tag_A:
  cos(R, centroid_A) = 0.86
  cos(R, centroid_B) = 0.88   ← best_other

  gap = 0.88 - 0.86 = 0.02

  0.02 < delta_move (0.05)  →  REJECT (move not clear enough)


Same row, but B is much closer:
  cos(R, centroid_A) = 0.86
  cos(R, centroid_B) = 0.94   ← best_other

  gap = 0.94 - 0.86 = 0.08

  0.08 > delta_move (0.05)  →  candidate passes this gate
```

### `delta_hyst = 0.02`

If the row's best-other tag is also the row's `prior_tag` (the tag it most recently came from), require an EXTRA `delta_hyst` on top of `delta_move`. Total: 0.07 gap to move BACK.

**What it means:** "don't ping-pong. Once you've moved, it's harder to undo."

**Example:**
```
Iter 1: row R moves from tag_A to tag_B.  prior_tag[R] = A.
Iter 2: same row R, now in B. Check: should it move back to A?

  cos(R, centroid_A) = 0.92   ← best_other (and == prior_tag)
  cos(R, centroid_B) = 0.86

  gap = 0.92 - 0.86 = 0.06

  Required: delta_move + delta_hyst = 0.05 + 0.02 = 0.07
  0.06 < 0.07  →  REJECT (don't ping back)


But if A really is dominantly closer:
  cos(R, A) = 0.95
  cos(R, B) = 0.86

  gap = 0.09 > 0.07  →  ACCEPT (move back is decisive)
```

### `knn_top_k = 10` and the kNN-majority gate (no separate threshold name)

For each candidate, compute the row's top-10 E5 nearest neighbors. Distribute their tags as a "share." Require: `share[best_other] > share[current]`.

**What it means:** "geometric centroid signal must be confirmed by the row's actual local neighborhood. A row tugged by one centroid but surrounded by neighbors of its current tag is ambiguous — don't move."

**Example:**
```
Row R in tag_A. Centroids say best_other = B.
Top-10 nearest E5 neighbors of R:
   8 are in tag_A      → share[A] = 0.8
   2 are in tag_B      → share[B] = 0.2
   share[B] (0.2) <= share[A] (0.8)  →  REJECT — neighborhood disagrees with centroid signal


Different scenario:
Top-10 nearest neighbors:
   2 are in tag_A      → share[A] = 0.2
   7 are in tag_B      → share[B] = 0.7
   1 is in tag_C       → share[C] = 0.1
   share[B] (0.7) > share[A] (0.2)  →  PASS (kNN agrees with centroid)
```

### `intake_cap_per_tag = 0.10`

A tag can receive at most this fraction of its current row count as reassignments INTO, per iter.

**What it means:** "don't let a tag be flooded by reassignments in one iter — its centroid would shift too much, making the next iter's signal stale."

**Example:**
```
Tag A has 100 rows currently.
Cap = 0.10 * 100 = 10 reassignments allowed INTO A this iter.

If 25 candidates want to move INTO A, the 10 with the strongest margin gap go through.
The other 15 are skipped this iter (may try again next iter if still eligible).
```

### `move_budget = 3`

Per-row counter. Each successful reassignment decrements it. At 0, the row is frozen for the rest of the run.

**What it means:** "a row that's bounced around 3 times is in an ambiguous zone — stop trying to place it, let it settle or get dropped."

**Example:**
```
Row R, original tag = A, move_budget = 3.
Iter 1: A → B (budget = 2)
Iter 2: B → C (budget = 1)
Iter 3: C → D (budget = 0)
Iter 4: still wants to move D → C (back). REJECTED — budget exhausted.
        Row R will stay in D for the rest of the run, even if its margin remains negative.
```

---

## Phase C — Merge tags

**The question this phase answers:** "Are these two tags geometrically the same intent? Should we collapse them?"

Only fires in **iter ≤ `merges_allowed_through_iter` (default 2)**. After iter 2, no more new merges.

### `merge_cosine_thresh = 0.92`

The CURRENT centroids of two candidate tags must be at least this cosine apart for them to be merge-eligible.

**What it means:** "tags must be geometrically very close to even consider merging."

**Example:**
```
Tag account_locked, centroid c_A
Tag account_locked_followup_a, centroid c_B

cos(c_A, c_B) = 0.94
0.94 > 0.92  →  PASS (geometric merge candidate)


Tag nid_fee, centroid c_C:
cos(c_A, c_C) = 0.78  →  REJECT (tags too geometrically different to merge)
```

### Frozen pairwise cosine ≥ 0.92 (hysteresis predicate)

In addition to the current centroids being close, the iter-0 frozen centroids must ALSO have been above threshold.

**What it means:** "the closeness must have been there at the start, not artificially created by reassignments dragging both centroids together."

**Example:**
```
Iter 0:  cos(c_A, c_B) was 0.93   ← above thresh ✓
Iter 1:  many reassignments
Iter 2:  cos(c_A, c_B) is now 0.95  ← still above thresh ✓
         AND iter-0 was also above thresh
         → eligible

Cascade-drift case (BLOCKED):
Iter 0:  cos(c_A, c_C) was 0.85   ← BELOW thresh
Iter 2:  cos(c_A, c_C) is now 0.93   ← above thresh, but...
         iter-0 was 0.85 < 0.92
         → REJECT — the closeness emerged from drift, not signal
```

### `mutual_confusion_thresh = 0.50` (in BOTH directions)

For each row in tag A, find its nearest non-A centroid. Compute fraction of A's rows whose nearest non-A centroid is B. That's `directed_confusion[A→B]`. Symmetrically `directed_confusion[B→A]`. Both must exceed 0.50.

**What it means:** "the confusion must be mutual and dominant — A's rows mostly point at B AND B's rows mostly point at A."

**Example:**
```
Tag A has 80 rows.
For each row in A, find nearest non-A centroid:
  60 rows → B, 15 → C, 5 → D
  directed_confusion[A→B] = 60/80 = 0.75 ✓ > 0.50

Tag B has 70 rows. Nearest non-B centroid for each:
  55 → A, 10 → C, 5 → D
  directed_confusion[B→A] = 55/70 = 0.79 ✓ > 0.50

Both directions > 0.50  →  PASS (mutual + dominant confusion)


Asymmetric case (BLOCKED):
A→B = 0.70 ✓
B→A = 0.30 ✗  →  REJECT — B's rows aren't actually drawn to A
```

### `merge_knn_overlap_thresh = 0.50`

Sample up to 50 rows from each of A and B. Compute pairwise top-10 E5 neighbors within the combined sample. Fraction that are CROSS-TAG (a B-row's neighbor is in A, or vice versa). Must exceed 0.50.

**What it means:** "row-level kNN must also agree — not just centroid geometry."

**Example:**
```
50 sampled rows from A + 50 sampled rows from B = 100 rows.
For each of the 100, find top-10 nearest within the 100.

Total neighbors checked = 100 * 10 = 1000.
Cross-tag neighbors: a B-row's top neighbor is in A (or vice versa) for 620 of them.
Overlap = 0.62 > 0.50  →  PASS

If overlap were e.g. 0.40 (too many neighbors stay within own sample):
→ REJECT — tags' rows don't actually mix in nearest-neighbor space
```

### `merges_allowed_through_iter = 2`

After iter 2, no more new merges fire. Rationale: late-iter merges are most likely artifacts of reassignment drift.

**Example:**
```
Iter 1: 40 merges proposed and applied.
Iter 2: 9 merges proposed and applied.
Iter 3: even if some pair passes all four merge gates, REJECT — past the cutoff.
```

### Canonical tiebreak (geometric composite)

When two tags merge, pick which one survives by:
```
score(t) = row_count(t)
         + 100 * mean(top quartile of cos(row, medoid))   ← neighborhood support
         - 50 * (1 - cos(medoid, centroid))                ← outlier penalty
```
Higher score wins. Tag names entirely ignored.

**Example:**
```
Tag account_locked: 250 rows, top-quartile mean cos = 0.94, medoid-centroid disagreement 0.02
  score = 250 + 100*0.94 - 50*0.02 = 250 + 94 - 1 = 343

Tag account_locked_followup_a: 35 rows, top-quartile mean cos = 0.91, disagreement 0.04
  score = 35 + 100*0.91 - 50*0.04 = 35 + 91 - 2 = 124

343 > 124  →  account_locked is canonical; account_locked_followup_a's rows merge in
```

---

## Phase B — Triage cross-tag near-duplicates + hard-margin drops

**The question this phase answers:** "These two rows are near-identical but in different tags. What do we do?"

### `cross_tag_dup_thresh = 0.985`

Two rows in different tags with cos(r_a, r_b) above this threshold trigger triage. Detected via FAISS `range_search`.

**What it means:** "near-duplicates across tag boundaries are the geometric evidence that the labels can't both be right."

**Example:**
```
Row r_a in tag_A: "অ্যাকাউন্ট লক হয়ে গেছে"
Row r_b in tag_B: "অ্যাকাউন্ট লকড হয়েছে"
cos(emb(r_a), emb(r_b)) = 0.991

0.991 > 0.985  →  triggers triage
```

### Triage logic (no separate threshold; uses `0.05` for own-sim margin)

Phase C runs FIRST. So if the pair's tags merged, the pair has already disappeared (both rows now in the canonical tag — same-tag, not cross-tag — and not flagged this iter).

For the pairs that REMAIN cross-tag after Phase C:

- **Case 1 — winner-take**: if `|own_sim(r_a) - own_sim(r_b)| > 0.05`, drop the loser:
  - If r_a's own_sim = 0.95 and r_b's own_sim = 0.85 → r_a wins, r_b dropped.
  - The 0.05 gap means one row clearly fits its tag better than the other does its own.

- **Case 2 — drop both** (last resort): no clear winner — neither row's own-tag fits dominantly. The pair is genuinely undecidable; drop both.

**Example (Case 1):**
```
Pair: r_a in tag_A (own_sim 0.95), r_b in tag_B (own_sim 0.83)
diff = |0.95 - 0.83| = 0.12 > 0.05
loser = r_b (lower own_sim)
→  KEEP r_a, DROP r_b

Both rows survived in the corpus until this point because their cos to their own
centroid was ≥ -0.10 (didn't trigger hard_drop). But now they collide cross-tag,
and one fits its tag much better.
```

**Example (Case 2):**
```
Pair: r_a own_sim 0.79, r_b own_sim 0.81
diff = 0.02 < 0.05
→  DROP BOTH (no clear winner; both rows fit their tags weakly)
```

### `hard_drop_thresh = -0.10`

Any kept row with margin below this is dropped outright (independent of cross-tag dup status).

**What it means:** "this row doesn't fit any tag well — it's either mis-tagged at source or genuinely off-topic. Geometry says no good home exists."

**Example:**
```
Row R in tag_account_locked:
  cos(R, centroid_account_locked) = 0.72
  cos(R, centroid_password_reset) = 0.83  ← best_other

  margin = 0.72 - 0.83 = -0.11

  -0.11 < hard_drop_thresh (-0.10)  →  DROP

Even though best_other = password_reset, the gap of 0.11 is large enough that this
row doesn't fit account_locked well, BUT also: by Phase A's gates, the row may have
already been reassigned earlier OR the kNN-majority might not have agreed with
the centroid signal. Whatever the reason, after Phase A, this row is still in
account_locked with margin -0.11 → drop.
```

### `drop_cap_per_tag = 0.20` and `drop_cap_strikes = 3`

A tag can lose at most 20% of its rows per iter. If the cap is hit, excess drops defer to next iter. After 3 consecutive iters of cap hits on the same tag, the loop **aborts**.

**What it means:** "if one tag is hemorrhaging, the loop's safety net stops the bleed; if the same tag keeps cap-hitting, the corpus is structurally beyond what the default thresholds can handle."

**Example:**
```
Tag A has 100 rows at the start of iter 1.
Iter 1: 35 rows in A would be dropped. cap = 0.20 * 100 = 20.
        20 dropped (worst margin first). 15 deferred to iter 2.
        cap_strikes[A] = 1.

Iter 2: A has 80 rows. Cap = 16. 28 want to drop. 16 dropped, 12 deferred.
        cap_strikes[A] = 2.

Iter 3: A has 64 rows. Cap = 13. 22 want to drop. 13 dropped, 9 deferred.
        cap_strikes[A] = 3.
        ABORT — loop_status = abort_drop_cap, exit at iter 3 cleanly.
```

This is exactly what happened on the Bengali corpus V1 run (8,545 drops, abort at iter 3).

---

## Phase D — Absorber detection + convergence

### `absorber_intake_thresh = 0.30` and `absorber_strikes_to_flag = 2`

A tag receiving > 30% of its size as reassignments-IN per iter, for 2 consecutive iters, is flagged "untrusted absorber."

**What it means:** "a tag pulling rows in faster than it can absorb them coherently is becoming a magnet — its centroid is drifting, attracting yet more rows. Break the loop by freezing its centroid."

**Example:**
```
Tag X has 50 rows at start of iter 1.
Iter 1: 18 rows reassigned INTO X. Intake fraction = 18/50 = 0.36 > 0.30 ✓
        absorber_history[X] = [0.36]
Iter 2: X now has 68 rows. 22 more reassigned IN. Intake = 22/68 = 0.32 > 0.30 ✓
        absorber_history[X] = [0.36, 0.32]
        Two consecutive strikes  →  FLAG X as untrusted absorber.
        Action:
          - X's centroid replaced with iter-0 frozen centroid
          - X excluded as reassignment target until intake drops below 0.10
```

### Convergence: `convergence_change_frac = 0.001`, `convergence_required_consec = 2`

Loop converges when total moves (reassign + drop + merge) per iter < 0.1% of total rows for 2 consecutive iters.

**Example (the V3 run on the Bengali corpus):**
```
Iter 1: 174 dropped, 14 merged, 0 reassigned. Total = 188. 188/70172 = 0.27% > 0.1% ✗
Iter 2: 32 dropped, 8 merged, 0 reassigned. Total = 40. 40/70172 = 0.057% < 0.1% ✓ (1 strike)
Iter 3: 8 dropped, 0 merged, 0 reassigned. Total = 8. 8/70172 = 0.011% < 0.1% ✓ (2 strikes)
        →  CONVERGED, exit loop with status `converged`
```

### Oscillation abort: `pct_neg_margin` rises 2 consec iters

If `pct_neg_margin = (margin < 0).mean()` increases for 2 iters straight, the loop aborts. This is the "the gates aren't producing monotone improvement" diagnostic.

**Why it matters:** the loop has no formal convergence proof (discrete reassignment can perturb other rows' margins). This signal catches pathological cases where moves make things worse.

### `max_iter = 8`

Hard ceiling. With move_budget=3 and damping schedule, most rows make their final move by iter 4-5. Hitting iter 8 without converging is a sign of either a stuck oscillation or a truly hard corpus.

### `min_tag_rows = 3`

After the loop ends, any tag with fewer than 3 rows is dissolved — its remaining rows are dropped.

**Example:**
```
Tag X: 2 rows survive after all reassign/drop/merge.
2 < min_tag_rows (3)  →  X dissolved, 2 rows get status = dropped.
```

---

## Cold-start triggers (iter 0 only)

### `medoid_diversity_trigger = 0.40`

A tag is "outlier-heavy" if `e5_diversity > 0.40`. e5_diversity = `1 - mean(cos(row, centroid))`.

**What it means:** "the tag's rows scatter widely from their mean — using the trimmed mean as a centroid would chase contamination. Use medoid bootstrapping instead."

**Example:**
```
Tag A: 100 rows. Trimmed-mean centroid c_A.
For each row r in A, cos(r, c_A) measured.
Mean of those cosines = 0.55.
e5_diversity = 1 - 0.55 = 0.45 > 0.40  →  OUTLIER-HEAVY

Cold-start path activated:
  - Compute medoid (row whose mean cos to other A-rows is highest)
  - Find core: rows with cos(r, medoid) > 0.85
  - Set centroid_A = mean of core (not all of A)
  - Mark A as "untrusted" — excluded as reassignment target in iter 1
```

### `medoid_centroid_disagreement_trigger = 0.15`

Alternative outlier-heavy trigger: `1 - cos(medoid, trimmed_centroid) > 0.15`.

**What it means:** "if the medoid disagrees with the trimmed mean, the tag is bimodal or contaminated."

**Example:**
```
Tag B: medoid = m, trimmed centroid = c.
cos(m, c) = 0.83
disagreement = 1 - 0.83 = 0.17 > 0.15  →  OUTLIER-HEAVY (cold-start)
```

---

## Summary table (all thresholds, defaults)

| Phase | Knob | Default | Lower = | Higher = |
|---|---|---:|---|---|
| A | `reassign_thresh` | -0.03 | More aggressive reassign | More conservative |
| A | `delta_move` | 0.05 | More moves accepted | Stricter "clear winner" |
| A | `delta_hyst` | 0.02 | Easier ping-pong | Stronger lock-in |
| A | `knn_top_k` | 10 | Tighter neighborhood | Broader, more stable |
| A | `intake_cap_per_tag` | 0.10 | Slower intake per iter | Faster (more drift) |
| A | `move_budget` | 3 | Less ping-pong | More room to settle |
| C | `merge_cosine_thresh` | **0.90** | More merges (more aggressive collapse) | Fewer merges |
| C | `mutual_confusion_thresh` | 0.50 | Easier merges | Stricter mutual evidence |
| C | `merge_knn_overlap_thresh` | 0.50 | Easier merges | Stricter row-level evidence |
| C | `merges_allowed_through_iter` | 2 | Less time for merges | More time (more cascade risk) |
| B | `cross_tag_dup_thresh` | **0.97** | More pairs flagged | Fewer pairs |
| B | `hard_drop_thresh` | **+0.004** | More aggressive drops | More conservative |
| B | `drop_cap_per_tag` | **0.30** | Slower drops per tag | Faster |
| B | `drop_cap_strikes` | **4** | Earlier abort | More patience |
| D | `absorber_intake_thresh` | 0.30 | Easier to flag absorbers | Harder |
| D | `absorber_strikes_to_flag` | 2 | Faster flag | Slower (more evidence required) |
| D | `convergence_change_frac` | 0.001 | Looser convergence | Tighter |
| D | `convergence_required_consec` | 2 | Faster exit | More confidence |
| Loop | `max_iter` | 8 | — | — |
| Loop | `min_tag_rows` | 3 | Smaller tags survive | Larger min size |
| Cold start | `medoid_diversity_trigger` | 0.40 | More tags use cold-start | Fewer |
| Cold start | `medoid_centroid_disagreement_trigger` | 0.15 | More tags use cold-start | Fewer |

**Defaults shipped in this branch are aggressive** — tuned for the Bengali NID corpus to land near 40k surviving rows from the 79k raw input. The thresholds in **bold** above were raised from earlier conservative defaults (`hard_drop_thresh: -0.10 → +0.004`, `cross_tag_dup_thresh: 0.985 → 0.97`, `merge_cosine_thresh: 0.92 → 0.90`, `drop_cap_per_tag: 0.20 → 0.30`). To get 70k surviving rows instead, override `hard_drop_thresh` back to -0.10.

## To loosen further (push past the 16% pct_neg_margin floor)

The `xray_cleanup.csv` in this repo was produced at default thresholds. To clean more aggressively:

1. **`hard_drop_thresh`**: -0.10 → -0.05. Drops more rows that fit weakly. Trade-off: lose more data.
2. **`merge_cosine_thresh`**: 0.92 → 0.88. Merges more tags. Trade-off: more semantic-distinction loss.
3. **`drop_cap_per_tag`**: 0.20 → 0.30, with `drop_cap_strikes` raised to 5. Lets tags shed more rows per iter. Trade-off: bigger swings, less stable convergence.
4. **`merges_allowed_through_iter`**: 2 → 4. Merges late-iter close pairs. Trade-off: more cascade risk.

To be MORE conservative:

1. **`hard_drop_thresh`**: -0.10 → -0.15. Drops only the most obvious outliers.
2. **`mutual_confusion_thresh`**: 0.50 → 0.65. Requires stronger mutual evidence to merge.
3. **`cross_tag_dup_thresh`**: 0.985 → 0.99. Triages only the very tight cross-tag pairs.

Defaults are calibrated for Bengali NID + E5-multilingual-large-instruct, where typical close-tag-pair cosines run 0.85-0.95. English or other languages may need recalibration — start by examining the histogram of pairwise tag-centroid cosines on a 5k-row subset before tuning.
