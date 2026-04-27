# tagclean — Harness Design

A standalone tool for cleaning tagged FAQ corpora where many tags are slight rewordings of each other. Built specifically for the Bangladesh Election Commission Bengali NID/voter FAQ dataset, where GPT-generated questions blur at tag boundaries (e.g. *self-service NID unlock* vs *retry-after-lock* vs *please-unlock-for-me*).

This document explains **why** the harness is shaped the way it is. For **how to run it**, see `docs/run_instructions.md`.

## The problem

GPT-generated FAQ data has two failure modes:

1. **Within a tag**, rows look fine — paraphrases of the same question.
2. **Across close-sibling tags**, rows leak — the model picks the wrong tag for a question that fits multiple intents.

A naive cleanup is "ask the LLM for each row whether it fits its tag." That tried-and-discarded approach was:
- Expensive (one LLM call per row × 60k rows).
- Unreliable (~50% of packets had missing decisions; the harness silently jettisoned them).
- Dominated (the LLM's per-row score weighted 40% of the composite, making the embeddings cosmetic).

Subsequent designs experimented with two narrower LLM roles — Claude writing per-cluster boundary policies (Stage 3) and Claude reviewing the deterministic top-N output (Stage QA / Stage 5). Both are removed in this branch. The decision is documented under [Why the LLM was removed entirely](#why-the-llm-was-removed-entirely).

So the design constraint on this branch: **no LLM calls anywhere.** Embeddings rank, deterministic heuristics filter, the production-truth audit drops what production E5 itself would misroute.

Each stage writes parquet/jsonl artifacts; resume is hash-based. The pipeline diagram lives in the README.

## Model roles

| Component | Role | Weight in ranking |
|---|---|---|
| **E5-multilingual-large-instruct** (1024d) | Primary geometry. Defines "tag fit". Production also uses E5 at inference, so we optimize against it. | 90% (own-sim 55 + margin 35) |
| **Penalties** | near_duplicate, cross_tag_duplicate, artifact_score | -5%, -10%, -10% |
| **MMR** | Within-tag re-rank for the top-N (λ=0.7) | 15% post-rerank |

The single-embedding choice is intentional. The earlier design used EmbeddingGemma as an independent second opinion to catch E5 idiosyncrasies on shared Bengali NID vocabulary. Dropping it loses that signal *at the cluster-discovery level*; the replacement is multi-criteria E5-only gates: cosine + row-level kNN overlap inside `find_close_tag_clusters`, plus reciprocal top-K + pair-threshold inside `discover`.

## Stage-by-stage rationale

### Stage 0 — Intake & normalize
- Parse CSV (UTF-8-sig, strip BOM). Validate `question, tag` columns.
- Normalize for comparison only; **keep raw text intact** in outputs.
  - NFC Unicode composition; strip ZWJ/ZWNJ; canonicalize `য়/য়`, `ড়/ঢ়`; Bengali↔Latin digit fold; collapse whitespace; quote/punct fold.
  - Bengali normalization gated by `config.language` (default `bn`); pass `--language none` to skip.
- Reject empty/malformed rows.
- Exact dedup within tag.
- Cross-tag exact duplicates **logged**, not auto-deleted — they're a signal for Stage 3.

### Stage 1 — Embeddings
- Embed every row with E5. Cached `.npy` files; ~80MB for 80k rows.
- E5 gets the instruct prefix (`config.e5_instruction`).
- FAISS `IndexFlatIP` written for E5 — small enough that exact search beats IVF on this corpus size.
- A second `emb_e5_query.npy` file is written as a soft alias of `emb_e5.npy` so Stage 8's LOO retrieval has stable file names; query and passage prefixes are identical in this codebase.

### Stage 2 — Tag profile

This is the bedrock geometry layer: every tag becomes a point in E5 space. Per-tag points are then used by every downstream stage.

For each canonical tag T:

- **Centroid (E5).** Take all of T's rows, embed each, and average the L2-normalized vectors. Outlier-trimmed: drop the top/bottom 5% by within-tag distance before averaging, so a few mis-tagged rows can't drag the centroid into a sibling's territory.
- **Medoid.** The actual row whose embedding is closest to the centroid.
- **Top-K central rows.** The 5 rows nearest the medoid. Used as illustrative anchors when humans inspect a tag.
- **Discriminative phrases.** Bigrams/trigrams in T's rows that have high log-odds against neighbor tags.

#### Why centroids are the right abstraction

The centroid lets us reduce a tag — typically 100–250 rows of paraphrased questions — to one point. From there, two things become trivial:

1. **Tag-to-tag similarity.** `cos(centroid(A), centroid(B))` answers "how close are these two intents in embedding space?" Used by `find_close_tag_clusters` and `discover` to find sibling families.
2. **Row-to-tag fit.** `cos(row_emb, centroid(tag))` answers "how well does this row belong to its assigned tag?" The composite score in Stage 4 is mostly this signal — own-tag cosine + margin against the nearest competing-tag centroid.

The 5%/5% outlier trim keeps the centroid stable when a small number of mis-tagged rows would otherwise drag it. If a tag is contaminated past ~50%, the centroid reflects the contamination — Stage 9's production-truth audit is the safety net.

### Stage 3 — Deterministic merge map

**Cluster discovery (within a single family run):**

- If `--target-tags` (or `--seed-tag` resolving to ≥2 tags) is supplied, **the user-provided tag set IS the cluster** — Stage 3 honors that scope directly.
- Otherwise the legacy fallback applies: `find_close_tag_clusters` finds connected components where `cos(centroid_e5) ≥ 0.85` AND row-level `_knn_overlap ≥ knn_overlap_threshold`. The kNN-overlap predicate is the second criterion that survives the dual-model drop.

**Merge decisions:**

For each candidate pair within a cluster, a deterministic check decides whether to merge tags into one canonical tag:

- Token overlap on tag names (e.g. `nid_fee` vs `nid_fee_01_followup_a` → same base).
- Answer-token overlap on `tag_answer.json` (when provided), as a merge-safety gate — two tags merge only if questions look equivalent AND their canonical answers do too.

The result is `tag_merge_map.csv` (`old_tag → canonical_tag`). No LLM call; the output is fully reproducible from the corpus + config.

### Stage 4 — Deterministic ranker

For each row in target scope, compute:

```
composite_score(r) =
   0.55 · cos(r_e5, centroid_e5(tag(r)))            (E5 own-similarity)
 + 0.35 · margin_e5(r, tag)                         (own − nearest other-tag centroid)
 − 0.05 · near_dup_count(r)
 − 0.10 · cross_tag_duplicate(r)
 − 0.10 · artifact_score(r)                         (short / repeat-char / synthetic)
```

Weights chosen so a typical clean row lands near 1.0; ambiguous rows fall well below. The previous design's token-alignment feature (15%) was deleted along with the boundary-policy authoring it depended on; that weight was redistributed into E5 own-similarity (45 → 55) and margin (30 → 35).

### Stage 5 — (deleted)

Was the Stage QA reviewer pass: per-tag anonymized prompt to Claude that flagged obvious outliers in the deterministic top-N. See [Why the LLM was removed entirely](#why-the-llm-was-removed-entirely).

### Stage 6 — Selection

- Filter: rows in scope (no Stage QA pre-filter; the deterministic ranker decides).
- MMR-adjust within each canonical tag (λ=0.7 in E5 space) so the top-K covers phrasing diversity, not 40 paraphrases of the medoid.
- Sort by `0.85 · composite_score + 0.15 · MMR`.
- Top `top_n` per tag (default 40) → `production_recommended=true`.

### Stage 7 — (deleted)

Was a per-row second-pass review over the bottom quartile of Stage 6 keeps. Removed in an earlier branch.

### Stage 8 — Validation

- Build a shadow FAISS index from `cleaned.csv`.
- **Leave-one-out self-retrieval**: for each row, remove from index, query, check top-1 tag matches. Per-tag and global accuracy.
- Confusion matrix at top-1 and top-5; surface remaining boundary issues.
- Report low-support tags (<5 surviving rows).

### Stage 9 — E5-only production-risk audit

Stage 4's ranker uses E5 plus deterministic penalties. Production inference uses E5 *alone*. So a row that scored well composite but E5 alone misroutes is a production failure, not a cleaning failure. Stage 9 is the production-truth gate.

- For each row in the input set, run leave-one-out top-K retrieval over E5 (no penalties, no Stage 3 merge map). Default `K=10`.
- Default policy: drop rows whose top-1 non-self neighbor is a different tag.
- Severity diagnostics: per-row `own_share_top_K` (fraction of K neighbors with the same tag) and `neighbor_tag_dist` (full distribution).
- Pure-numpy chunked top-K (`vecs @ vecs.T` → `argpartition`, default chunk = 4096 rows). FAISS+sentence-transformers segfault when both load in the same process on Python 3.14/Mac.
- Single encode (passages = queries with same E5-instruct prefix); skip the redundant second encode.
- Standalone — never auto-chains stage0–8. Auto-chain silently overwrote `cleaned.csv` when geometry config differed across invocations.

Stage 9 takes either the run's own `stage6/question_tag.cleaned.csv` (per-family diagnostics) or an external CSV via `--e5-audit-input` (the cross-family production union). The latter is the production exit shape.

## Why the LLM was removed entirely

The branch history is:

1. **Stage QA branch** (`claude-cli-stage-qa`): a single anonymized Claude pass per tag, post-Stage-6, flagged obvious outliers in the deterministic top-N. Worked. Empirically improved cleanliness for 157/251 families on the Bengali corpus before subscription rate limits stopped 94 families mid-run.
2. **This branch** (`deterministic-only`): drop Stage QA. Ship the deterministic top-N + global Stage 9 audit.

The trade-off is small and deliberate. Stage QA's incremental contribution on the Bengali NID corpus was modest paraphrase removal — useful, but not load-bearing for production retrieval accuracy (which is dominated by E5 geometry plus the Stage 9 production-truth audit). Holding that against:

- Subscription rate limits stalling corpus-scale runs.
- Non-determinism — running the same input twice does not produce identical output.
- Per-call telemetry, breakers, retries, vacuity checks, prompt-version hashes — all infrastructure that exists only because the LLM is on the hot path.

…the deterministic pipeline is preferable when reproducibility and operational simplicity outweigh the last few percent of paraphrase polish.

What's lost: a domain-drift safety valve. Deterministic geometry will silently canonize whatever the corpus happens to contain. If tags get noisier, new services appear, or phrasing shifts, the pipeline cannot say "this row is obviously broken/off-intent." Stage 9 catches embedding-routable errors but not semantic-but-still-embeds-plausibly errors. Mitigation: human spot-check `stage6/top40.csv` and `stage9/e5_dropped.csv` per family.

## Resume hash hardening

`stage_done(stage_dir, expected_hash)` compares the pre-stage input hash against `manifest.json`. Each stage's input hash bakes in:

- File hashes of all input CSVs / parquets / numpy arrays.
- Geometry config (`e5_merge_threshold`, `knn_overlap_threshold`, `near_duplicate_threshold`, `top_n`, etc.).
- Scope when applicable (`target_tags` set hash for target-restricted runs).

A target-restricted run is treated as a different artifact than a corpus-wide one and must not silently resume from each other.

## Scaling: from 3 hand-picked families to all tags

The original `--seed-tag` does union-find on tag centroids: connected components above threshold. At Bengali NID corpus scale (1394 tags, heavily shared vocabulary), this collapses into a single ~488-tag mega-component — `max_cluster_size=6` then truncates the component by alphabetical tag-index order, returning a useless cluster that doesn't even contain the seed.

Replacement: **`tagclean discover`**, seed-centric reciprocal-NN ego-family expansion (E5-only).

1. Per tag, compute top-K centroid neighbors by `cos_e5`.
2. Edge kept iff reciprocal *and* `cos_e5 ≥ threshold` (default 0.88).
3. Per seed, ego-family = seed + reciprocal neighbors, capped at `max_family_size` (default 4), requiring pairwise `min_sim ≥ pair_threshold` (default 0.86) with every existing member.
4. Score each candidate by `(min_edge_sim, avg_edge_sim, size)`.
5. Greedy descending selection; mark members covered. Uncovered tags become singletons.

No transitive closure → no mega-components. Output is a hand-editable `families.yaml` with deterministic `family_id` (8-char hash of the sorted tag set) so re-running discover produces stable IDs and `run-families --skip-completed` resumes naturally.

Companion commands:

- **`tagclean run-families --manifest families.yaml`** dispatches `stage8` per approved family. Stage 0–2 are symlinked from the manifest's `centroids_run_id`, so the corpus-wide embeddings are computed once and reused.
- **`tagclean compose --manifest families.yaml --compose-source top40 --out PATH`** unions per-family `stage6/top40.csv` into one production candidate set with `(question, tag)` dedup and deterministic ordering.
- **`tagclean stage9 --e5-audit-input PATH --run-id production`** does the final cross-family E5 audit.

Production end-state: `runs/<production_run>/stage9/production_filtered.csv`.

## Decisions deliberately rejected

| Rejected | Why |
|---|---|
| Per-row LLM judging | Dominates the ranking, expensive, unreliable on partial responses. |
| LLM-authored boundary policy (Stage 3) | The policy artifact was contaminated by tag-name anchoring; deterministic merge map is more reproducible. |
| LLM Stage QA reviewer (Stage 5) | Subscription rate limits and non-determinism outweighed incremental polish at corpus scale. See [Why the LLM was removed entirely](#why-the-llm-was-removed-entirely). |
| First-3-row anchors per tag | The CSV is alphabetically sorted within tag, so "first 3" picks earliest-by-Bengali-sort questions, not canonical seeds. Medoid + top-K central is robust. |
| Auto-cluster as default workflow | Footgun. Conflates discovery, prioritization, and execution. *"Hard to triage which clusters need attention vs which are fine."* |
| Use `tag_answer.json` as judge context for row decisions | Anchors the LLM (and any future deterministic per-row scorer) incorrectly. Reserved for the merge-safety gate only. |
| `--language` global toggle inside `normalize_question` | Threading config is invasive. Module-level `_NORMALIZATION_LANGUAGE` set by `main()` — small but pragmatic compromise. |

## Known risks

- **Single-embedding cluster discovery may miss human siblings** that the prior dual-model setup caught via Gemma's independent geometry. Mitigation: row-level kNN-overlap is now an explicit second criterion in `find_close_tag_clusters`, and `discover`'s reciprocal + pair-threshold pipeline already provides multi-criteria robustness. Validate by inspecting `runs/families_report.md`.
- **Cluster threshold can miss human siblings.** `boundary_policy_threshold` (default 0.85) is calibrated to tight Bengali NID clusters; English-style domains with looser sibling similarity may leave human-recognizable family members below the cut. Mitigation: `--seed-tag` prints the nearest excluded tags with similarity scores; lower the threshold or pass `--target-tags` explicitly when you see siblings just below the line.
- **Dirty centroids can canonize bad clusters.** Stage 2's medoid is computed AFTER outlier trimming, but if a tag is contaminated past 50%, the medoid itself reflects the contamination. Mitigation: human-review `top40.csv` per tag; sample dropped rows in `stage9/e5_dropped.csv`.
- **Stage 8 LOO accuracy is supportive, not definitive.** It can look high (>0.95) even when the taxonomy is wrong, because each row is its own nearest neighbor in the cluster GPT chose. Always pair LOO with a manual eyeball of `top40.csv`.
- **No semantic safety valve.** The deterministic pipeline cannot catch rows that embed plausibly but are off-intent. If domain drifts, regenerate the corpus and re-run; do not rely on this pipeline to flag drift.

## Cost & runtime envelope

For a 3-tag close-cluster run on the Bengali EC FAQ data (~540 rows):
- Stage 1 embedding (E5 on Mac MPS, model cached): ~30 sec.
- Stages 0/2/3/4/6/8 deterministic: <30 sec total.
- Stage 9 deterministic E5 audit: <10 sec for the family.
- **Total: <2 min, no API spend.**

For a 60k-row corpus-wide run, multiply embedding by ~150× (~30 min on a single L4 GPU; 30+ min on Mac MPS). Subsequent per-family stage4/6/8 runs are CPU-bound and parallelizable across families.
