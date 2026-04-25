# tagclean — Harness Design

A standalone tool for cleaning tagged FAQ corpora where many tags are slight rewordings of each other. Built specifically for the Bangladesh Election Commission Bengali NID/voter FAQ dataset, where GPT-generated questions blur at tag boundaries (e.g. *self-service NID unlock* vs *retry-after-lock* vs *please-unlock-for-me*).

This document explains **why** the harness is shaped the way it is. For **how to run it**, see `docs/run_instructions.md`.

## The problem

GPT-generated FAQ data has two failure modes:

1. **Within a tag**, rows look fine — paraphrases of the same question.
2. **Across close-sibling tags**, rows leak — the model picks the wrong tag for a question that fits multiple intents.

A naive cleanup is "ask GPT for each row whether it fits its tag." That tried-and-discarded approach was:
- Expensive (one GPT call per row × 60k rows).
- Unreliable (~50% of packets had missing decisions; the harness silently jettisoned them).
- Dominated (the LLM's per-row score weighted 40% of the composite, making the embeddings cosmetic).

So the design constraint: **GPT must not be the per-row scorer.** Embeddings rank, GPT only writes policy and audits the top-N buffer.

Each stage writes parquet/jsonl artifacts; resume is hash-based. The full pipeline diagram lives in the README so the design doc stays focused on rationale, not flowcharts.

## Model roles

| Model | Role | Weight in ranking |
|---|---|---|
| **E5-multilingual-large-instruct** (1024d) | Primary geometry. Defines "tag fit". Production also uses E5 at inference, so we optimize against it. | 50% (own-sim 30 + margin 20) |
| **EmbeddingGemma-300m** (768d) | Independent second opinion. Catches E5 idiosyncrasies via rank disagreement. Used for cluster discovery. | 30% (own-sim 15 + margin 10 + agreement 10) |
| **GPT-5.5** (or 5.4) | (a) Boundary-rule author for close-tag clusters; (b) Auditor of the top-N buffer. Never scores rows. | 0% — appears only as a `keep`/`flag` filter on top-ranked rows |

## Stage-by-stage rationale

### Stage 0 — Intake & normalize
- Parse CSV (UTF-8-sig, strip BOM). Validate `question, tag` columns.
- Normalize for comparison only; **keep raw text intact** in outputs.
  - NFC Unicode composition; strip ZWJ/ZWNJ; canonicalize `য়/য়`, `ড়/ঢ়`; Bengali↔Latin digit fold; collapse whitespace; quote/punct fold.
  - Bengali normalization gated by `config.language` (default `bn`); pass `--language none` to skip.
- Reject empty/malformed rows.
- Exact dedup within tag.
- Cross-tag exact duplicates **logged**, not auto-deleted — they're a signal for Stage 3.

### Stage 1 — Dual embeddings
- Embed every row with both E5 and Gemma. Cached `.npy` files; ~80MB for 80k rows.
- E5 gets the instruct prefix (`config.e5_instruction`); Gemma takes raw text.
- FAISS `IndexFlatIP` written for both — small enough that exact search beats IVF on this corpus size.

### Stage 2 — Tag profile
- For each canonical tag T, compute:
  - Centroid in E5 + Gemma space.
  - **Medoid** (row closest to centroid) — robust to outliers.
  - **Top-K central rows** (5 nearest medoid).
  - **Discriminative phrases** — bigrams/trigrams with high log-odds vs neighbor tags.
- Trim outliers (top/bottom 5% by within-tag distance) before computing the centroid.

### Stage 3 — Tag-boundary policy
The bottleneck where embeddings alone aren't enough.

- `find_close_tag_clusters`: connected components where `cos(centroid_e5) ≥ 0.85 AND cos(centroid_gemma) ≥ 0.85`. Clusters of ≥2 tags become candidates.
- For each cluster, GPT-5.5 authors a `BoundaryPolicyResult`:
  - `one_line_intent` per tag.
  - `must_have_concepts` — 3–7 short cues a clean row should mention.
  - `must_avoid_concepts` — cues that signal a sibling tag instead.
- **Single pass, fixed tag order, high reasoning** — multi-pass self-consistency was tried and rejected because GPT phrases concepts differently across passes for close siblings, making the consistency check too strict.
- Validation: schema must validate AND the returned tag set must cover the input tags. Else fall back to a heuristic stub built from tag-name tokens.
- The `tag_answer.json`, when provided, gates merging: two tags merge only if questions look equivalent AND their canonical answers do too. Otherwise keep both and tighten the boundary.

### Stage 4 — Deterministic ranker
For each row in target scope, compute:

```
score(r) =
   0.30 · cos(r_e5,    centroid_e5(tag(r)))
 + 0.20 · margin_e5(r, tag)                         (own − nearest other-tag centroid)
 + 0.15 · cos(r_gemma, centroid_gemma(tag(r)))
 + 0.10 · margin_gemma(r, tag)
 + 0.10 · rank_agreement(E5, Gemma)                 (1 if both name the same competing tag)
 + 0.10 · token_alignment(r, must_have / must_avoid)  (from Stage 3 policy)
 − 0.05 · near_dup_count(r)
 − 0.10 · cross_tag_duplicate(r)
 − 0.10 · artifact_score(r)                         (short / repeat-char / synthetic)
```

- Weights chosen so a typical clean row lands near 1.0; ambiguous rows fall well below.
- Top `audit_buffer_size` rows per tag (default 80) are marked `audit_buffer=true`. Below-buffer rows are jettisoned by Stage 6.
- The earlier auto_clean/judge route concept is gone. Every row gets a deterministic score; nothing flows to GPT directly from Stage 4.

### Stage 5 — Buffer audit
- Per tag, group buffer rows into packets of `audit_rows_per_packet` (default 24).
- GPT-5.5 (Agents SDK or sync Responses API) gets:
  - The tag's `one_line_intent`, `must_have_concepts`, `must_avoid_concepts` (from Stage 3).
  - The tag's central exemplars and discriminative phrases (from Stage 2).
  - Up to 24 candidate rows.
- For each row: `keep` or `flag` with `reason_code` ∈ {`clean`, `wrong_intent`, `sibling_collision`, `too_generic`, `duplicate`, `synthetic_artifact`, `context_dependent`}.
- Policy: `prefer_flag_when_unsure`.
- **Missing-row retry**: if a packet response omits some `row_id`s, retry each missing row individually. Both packet and per-row results aggregate into `audit_results.jsonl`. Without this rescue, partial GPT responses turned into silent jettisons (the harness's earlier 261-row coverage gap).

### Stage 6 — Selection
- Filter: rows `in audit_buffer AND audit decision = keep`.
- MMR-adjust within each tag (λ=0.7 in E5 space) so top-K covers phrasing diversity, not 40 paraphrases of the medoid.
- Sort by `0.85 · composite_score + 0.15 · MMR`.
- Top `top_n` per tag (default 40) → `production_recommended=true`.

### Stage 7 — (disabled by default)
Was a per-row second-pass review over the bottom quartile of Stage 6 keeps. Superseded by Stage 5's buffer audit. Keep `review_enabled: false` unless you have reason to ramp.

### Stage 8 — Validation
- Build a shadow FAISS index from `cleaned.csv`.
- **Leave-one-out self-retrieval**: for each row, remove from index, query, check top-1 tag matches. Per-tag and global accuracy.
- Confusion matrix at top-1 and top-5; surface remaining boundary issues.
- Report low-support tags (<5 surviving rows).

## Decisions deliberately rejected

| Rejected | Why |
|---|---|
| Per-row GPT judging | Dominates the ranking, expensive, unreliable on partial responses. |
| First-3-row anchors per tag | The CSV is alphabetically sorted within tag, so "first 3" picks earliest-by-Bengali-sort questions, not canonical seeds. Medoid + top-K central is robust. |
| Multi-pass self-consistency with string-overlap agreement check | Too strict — GPT phrases concepts differently across passes for close siblings. Single pass + structural validation is sufficient. |
| Auto-cluster as default workflow | Footgun. Conflates discovery, prioritization, and execution. *"Hard to triage which clusters need attention vs which are fine."* |
| Use `tag_answer.json` as judge context for row decisions | Anchors GPT incorrectly. Reserved for the merge-safety gate only. |
| `--language` global toggle inside `normalize_question` | Threading config is invasive. Module-level `_NORMALIZATION_LANGUAGE` set by `main()` — small but pragmatic compromise. |

## Known risks

- **Cluster threshold can miss human siblings.** `boundary_policy_threshold` (default 0.85) is calibrated to tight Bengali NID clusters; English-style domains with looser sibling similarity may leave human-recognizable family members below the cut. Mitigation: `--seed-tag` prints the nearest excluded tags with similarity scores; lower the threshold or pass `--target-tags` explicitly when you see siblings just below the line.
- **Dirty centroids can canonize bad clusters.** Stage 2's medoid is computed AFTER outlier trimming, but if a tag is contaminated past 50%, the medoid itself reflects the contamination and Stage 3's policy will codify it. Mitigation: human-review the boundary policy for any tag that looks suspect; sample top-5 of `cleaned.csv` per tag.
- **Stage 8 LOO accuracy is supportive, not definitive.** It can look high (>0.95) even when the taxonomy is wrong, because each row is its own nearest neighbor in the cluster GPT chose. Always pair LOO with a manual eyeball of `top40.csv` and a sample of `jettisoned_rows.csv` rationales.

## Cost & runtime envelope

For a 3-tag close-cluster run on the Bengali EC FAQ data (~540 rows):
- Stage 1 embedding (E5 + Gemma on Mac MPS, models cached): ~30 sec.
- Stage 3 boundary policy: 1 GPT call (~60 sec).
- Stage 5 audit: ~10 packets × ~30–90 sec each = 5–15 min.
- Stages 0/2/4/6/8 deterministic: <30 sec total.
- **Total: ~6–17 min, ~$0.50 GPT spend.**

For a 60k-row corpus-wide run, multiply embedding by ~150× (still 30 min on a single L4 GPU; 30+ min on Mac MPS). Audit GPT cost scales with `audit_buffer_size × n_tags / audit_rows_per_packet` — for the full ~1394-tag Bengali corpus at default settings, around 4600 audit calls (~$50–150). **Don't run corpus-wide without explicit budget approval.**

## Schema details (pydantic)

All models forbid extra fields and have all properties listed in `required` (OpenAI strict-mode compliance — see Stage 3 schema bug history below).

```python
class TagBoundaryRule(BaseModel):
    tag: str
    one_line_intent: str
    must_have_concepts: list[str]   # required, can be []
    must_avoid_concepts: list[str]  # required, can be []

class BoundaryPolicyResult(BaseModel):
    cluster_id: str
    rules: list[TagBoundaryRule]
    cluster_rationale: str

class AuditRowDecision(BaseModel):
    row_id: int
    decision: Literal["keep", "flag"]
    reason_code: Literal["clean", "wrong_intent", "sibling_collision",
                          "too_generic", "duplicate", "synthetic_artifact",
                          "context_dependent"]
    rationale: str
```

### History note: the `default_factory` schema bug
Earlier versions had `must_have_concepts: list[str] = Field(default_factory=list)`. Pydantic generated a JSON schema with that field in `properties` but **not** in `required`. OpenAI's strict-mode response_format enforces "every property must be in required" and rejected the call with HTTP 400. Two real-GPT runs fell back to heuristic boundary policy before this was caught. Fix: never use `default_factory=` or `default=` on fields used as `output_type` for OpenAI structured-output calls.
