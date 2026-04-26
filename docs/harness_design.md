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

So the design constraint: **the LLM must not be the per-row scorer.** Embeddings rank, the LLM only writes policy and audits the top-N buffer.

Each stage writes parquet/jsonl artifacts; resume is hash-based. The full pipeline diagram lives in the README so the design doc stays focused on rationale, not flowcharts.

## Model roles

| Model | Role | Weight in ranking |
|---|---|---|
| **E5-multilingual-large-instruct** (1024d) | Primary geometry. Defines "tag fit". Production also uses E5 at inference, so we optimize against it. | 75% (own-sim 45 + margin 30) |
| **Token alignment** | Cross-validates LLM-authored boundary rules against row text. | 15% |
| **Claude (Opus, `--effort high`)** | Boundary-rule author for close-tag clusters (Stage 3). 1 call per cluster. Never scores rows. | 0% — appears only as `must_have_concepts` / `must_avoid_concepts` feeding token alignment |
| **Claude (Sonnet, `--effort medium`)** | Auditor of the top-N buffer (Stage 5). ~10 calls per family. | 0% — appears only as a `keep`/`flag` filter on top-ranked rows |

The single-embedding choice is intentional. The earlier design used EmbeddingGemma as an independent second opinion to catch E5 idiosyncrasies on shared Bengali NID vocabulary. Dropping it loses that signal *at the cluster-discovery level*; the replacement is multi-criteria E5-only gates: cosine + row-level kNN overlap inside `find_close_tag_clusters`, plus reciprocal top-K + pair-threshold inside `discover`. Stage 4 row scoring still benefits indirectly: a row that fails the deterministic ranker is rarely rescued by a second embedding; the original Gemma weights mostly correlated with E5.

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
- **Top-K central rows.** The 5 rows nearest the medoid. These are what Stage 3 (boundary policy) and Stage 5 (audit) show Claude as "this is what a clean example of T looks like."
- **Discriminative phrases.** Bigrams/trigrams in T's rows that have high log-odds against neighbor tags.

#### Why centroids are the right abstraction

The centroid lets us reduce a tag — typically 100–250 rows of paraphrased questions — to one point. From there, two things become trivial:

1. **Tag-to-tag similarity.** `cos(centroid(A), centroid(B))` answers "how close are these two intents in embedding space?" Used by `find_close_tag_clusters` and `discover` to find sibling families.
2. **Row-to-tag fit.** `cos(row_emb, centroid(tag))` answers "how well does this row belong to its assigned tag?" The composite score in Stage 4 is mostly this signal — own-tag cosine + margin against the nearest competing-tag centroid.

The 5%/5% outlier trim keeps the centroid stable when a small number of mis-tagged rows would otherwise drag it. If a tag is contaminated past ~50%, the centroid reflects the contamination and Stage 3's policy will codify the wrong intent — human review of the boundary policy is the mitigation there.

### Stage 3 — Tag-boundary policy
The bottleneck where embeddings alone aren't enough.

**Cluster discovery (within a single family run):**

- If `--target-tags` (or `--seed-tag` resolving to ≥2 tags) is supplied, **the user-provided tag set IS the cluster** — Stage 3 skips union-find and authors a policy for exactly those tags. This is the recommended path at corpus scale.
- Otherwise the legacy fallback applies: `find_close_tag_clusters` finds connected components where `cos(centroid_e5) ≥ 0.85` AND row-level `_knn_overlap ≥ knn_overlap_threshold`. The kNN-overlap predicate replaces the prior Gemma cosine as the second criterion that survives the dual-model drop.

**Boundary policy authoring:**

For each cluster, Claude (Opus, `--effort high`) authors a `BoundaryPolicyResult`:
- `one_line_intent` per tag.
- `must_have_concepts` — 3–7 short cues a clean row should mention.
- `must_avoid_concepts` — cues that signal a sibling tag instead.

Single pass, fixed tag order. Validation: schema must validate AND the returned tag set must cover the input tags. Else fall back to a heuristic stub built from tag-name tokens. The `tag_answer.json`, when provided, gates merging: two tags merge only if questions look equivalent AND their canonical answers do too. Otherwise keep both and tighten the boundary.

The call goes through `claude -p --json-schema`. Per-call cost / cache tokens / subtype are persisted to `stage3/llm_calls.jsonl`.

### Stage 4 — Deterministic ranker
For each row in target scope, compute:

```
score(r) =
   0.45 · cos(r_e5, centroid_e5(tag(r)))
 + 0.30 · margin_e5(r, tag)                         (own − nearest other-tag centroid)
 + 0.15 · token_alignment(r, must_have / must_avoid)  (from Stage 3 policy)
 − 0.05 · near_dup_count(r)
 − 0.10 · cross_tag_duplicate(r)
 − 0.10 · artifact_score(r)                         (short / repeat-char / synthetic)
```

- Weights chosen so a typical clean row lands near 1.0; ambiguous rows fall well below.
- Top `audit_buffer_size` rows per tag (default 80) are marked `audit_buffer=true`. Below-buffer rows are jettisoned by Stage 6.
- Gemma's prior 0.15 (own) + 0.10 (margin) + 0.10 (rank_agreement) was absorbed into E5 (0.30→0.45, 0.20→0.30) and token_alignment (0.10→0.15) so the positive weights still sum to ~1.0.

### Stage 5 — Buffer audit
- Per tag, group buffer rows into packets of `audit_rows_per_packet` (default 24).
- Claude (Sonnet, `--effort medium`) gets:
  - The tag's `one_line_intent`, `must_have_concepts`, `must_avoid_concepts` (from Stage 3).
  - The tag's central exemplars and discriminative phrases (from Stage 2).
  - Up to 24 candidate rows.
- For each row: `keep` or `flag` with `reason_code` ∈ {`clean`, `wrong_intent`, `sibling_collision`, `too_generic`, `duplicate`, `synthetic_artifact`, `context_dependent`, `audit_missing`}.
- Policy: `prefer_flag_when_unsure`.
- **Fail-closed on missing decisions**: if a packet response omits a buffer row, that row defaults to `flag` with `reason_code=audit_missing`. The earlier design defaulted missing rows to `keep` ("the ranker already endorsed them"); at corpus scale that turned silent rate-limit storms into thousands of false-positive inclusions.
- **Circuit breaker**: if the most recent N audit packets all errored (default N=10), the stage halts with a diagnostic and refuses to write `audit_results.jsonl`.
- **Per-call telemetry**: `subtype`, `total_cost_usd`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `num_turns`, `duration_ms` written to `stage5/llm_calls.jsonl`.

### Stage 6 — Selection
- Filter: rows `in audit_buffer AND audit decision = keep`.
- MMR-adjust within each tag (λ=0.7 in E5 space) so top-K covers phrasing diversity, not 40 paraphrases of the medoid.
- Sort by `0.85 · composite_score + 0.15 · MMR`.
- Top `top_n` per tag (default 40) → `production_recommended=true`.

### Stage 7 — (deleted)
Was a per-row second-pass review over the bottom quartile of Stage 6 keeps. Stage 5's buffer audit subsumes it. The dispatcher in `run_stage8` skips directly from Stage 6 to validation.

### Stage 8 — Validation
- Build a shadow FAISS index from `cleaned.csv`.
- **Leave-one-out self-retrieval**: for each row, remove from index, query, check top-1 tag matches. Per-tag and global accuracy.
- Confusion matrix at top-1 and top-5; surface remaining boundary issues.
- Report low-support tags (<5 surviving rows).

### Stage 9 — E5-only production-risk audit

Stage 4's ranker is E5 + token-alignment. Production inference uses E5 *alone*. So a row that scored well jointly but E5 alone misroutes is a production failure, not a cleaning failure. Stage 9 is the production-truth gate.

- For each row in the input set, run leave-one-out top-K retrieval over E5 (no token alignment, no Stage 3 policy). Default `K=10`.
- Default policy: drop rows whose top-1 non-self neighbor is a different tag.
- Severity diagnostics: per-row `own_share_top_K` (fraction of K neighbors with the same tag) and `neighbor_tag_dist` (full distribution).
- Pure-numpy chunked top-K (`vecs @ vecs.T` → `argpartition`, default chunk = 4096 rows). FAISS+sentence-transformers segfault when both load in the same process on Python 3.14/Mac.
- Single encode (passages = queries with same E5-instruct prefix); skip the redundant second encode.
- Standalone — never auto-chains stage0–8. Auto-chain silently overwrote `cleaned.csv` when `judge_mode` differed across invocations.

Stage 9 takes either the run's own `stage6/question_tag.cleaned.csv` (per-family diagnostics) or an external CSV via `--e5-audit-input` (the cross-family production union). The latter is the production exit shape.

## How `claude -p` is called

A single async helper (`tagclean.claude_cli.spawn_claude`) wraps every Stage 3 / Stage 5 invocation:

```
claude -p \
    --model {opus|sonnet} --fallback-model sonnet --effort {high|medium} \
    --output-format json --no-session-persistence \
    --json-schema '<pydantic-derived schema>'
# prompt streamed via stdin, response parsed from envelope.structured_output
```

The subprocess env strips `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_BASE_URL` so subscription routing wins. The envelope's `subtype` field is the canonical success signal — `is_error` alone is too narrow because Claude returns valid JSON envelopes with `subtype != "success"` for many real failures (`error_during_execution`, `error_max_structured_output_retries`, `error_max_turns`).

Failure classification:

| Class | Examples | Retry |
|---|---|---|
| INFRA | binary missing, perm denied, "not logged in" stderr | no |
| TRANSIENT | timeout, claude `subtype != success`, non-zero exit | yes |
| PARSE | envelope JSON unreadable | yes |

The Stage 5 audit loop wraps `spawn_claude` with `asyncio.Semaphore(concurrency=6)`, an atomic tmp+rename packet write, a per-call telemetry append, and a sliding-window failure circuit breaker.

## Resume hash hardening

`stage_done(stage_dir, expected_hash)` compares the pre-stage input hash against `manifest.json`. The historical Stage 3 / Stage 5 hashes only included data inputs — **not** model identity. That meant an OpenAI→Claude migration (or any prompt/effort change) silently reused stale boundary policies and audits. Both stages now bake in:

- `judge_mode`
- model name
- effort
- prompt-version constant (`STAGE3_PROMPT_VERSION` / `STAGE5_PROMPT_VERSION`)
- pydantic schema fingerprint (sha256 of the JSON schema, sorted-keys)
- packet geometry (Stage 5 only)

Bump the prompt-version constant when you change prompt text or schema shape.

## Scaling: from 3 hand-picked families to all 1394 tags

The original `--seed-tag` does union-find on tag centroids: connected components above threshold. At Bengali NID corpus scale (1394 tags, heavily shared vocabulary), this collapses into a single 488-tag mega-component — `max_cluster_size=6` then truncates the component by alphabetical tag-index order, returning a useless cluster that doesn't even contain the seed.

Replacement: **`tagclean discover`**, seed-centric reciprocal-NN ego-family expansion (E5-only).

1. Per tag, compute top-K centroid neighbors by `cos_e5`.
2. Edge kept iff reciprocal *and* `cos_e5 ≥ threshold` (default 0.88).
3. Per seed, ego-family = seed + reciprocal neighbors, capped at `max_family_size` (default 4), requiring pairwise `min_sim ≥ pair_threshold` (default 0.86) with every existing member.
4. Score each candidate by `(min_edge_sim, avg_edge_sim, size)`.
5. Greedy descending selection; mark members covered. Uncovered tags become singletons.

No transitive closure → no mega-components. Output is a hand-editable `families.yaml` with deterministic `family_id` (8-char hash of the sorted tag set) so re-running discover produces stable IDs and `run-families --skip-completed` resumes naturally.

The reciprocal top-K + pair-threshold gates are themselves multi-criteria robustness — they don't depend on Gemma; the prior dual-cosine `min(E5, Gemma) ≥ threshold` was an additional layer that's been collapsed.

Companion commands:

- **`tagclean run-families --manifest families.yaml`** dispatches `stage8` per approved family. Stage 0–2 are symlinked from the manifest's `centroids_run_id`, so the corpus-wide embeddings are computed once and reused.
- **`tagclean compose --manifest families.yaml --compose-source cleaned --out PATH`** unions per-family `stage6/cleaned.csv` (or `top40.csv`) into one production candidate set with `(question, tag)` dedup and deterministic ordering.
- **`tagclean stage9 --e5-audit-input PATH --run-id production`** does the final cross-family E5 audit.

Production end-state: `runs/<production_run>/stage9/production_filtered.csv`.

## Decisions deliberately rejected

| Rejected | Why |
|---|---|
| Per-row LLM judging | Dominates the ranking, expensive, unreliable on partial responses. |
| First-3-row anchors per tag | The CSV is alphabetically sorted within tag, so "first 3" picks earliest-by-Bengali-sort questions, not canonical seeds. Medoid + top-K central is robust. |
| Multi-pass self-consistency with string-overlap agreement check | Too strict — the LLM phrases concepts differently across passes for close siblings. Single pass + structural validation is sufficient. |
| Auto-cluster as default workflow | Footgun. Conflates discovery, prioritization, and execution. *"Hard to triage which clusters need attention vs which are fine."* |
| Use `tag_answer.json` as judge context for row decisions | Anchors the LLM incorrectly. Reserved for the merge-safety gate only. |
| `--language` global toggle inside `normalize_question` | Threading config is invasive. Module-level `_NORMALIZATION_LANGUAGE` set by `main()` — small but pragmatic compromise. |
| Tool-use / MCP for Stage 3 | Considered: let the boundary-policy author `Read` more sample rows on demand. Trades determinism + cache-keyability for context the model usually doesn't need. Pure structured-output transport wins for a batch tool. |
| Long-running stateful Claude session per family | Considered: most "harness-native." Loses within-family parallelism, harder failure semantics. |

## Known risks

- **Single-embedding cluster discovery may miss human siblings** that the prior dual-model setup caught via Gemma's independent geometry. Mitigation: row-level kNN-overlap is now an explicit second criterion in `find_close_tag_clusters`, and `discover`'s reciprocal + pair-threshold pipeline already provides multi-criteria robustness. Validate by comparing `runs/families_report.md` against the prior dual-model manifest if you have one.
- **Cluster threshold can miss human siblings.** `boundary_policy_threshold` (default 0.85) is calibrated to tight Bengali NID clusters; English-style domains with looser sibling similarity may leave human-recognizable family members below the cut. Mitigation: `--seed-tag` prints the nearest excluded tags with similarity scores; lower the threshold or pass `--target-tags` explicitly when you see siblings just below the line.
- **Dirty centroids can canonize bad clusters.** Stage 2's medoid is computed AFTER outlier trimming, but if a tag is contaminated past 50%, the medoid itself reflects the contamination and Stage 3's policy will codify it. Mitigation: human-review the boundary policy for any tag that looks suspect; sample top-5 of `cleaned.csv` per tag.
- **Stage 8 LOO accuracy is supportive, not definitive.** It can look high (>0.95) even when the taxonomy is wrong, because each row is its own nearest neighbor in the cluster GPT chose. Always pair LOO with a manual eyeball of `top40.csv` and a sample of `jettisoned_rows.csv` rationales.
- **Subscription rate limits.** The Claude Code subscription has per-minute quotas. Stage 5's circuit breaker (default: halt after 10 consecutive failures) catches sustained outages; for transient bursts, lower `concurrency` (default 6).

## Cost & runtime envelope

For a 3-tag close-cluster run on the Bengali EC FAQ data (~540 rows):
- Stage 1 embedding (E5 on Mac MPS, model cached): ~30 sec.
- Stage 3 boundary policy: 1 Claude (Opus) call (~30–90 sec).
- Stage 5 audit: ~10 packets × ~10–30 sec each = 2–5 min on Sonnet.
- Stages 0/2/4/6/8 deterministic: <30 sec total.
- **Total: ~4–10 min, billed against the user's Claude Code subscription.**

For a 60k-row corpus-wide run, multiply embedding by ~150× (still 30 min on a single L4 GPU; 30+ min on Mac MPS). Audit Claude cost scales with `audit_buffer_size × n_tags / audit_rows_per_packet` — for the full ~1394-tag Bengali corpus at default settings, around 4600 audit calls. Subscription quota and rate limits become the real bottleneck rather than dollar cost.

## Schema details (pydantic)

All models forbid extra fields and have all properties listed in `required` (Anthropic strict-mode `--json-schema` compliance — same constraint OpenAI strict mode had).

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
                          "context_dependent", "audit_missing"]
    rationale: str
```

### History note: the `default_factory` schema bug
Earlier versions had `must_have_concepts: list[str] = Field(default_factory=list)`. Pydantic generated a JSON schema with that field in `properties` but **not** in `required`. OpenAI's strict-mode `response_format` rejected the call with HTTP 400. Two real-GPT runs fell back to heuristic boundary policy before this was caught. Anthropic's `--json-schema` strict mode has the same constraint — never use `default_factory=` or `default=` on fields used as the response schema for an LLM structured-output call.
