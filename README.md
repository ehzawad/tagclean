# tagclean

LLM-assisted dataset cleaner for tagged FAQ corpora. Built specifically for the
Bangladesh Election Commission Bengali NID/voter FAQ domain (1395 tags, 79k
GPT-generated rows), but the harness is language-neutral when you set
`language: none`.

You give it `question_tag.csv` (and optionally `tag_answer.json`); it gives back
a cleaned production subset where ambiguous, duplicated, and synthetic-looking
rows are jettisoned and the survivors are filtered against the model
production actually uses at inference (E5).

## The problem

GPT-generated FAQ data has two failure modes:

1. **Within a tag**, rows look fine — they're paraphrases of the same question.
2. **Across close-sibling tags**, rows leak — the model picks the wrong tag for
   a question that legitimately fits multiple intents.

Bengali NID/voter is the hard case. Many tags are slight rewordings of each
other:

- `account_locked` vs `account_locked_retrials` vs `account_locked_unlock_request` — self-help vs retry-after-correct-info vs ask-an-agent-to-unlock.
- `name_correction_in_nid_card` vs `parents_name_correction_new` vs `spouse_name_correction_new` — same correction-of-name intent, different person-of-reference.
- `otp_not_received` vs `otp_delivery_time` vs `otp_send_button` — same OTP cluster, different sub-intent.

The earlier "ask GPT for each row whether it fits its tag" attempt was
expensive (one call × 60k rows), unreliable (~50% of packets had missing
decisions), and made the embedding ranker cosmetic. So the design constraint:
**GPT is never the per-row scorer.** Embeddings rank, GPT writes policy and
audits the top-N buffer.

A second realization came late: production inference uses E5 alone. Anything
the dual-embedding ranker scored well but E5 alone misroutes is a *production*
failure, not a cleaning failure. So the cleaned set has to pass an E5-only
final filter (Stage 9).

## The solution

```
question_tag.csv  →  Stage 0  intake / Bengali normalize / dedup
                  →  Stage 1  E5 + Gemma embeddings + FAISS
                  →  Stage 2  per-tag medoid + central rows + discriminative phrases
                  →  Stage 3  GPT writes per-tag boundary rules for close-tag clusters
                  →  Stage 4  deterministic ranker → top-N audit buffer per tag
                  →  Stage 5  GPT audits the buffer (keep / flag with reasons)
                  →  Stage 6  walk audited-pass rows, MMR-diversify, take top K
                  →  Stage 7  optional second-pass review (default OFF)
                  →  Stage 8  leave-one-out validation
                  →  Stage 9  E5-only production-risk audit + filter

For multi-family production composition:

  family_A/stage6/top40.csv ┐
  family_B/stage6/top40.csv ┼─→ tagclean compose → composed.csv → stage9 → production_filtered.csv
  family_C/stage6/top40.csv ┘
```

GPT cost for a 3-tag cluster: ~12 calls (~1 boundary policy + ~10–12 audit
packets). Embedding cost is GPU/CPU time, not API.

## Roles

| Component | Role | Weight in ranking |
|---|---|---|
| **E5-multilingual-large-instruct** (1024d) | Primary geometry. Same model production uses at inference. | 0.30 (cos) + 0.20 (margin) = 0.50 |
| **EmbeddingGemma-300m** (768d) | Independent second opinion. Catches E5 idiosyncrasies via rank disagreement. | 0.15 (cos) + 0.10 (margin) = 0.25 |
| Cross-validation features | rank_agreement (E5↔Gemma), token_alignment (must_have/avoid from policy) | 0.10 + 0.10 = 0.20 |
| Penalties | near_duplicate, cross_tag_duplicate, artifact_score | -0.05, -0.10, -0.10 |
| **GPT-5.5** (high reasoning) | Boundary-rule author (Stage 3) + buffer auditor (Stage 5). Never scores rows. | 0% direct |

## Quickstart

```bash
git clone <this repo> tagclean
cd tagclean
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q                              # 17 unit tests, all offline (no API)
```

Real run with GPT-authored boundary policy + audit:

```bash
export OPENAI_API_KEY=sk-...
cp configs/example.yaml my-config.yaml
# edit my-config.yaml: input_csv, tag_answer_json, run_id, e5_instruction
tagclean stage8 --config my-config.yaml --judge-mode sync --openai-model gpt-5.5
```

## End-to-end production recipe

This is the full path from raw `question_tag.csv` to the final production CSV
that gets indexed by E5 at inference time. We validated this on three
3-tag close-cluster families on the Bengali EC/NID corpus.

```bash
export OPENAI_API_KEY=sk-...
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 1. Clean each close-tag family separately. One stage8 invocation per family.
#    Use --target-tags to specify the cluster explicitly (recommended at corpus
#    scale — see "known limitations" for why --seed-tag falls short on big
#    corpora). --judge-mode sync is more reliable than agents for the Stage 3
#    boundary policy GPT call.
tagclean stage8 \
    --config configs/bn_full.yaml \
    --target-tags account_locked,account_locked_retrials,account_locked_unlock_request \
    --run-id bn_account_locked \
    --judge-mode sync --openai-model gpt-5.5

tagclean stage8 \
    --config configs/bn_full.yaml \
    --target-tags name_correction_in_nid_card,parents_name_correction_new,spouse_name_correction_new \
    --run-id bn_name_correction \
    --judge-mode sync --openai-model gpt-5.5

tagclean stage8 \
    --config configs/bn_full.yaml \
    --target-tags otp_not_received,otp_delivery_time,otp_send_button \
    --run-id bn_otp \
    --judge-mode sync --openai-model gpt-5.5

# 2. Compose all family top-40s into a single production candidate CSV.
#    Compose from top40 (NOT from per-family stage9/production_filtered.csv) —
#    we want one global E5 audit, not stacked filters.
tagclean compose \
    --config configs/bn_full.yaml \
    --from-runs bn_account_locked,bn_name_correction,bn_otp \
    --compose-source top40 \
    --run-id bn_production \
    --out runs/bn_production/composed_top40.csv

# 3. Cross-family Stage 9 audit: drops rows whose top-1 E5 LOO neighbor is a
#    different tag — i.e. rows production E5 itself would misroute.
tagclean stage9 \
    --config configs/bn_full.yaml \
    --e5-audit-input runs/bn_production/composed_top40.csv \
    --run-id bn_production \
    --judge-mode heuristic --device cpu

# Final E5-ready file: runs/bn_production/stage9/production_filtered.csv
```

For just one family (e.g. you're iterating on a single close-tag cluster) you
can stop after step 1 and run `tagclean stage9 --run-id <family>` directly.
The per-family Stage 9 reads `<family>/stage6/question_tag.cleaned.csv` by
default.

## Inputs

- **`question_tag.csv`** — required. Columns: `question`, `tag`. UTF-8.
- **`tag_answer.json`** — optional. `{ "<tag>": "<canonical answer>" }`. Used only as a merge-safety gate in Stage 3 so we don't merge two tags whose answers are substantively different.
- **`config.yaml`** — see `configs/example.yaml` for the full template.

## Outputs

For a run with `run_id=foo` under `<artifact_root>/foo/`:

| File | What |
|---|---|
| `run_manifest.json` | Top-level run identity (version, file hashes, model names, config) |
| `stage0/intake.parquet` | Normalized rows + dedup status (binary; not in git) |
| `stage1/emb_*.npy` + `faiss_*.idx` | E5 + Gemma embeddings + FAISS indices (binary; not in git) |
| `stage2/tag_profile.parquet`, `tag_centroids_*.npy`, `tag_index.json` | Per-tag profile + centroids |
| `stage3/tag_merge_map.csv` | `old_tag → canonical_tag` if any tags were merged |
| `stage3/tag_boundary_policy.jsonl` | GPT-authored discriminative rules per close-tag cluster |
| `stage3/merge_candidates.jsonl` | Pairs that almost merged but didn't |
| `stage4/row_features.parquet` | Per-row composite scores + audit_buffer flag (binary; not in git) |
| `stage5/audit_results.jsonl` | Per-row keep/flag with reason_code + rationale |
| `stage5/audit_packets/*.json` | Raw GPT audit packet inputs/outputs |
| `stage6/question_tag.cleaned.csv` | All audit-pass rows, ranked, with composite score + rationale |
| `stage6/question_tag.top40.csv` | Top-N per tag (`production_recommended` subset), 2-col `question,tag` |
| `stage6/jettisoned_rows.csv` | Dropped rows with `status_reason` and GPT rationale |
| `stage8/cleaning_report.json` | Leave-one-out retrieval accuracy + confusion matrix |
| `stage9/e5_neighbor_audit.csv` | Per-row E5 top-K neighborhood (top-1 tag, own-share, neighbor distribution) |
| `stage9/production_filtered.csv` | **The final E5-ready production set.** |
| `stage9/e5_dropped.csv` | Rows production E5 would misroute, with neighbor breakdown |
| `stage9/audit_report.json` | Per-tag E5 top-1 accuracy + median own-share severity |

## CLI reference

```
tagclean <verb> [options]

verbs:
  stage0..stage9    run a specific stage (chains predecessors via cache)
  all               alias for stage8 (full per-family pipeline)
  compose           concatenate per-run top40/cleaned CSVs into one production CSV
                    (requires --from-runs r1,r2,...)

selection (per-family stages):
  --target-tags T1,T2,T3        explicit cluster
  --seed-tag X                  resolve X's close-tag cluster (see limitations)
  (omit both = corpus-wide; expensive, see costs)

audit:
  --judge-mode {sync,agents,heuristic,batch_prepare,batch_submit,batch_collect}
  --openai-model NAME           default from config
  --concurrency N               Stage 5 audit parallelism (default 6)

stage 9 (E5-only audit):
  --e5-audit-input PATH         external CSV (question,tag); default = stage6/cleaned.csv
  --e5-audit-k INT              top-K neighborhood size (default 10)
  --no-e5-drop                  report-only mode; do not drop top-1 mismatches

compose:
  --from-runs r1,r2,r3          run_ids to combine
  --compose-source top40|cleaned (default: top40)
  --out PATH                    output CSV path

other:
  --config FILE                 YAML config
  --input PATH                  override config.input_csv
  --tag-answer PATH             override config.tag_answer_json
  --artifact-root PATH          override config.artifact_root (default: runs/)
  --run-id NAME                 default: timestamped
  --device {auto,cpu,cuda,mps}
  --resume / --no-resume        default: resume
```

## Configuration

`configs/example.yaml` is the starter template. Copy and edit:

```yaml
input_csv: ~/ec-faq-bot/full_dataset/question_tag.csv
tag_answer_json: ~/ec-faq-bot/full_dataset/tag_answer.json
artifact_root: runs

language: bn

e5_instruction: |
  You are a careful matcher of FAQ questions to canonical intents. Identify
  the most semantically relevant question, considering context, intent, and
  specific details. Use semantic similarity and contextual understanding;
  prioritize exact phrase matches and context-aware matching.

embedding_backend: sentence-transformers
e5_model: intfloat/multilingual-e5-large-instruct
gemma_model: google/embeddinggemma-300m

openai_model: gpt-5.5
openai_reasoning_effort: high
judge_mode: sync                       # sync more reliable than agents at scale
concurrency: 6                          # bounded; gpt-5.5 high-reasoning saturates well below 32

audit_buffer_size: 80
audit_rows_per_packet: 24
boundary_policy_threshold: 0.85
top_n: 40

e5_audit_top_k: 10
e5_audit_drop_on_top1_mismatch: true
```

## Validation results (Bengali EC/NID, 3 close-tag families)

| family | input rows | cleaned | top-40 | Stage 8 LOO | Stage 9 dropped | weakest tag |
|---|---|---|---|---|---|---|
| account_locked × 3 | 539 | 194 | 120 | **0.974** | 6 (3.1%) | unlock_request (top-1=0.91) |
| name_correction × 3 | 717 | 165 | 117 | **0.958** | 8 (4.8%) | parents (top-1=0.86) |
| otp × 3 | 301 | 212 | 120 | **0.986** | 3 (1.4%) | otp_not_received (top-1=0.97) |

**Cross-family production:** 357 composed → **343 kept**, top-1 LOO **0.961**, **0/14 cross-family drops**. The three families are semantically separated in E5 space (no inter-family interference); all 14 drops are within-family boundary paraphrases.

Per-tag retention in the final production CSV (`runs/bn_production/stage9/production_filtered.csv`):

| tag | rows | gap to 40 | top-1 acc |
|---|---|---|---|
| account_locked | 40 | ✓ | 1.000 |
| account_locked_retrials | 39 | -1 | 0.975 |
| account_locked_unlock_request | 37 | -3 | 0.925 |
| name_correction_in_nid_card | 37 | -3 | 0.925 |
| otp_delivery_time | 39 | -1 | 0.975 |
| otp_not_received | 38 | -2 | 0.950 |
| otp_send_button | 40 | ✓ | 1.000 |
| parents_name_correction_new | 33 | -7 | 0.892 |
| spouse_name_correction_new | 40 | ✓ | 1.000 |

We chose to ship 343 rows as-is rather than backfill to 40-per-tag with rank-41+ rows. The Stage 9 gate is the production-aligned guarantee — backfilling re-introduces rows production E5 itself routes to the wrong tag, weakening that guarantee. If a tag has 33 clean rows, ship 33.

## Scaling commands (for cleaning all 1394 tags)

```bash
# 1. Bootstrap stage0/1/2 once on the full corpus (any small family run does it).
tagclean stage8 --config configs/bn_full.yaml \
    --target-tags account_locked,account_locked_retrials,account_locked_unlock_request \
    --run-id corpus_bootstrap --judge-mode sync --openai-model gpt-5.5

# 2. Auto-discover close-tag families. No GPT, no spend.
tagclean discover --config configs/bn_full.yaml \
    --centroids-from corpus_bootstrap \
    --threshold 0.88 --pair-threshold 0.86 --top-k 8 --max-family-size 4 \
    --exclude-tag-pattern '_followup_[a-d]$' \
    --out runs/families.yaml --report runs/families_report.md

# 3. (optional) hand-edit families.yaml; flip questionable families to status: rejected.

# 4. Run all approved families. stage0/1/2 are symlinked from corpus_bootstrap.
tagclean run-families --config configs/bn_full.yaml \
    --manifest runs/families.yaml \
    --judge-mode sync --openai-model gpt-5.5
# add --include-singletons to also clean lone tags

# 5. Compose all family outputs into one production candidate CSV.
tagclean compose --config configs/bn_full.yaml \
    --manifest runs/families.yaml \
    --compose-source cleaned \
    --out runs/production/composed.csv

# 6. Cross-family Stage 9 audit (E5-only, no GPT).
tagclean stage9 --config configs/bn_full.yaml \
    --e5-audit-input runs/production/composed.csv \
    --run-id production --judge-mode heuristic --device cpu
```

Final clean data: `runs/production/stage9/production_filtered.csv`.

## Adjusting which tags to clean

Three places, depending on whether you've discovered families yet or are running a single family by hand.

**1. Per-family in `families.yaml`** (the manifest produced by `tagclean discover`).

Open `runs/families.yaml`. Each family is one block:

```yaml
- family_id: fam_feb20f8e            # stable hash of sorted tag set
  status: approved                    # approved | singleton | rejected
  target_tags:
    - account_locked
    - account_locked_retrials
    - account_locked_unlock_request
  score: { min_edge_sim: 0.906, avg_edge_sim: 0.914 }
  row_counts: { account_locked: 169, ... }
  excluded_neighbors: [...]           # diagnostics only
  notes: ""
```

To customize:

- **Skip a family** entirely → flip `status: approved` to `status: rejected`. `run-families` ignores it; `compose --manifest` excludes it from the production union.
- **Add a tag** to a family → append to `target_tags` and change `status` if needed. Note: the `family_id` is no longer the canonical hash of the new tag set, so add a comment in `notes` if you care; `run-families` keys runs by `family_id` regardless.
- **Remove a tag** from a family → drop it from `target_tags`. Same caveat about the now-stale `family_id`.
- **Make a singleton clean as part of a multi-tag family** → find the singleton's record, change `status: singleton` to `approved`, and add the sibling tag(s) to `target_tags`. `run-families` will then run it as a multi-tag family.
- **Re-discover with different thresholds** → `tagclean discover ... --threshold 0.86 --pair-threshold 0.84` (looser, more families) or `0.92` (tighter, fewer / cleaner). Re-runs are deterministic; same tag sets produce the same `family_id` so completed runs survive.

After editing, just rerun:

```bash
tagclean run-families --config configs/bn_full.yaml --manifest runs/families.yaml --judge-mode sync --openai-model gpt-5.5
```

`--skip-completed` (default) means previously-clean families are not re-cleaned.

**2. One-off via `--target-tags`** when you know exactly which tags to clean and don't want a manifest:

```bash
tagclean stage8 --config configs/bn_full.yaml \
    --target-tags tag1,tag2,tag3 \
    --run-id my_run \
    --judge-mode sync --openai-model gpt-5.5
```

This runs the full per-family pipeline (Stage 0–8) for that exact target set. Stage 3 treats `--target-tags` (≥2 tags) as the cluster directly — no `find_close_tag_clusters` mega-component issue.

**3. Pre-discover allowlist** if you want `discover` to only consider a subset of the corpus:

```bash
echo "tag1
tag2
tag3" > production_tags.txt

tagclean discover --config configs/bn_full.yaml \
    --centroids-from corpus_bootstrap \
    --production-tags production_tags.txt \
    --out runs/families.yaml
```

Unknown tags in the allowlist hard-fail (no silent typos). Use this when you have a curated production-tag list and want discovery confined to it.

## Validation status

The 9-tag, 3-family validation reported above (account_locked / name_correction / otp) was run end-to-end against gpt-5.5 high-reasoning. Final clean sets in `runs/bn_production/stage9/production_filtered.csv` (343 rows, top-40 path) and `runs/bn_production_max/stage9/production_filtered.csv` (554 rows, max-clean path).

The new scaling commands (`discover`, `run-families`, `compose --manifest`) are unit-tested (19 tests pass: stable-family-id order invariance, symlink geometry-mismatch refusal, compose dedup + schema, Stage 9 audit/drop semantics, etc.). End-to-end validation in `--judge-mode heuristic` confirmed the dispatcher iterates families correctly, stage0/1/2 symlinks resolve and SKIP cache, and Stage 3's singleton path emits an empty boundary policy as designed.

End-to-end validation against real GPT (gpt-5.5 sync) for the discover→run-families pipeline at corpus scale was NOT run — the OpenAI quota was exhausted partway through. To resume:

```bash
# (assumes API credit is available again)
tagclean run-families --config configs/bn_full.yaml \
    --manifest runs/families.yaml \
    --judge-mode sync --openai-model gpt-5.5
# --skip-completed (default) means a partial run can resume safely.
```

The 6 issues codex flagged on the first pass of the scaling commands are fixed in commit `ba4f8d7` (compose-includes-singletons, force-rerun semantics, singleton Stage 3 no-op policy, --production-tags allowlist hard-fail on unknown entries, excluded_neighbors reason classification, symlink geometry-mismatch refusal).

## Known limitations (read before scaling)

1. **Stage 4 ranker scales linearly with full-corpus rows × target families.** It loops every row in target scope but uses the full corpus as KNN-overlap evidence, so each per-family `stage8` invocation walks all 79k rows. We saw a ~20-min Stage 4 in a 2-tag heuristic run. At 264 family runs, this is the new wall-clock bottleneck (the dominant scaling cost was previously Stage 1 embedding, which `run-families` already eliminates via stage0/1/2 symlinks). Future fix: compute the expensive row features only for target-scope rows while still using global centroids/neighbors as evidence — should drop per-family Stage 4 from minutes to seconds.

2. **Stage 9 is chunked but still has a ceiling.** The dense `vecs @ vecs.T` is replaced with row-chunked top-K (default chunk = 4096 rows; peak memory ≈ chunk × N × 4 bytes). At N=40k that's ~640 MB per chunk — fine for 16+ GB machines, tight on smaller. If you hit OOM, drop the `CHUNK` constant in `run_stage9` (currently a magic number; could be made configurable later).

3. **`compose --manifest` skips families whose stage6 output is missing.** Convenient for partial runs but dangerous for the production handoff if a family failed silently. Pass `--require-complete` to compose to make it hard-fail when any approved/singleton family lacks stage6 outputs.

## Other limitations

1. **`--seed-tag` cluster expansion is broken at corpus scale.** With 1395 tags, `find_close_tag_clusters` at the default 0.85 threshold forms a 488-tag mega-component (Bengali NID vocabulary is heavily shared). `max_cluster_size=6` then truncates by alphabetical tag-index order, dropping the seed entirely. Workaround: use `--target-tags` with the explicit list. Stage 3 honors the user's choice directly when `--target-tags` has ≥2 tags (skips union-find). A proper seed-centric expansion (seed + N nearest direct neighbors, no transitive closure) is a planned follow-up.

2. **`--judge-mode agents` can hang indefinitely on Stage 3.** Observed once on a corpus-wide Stage 3 boundary policy call — the openai-agents SDK got stuck in an SSL select for 1+ hour with no timeout. `--judge-mode sync` uses the responses API directly with explicit SDK timeouts and is currently more reliable.

3. **FAISS + sentence-transformers SIGSEGV on Python 3.14 / Mac.** When both libraries are loaded in the same process and the model encode runs first, a subsequent `import faiss` segfaults. Stage 9 uses pure-numpy top-K instead. Affects Stage 1+ if you re-encode in the same process.

4. **OMP duplicate library warning.** Running two `tagclean` processes simultaneously on Mac can trip an OpenMP duplicate-library check. Set `KMP_DUPLICATE_LIB_OK=TRUE` if you need concurrent runs.

## Architecture choices

- **Plain Python, no agent framework.** The pipeline is a deterministic DAG of stages with parquet/jsonl artifacts. Each stage is idempotent on its inputs.
- **GPT only at narrow points.** Stage 3 (cluster boundary policy) and Stage 5 (buffer audit). Per-row LLM judging was tried and discarded.
- **Default Bengali normalization.** `language: bn` runs `bnunicodenormalizer`. Pass `--language none` for English/multilingual.
- **Stage 9 is the production-truth gate.** Stage 4 ranking uses E5+Gemma; production inference uses E5 alone. Anything that scored well jointly but E5 alone misroutes is a production failure caught here.
- **Compose from `top40.csv`, not per-family `production_filtered.csv`.** One global Stage 9 audit on the union, not stacked filters.
- **No backfill in Stage 9.** Hitting a fixed row count by promoting rank-41+ rows defeats the E5-purity guarantee. If you need exactly N per tag, that's a separate, explicit operation.

## Tests

```bash
pytest -q
```

17 unit tests covering helpers (text normalization, MMR ranking, cluster discovery, packet building, audit decisions, Stage 9 audit/drop semantics, compose). End-to-end validation is by running the pipeline against your dataset — `stage8/cleaning_report.json` and `stage9/audit_report.json` are the empirical signals.

## Repo layout

```
src/tagclean/cleaner.py     all stages + CLI in one module
configs/example.yaml        starter config
configs/bn_full.yaml        Bengali NID full-corpus config used for the validation runs
docs/run_instructions.md    operational runbook (recipes, troubleshooting, costs)
docs/harness_design.md      design rationale (why each stage is shaped this way)
tests/test_cleaner.py       unit tests (offline, no API)
tools/githooks/pre-commit   refuses staged artifact paths under runs/ or artifacts/
runs/                       gitignored; per-run-id outputs land here
```

## License

MIT.
