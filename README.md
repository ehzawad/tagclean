# tagclean

LLM-assisted dataset cleaner for tagged FAQ corpora. Built for the Bangladesh
Election Commission Bengali NID/voter FAQ domain (1395 tags, 79k GPT-generated
rows), but the harness is language-neutral when you set `language: none`.

You give it `question_tag.csv` (and optionally `tag_answer.json`); it gives back
a cleaned production subset where ambiguous, duplicated, and synthetic-looking
rows are jettisoned and the survivors are filtered against the model
production actually uses at inference (E5).

**This branch (`claude-cli-harness`):** the cleaner shells out to the `claude`
CLI binary instead of calling the OpenAI Responses API. No API key, no per-call
billing — calls route through your Claude Code subscription. The second
embedding model (EmbeddingGemma) was also dropped; cluster discovery now uses
E5 cosine + row-level kNN-overlap as multi-criteria gates.

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
**the LLM is never the per-row scorer.** Embeddings rank, the LLM writes
policy and audits the top-N buffer.

A second realization came late: production inference uses E5 alone. Anything
the dual-embedding ranker scored well but E5 alone misroutes is a *production*
failure, not a cleaning failure. So the cleaned set has to pass an E5-only
final filter (Stage 9).

## The solution

```
question_tag.csv  →  Stage 0  intake / Bengali normalize / dedup
                  →  Stage 1  E5 embeddings + FAISS
                  →  Stage 2  per-tag medoid + central rows + discriminative phrases
                  →  Stage 3  Claude writes per-tag boundary rules for close-tag clusters
                  →  Stage 4  deterministic ranker → top-N audit buffer per tag
                  →  Stage 5  Claude audits the buffer (keep / flag with reasons)
                  →  Stage 6  walk audited-pass rows, MMR-diversify, take top K
                  →  Stage 8  leave-one-out validation
                  →  Stage 9  E5-only production-risk audit + filter

For multi-family production composition:

  family_A/stage6/top40.csv ┐
  family_B/stage6/top40.csv ┼─→ tagclean compose → composed.csv → stage9 → production_filtered.csv
  family_C/stage6/top40.csv ┘
```

(Stage 7 was a per-row second-pass review; deleted in this branch — Stage 5's
buffer audit subsumes it.)

Claude cost for a 3-tag cluster: ~12 calls (~1 boundary policy + ~10–12 audit
packets) against the user's Claude Code subscription. Embedding cost is GPU/CPU
time, not API.

## Roles

| Component | Role | Weight in ranking |
|---|---|---|
| **E5-multilingual-large-instruct** (1024d) | The single source of geometry. Same model production uses at inference. | 0.45 (cos) + 0.30 (margin) = 0.75 |
| Token alignment (must_have/avoid from policy) | Cross-validates LLM-authored boundary rules against row text | 0.15 |
| Penalties | near_duplicate, cross_tag_duplicate, artifact_score | -0.05, -0.10, -0.10 |
| **Claude (Opus)** for Stage 3 | Boundary-rule author, 1 call per close-tag cluster, `--effort high` | 0% direct |
| **Claude (Sonnet)** for Stage 5 | Buffer auditor, ~10 calls per family, `--effort medium` | 0% direct |

The dual-embedding (E5+Gemma) cross-validation that earlier validations
relied on has been replaced by E5-only multi-criteria gates: cosine threshold
+ row-level kNN-overlap inside Stage 3 (`find_close_tag_clusters`), and
reciprocal top-K + pair-threshold inside `discover`.

## Quickstart

```bash
git clone <this repo> tagclean
cd tagclean
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q                              # offline tests, no API/CLI
```

You also need the Claude Code CLI installed and logged in:

```bash
# Install Claude Code (see https://claude.com/claude-code).
claude /login                          # one-time browser-flow login
```

`tagclean` runs a cheap `claude -p` probe at startup to confirm the
subscription session is healthy before launching a multi-hour pipeline.

Real run with Claude-authored boundary policy + audit:

```bash
cp configs/example.yaml my-config.yaml
# edit my-config.yaml: input_csv, tag_answer_json, run_id, e5_instruction
tagclean stage8 --config my-config.yaml --target-tags T1,T2,T3
```

## End-to-end production recipe

This is the full path from raw `question_tag.csv` to the final production CSV
that gets indexed by E5 at inference time.

```bash
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 1. Clean each close-tag family separately. One stage8 invocation per family.
#    --target-tags is the recommended scoping at corpus scale.
tagclean stage8 \
    --config configs/bn_full.yaml \
    --target-tags account_locked,account_locked_retrials,account_locked_unlock_request \
    --run-id bn_account_locked

tagclean stage8 \
    --config configs/bn_full.yaml \
    --target-tags name_correction_in_nid_card,parents_name_correction_new,spouse_name_correction_new \
    --run-id bn_name_correction

tagclean stage8 \
    --config configs/bn_full.yaml \
    --target-tags otp_not_received,otp_delivery_time,otp_send_button \
    --run-id bn_otp

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
| `stage1/emb_e5.npy` + `faiss_e5.idx` | E5 embeddings + FAISS index (binary; not in git) |
| `stage2/tag_profile.parquet`, `tag_centroids_e5.npy`, `tag_index.json` | Per-tag profile + centroids |
| `stage3/tag_merge_map.csv` | `old_tag → canonical_tag` if any tags were merged |
| `stage3/tag_boundary_policy.jsonl` | Claude-authored discriminative rules per close-tag cluster |
| `stage3/merge_candidates.jsonl` | Pairs that almost merged but didn't |
| `stage3/llm_calls.jsonl` | Per-call telemetry: subtype, cost, cache tokens, num_turns |
| `stage4/row_features.parquet` | Per-row composite scores + audit_buffer flag (binary; not in git) |
| `stage5/audit_results.jsonl` | Per-row keep/flag with reason_code + rationale |
| `stage5/audit_packets/*.json` | Raw Claude audit packet inputs/outputs |
| `stage5/llm_calls.jsonl` | Per-packet telemetry (same fields as Stage 3) |
| `stage6/question_tag.cleaned.csv` | All audit-pass rows, ranked, with composite score + rationale |
| `stage6/question_tag.top40.csv` | Top-N per tag (`production_recommended` subset), 2-col `question,tag` |
| `stage6/jettisoned_rows.csv` | Dropped rows with `status_reason` and Claude rationale |
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
                    (requires --from-runs r1,r2,... or --manifest families.yaml)
  discover          auto-discover close-tag families from cached centroids
  run-families      run stage8 per approved family from a manifest

selection (per-family stages):
  --target-tags T1,T2,T3        explicit cluster
  --seed-tag X                  resolve X's close-tag cluster (see limitations)
  (omit both = corpus-wide; expensive, see costs)

claude:
  --judge-mode {claude,heuristic}    default: claude
  --stage3-model NAME                default: opus     (1 call per cluster)
  --stage3-effort {low,medium,high,max}  default: high
  --stage5-model NAME                default: sonnet   (~10 calls per family)
  --stage5-effort {low,medium,high,max}  default: medium
  --claude-fallback-model NAME       default: sonnet
  --claude-call-timeout S            default: 300
  --concurrency N                    Stage 5 audit parallelism (default 6)

stage 9 (E5-only audit):
  --e5-audit-input PATH         external CSV (question,tag); default = stage6/cleaned.csv
  --e5-audit-k INT              top-K neighborhood size (default 10)
  --no-e5-drop                  report-only mode; do not drop top-1 mismatches

compose:
  --from-runs r1,r2,r3          run_ids to combine
  --manifest PATH               or pull all approved families from a manifest
  --compose-source top40|cleaned (default: top40)
  --out PATH                    output CSV path
  --require-complete            hard-fail if any approved manifest family is missing stage6

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

judge_mode: claude
stage3_model: opus              # 1 call/cluster, careful reasoning
stage3_effort: high
stage5_model: sonnet            # ~10 calls/family, simpler keep/flag
stage5_effort: medium
claude_fallback_model: sonnet
claude_call_timeout_s: 300
audit_circuit_breaker_window: 10

concurrency: 6
audit_buffer_size: 80
audit_rows_per_packet: 24
boundary_policy_threshold: 0.85
top_n: 40

e5_audit_top_k: 10
e5_audit_drop_on_top1_mismatch: true
```

## How the LLM calls work

Every Stage 3 / Stage 5 call invokes:

```
claude -p \
    --model <opus|sonnet> --fallback-model sonnet --effort <effort> \
    --output-format json --no-session-persistence \
    --json-schema '<pydantic-derived schema>'
# prompt streamed via stdin
```

The subprocess env strips `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and
`ANTHROPIC_BASE_URL` so the call always routes through your Claude Code
subscription (no API key needed; no per-call billing). Output is parsed from
the envelope's `structured_output` field and validated against the same
pydantic schema.

Failures are classified into three buckets:

- **INFRA** (missing binary, auth failure, perm denied) — don't burn retries.
- **TRANSIENT** (timeout, claude `subtype != success`) — count against retries.
- **PARSE** (envelope JSON unreadable) — count against retries.

Per-call telemetry is persisted to `stageN/llm_calls.jsonl` so cost / cache-hit
drift across runs is visible without grepping logs:

```json
{"packet_id": "audit_packet:000003", "stage": "stage5",
 "model": "sonnet", "effort": "medium", "subtype": "success", "ok": true,
 "total_cost_usd": 0.0042, "cache_creation_input_tokens": 18200,
 "cache_read_input_tokens": 17900, "num_turns": 1, "duration_ms": 6800}
```

## Correctness behaviors worth knowing

These are deliberate choices flagged by review:

- **Resume hashes include LLM identity.** Stage 3 and Stage 5 input hashes
  bake in `judge_mode`, model, effort, prompt-version constant, and schema
  fingerprint. A model/effort/prompt change forces a re-run rather than
  silently reusing stale boundary policies or audits.
- **Stage 5 fails closed.** A buffer row whose audit packet returned no
  decision is flagged with `reason_code=audit_missing` (not silently kept).
  At corpus scale silent fail-keeps would otherwise pile up to thousands.
- **Stage 5 has a circuit breaker.** If the most recent N audit packets all
  errored (default N=10, configurable via `audit_circuit_breaker_window`), the
  stage halts with a diagnostic and refuses to write `audit_results.jsonl`.
  Inspect `llm_calls.jsonl`, fix root cause, then rerun.
- **Stage 9 is the production-truth gate.** Stage 4 ranking is E5+token-alignment;
  production inference is E5 alone. Anything that scored well jointly but E5
  alone misroutes is a production failure, caught here.

## Validation status

The 9-tag, 3-family validation in the prior README (top-1 LOO 0.961 across
account_locked / name_correction / otp families, 0/14 cross-family drops) was
done with the **earlier OpenAI gpt-5.5 + Gemma** combination. Those numbers
do not transfer mechanically to this branch:

- **What still holds:** the deterministic stages (0, 1, 2, 4, 6, 8, 9) are
  shape-equivalent; the Stage 4 weight rebalancing only redistributed Gemma's
  ~35% across E5 own/margin and token_alignment.
- **What needs re-validation:** Stage 3 boundary policies and Stage 5 audit
  decisions are now Claude-authored, not GPT-authored. Expect different
  rationales and possibly different keep/flag decisions on borderline rows.
- **What replaced what:** Gemma cluster discovery was replaced by E5 cosine +
  row-level kNN-overlap multi-gates. `discover`'s reciprocal-NN ego-family
  expansion already had multi-criteria robustness (reciprocal top-K +
  pair-threshold) so the loss is most felt in `find_close_tag_clusters`.

To re-validate end-to-end, run the production recipe above against a 3-family
test set, compare `stage8/cleaning_report.json` and `stage9/audit_report.json`
against the OpenAI-era artifacts in `runs/bn_*` directories.

## Scaling commands (for cleaning all 1394 tags)

```bash
# 1. Bootstrap stage0/1/2 once on the full corpus.
tagclean stage8 --config configs/bn_full.yaml \
    --target-tags account_locked,account_locked_retrials,account_locked_unlock_request \
    --run-id corpus_bootstrap

# 2. Auto-discover close-tag families. No claude calls, no spend.
tagclean discover --config configs/bn_full.yaml \
    --centroids-from corpus_bootstrap \
    --threshold 0.88 --pair-threshold 0.86 --top-k 8 --max-family-size 4 \
    --exclude-tag-pattern '_followup_[a-d]$' \
    --out runs/families.yaml --report runs/families_report.md

# 3. (optional) hand-edit families.yaml; flip questionable families to status: rejected.

# 4. Run all approved families. stage0/1/2 are symlinked from corpus_bootstrap.
tagclean run-families --config configs/bn_full.yaml \
    --manifest runs/families.yaml
# add --include-singletons to also clean lone tags

# 5. Compose all family outputs into one production candidate CSV.
tagclean compose --config configs/bn_full.yaml \
    --manifest runs/families.yaml \
    --compose-source cleaned \
    --out runs/production/composed.csv

# 6. Cross-family Stage 9 audit (E5-only, no claude).
tagclean stage9 --config configs/bn_full.yaml \
    --e5-audit-input runs/production/composed.csv \
    --run-id production --judge-mode heuristic --device cpu
```

Final clean data: `runs/production/stage9/production_filtered.csv`.

## Adjusting which tags to clean

See `docs/run_instructions.md` for the full breakdown. Quick pointer:

- **`families.yaml`** is the canonical place once you've run `tagclean discover`. Flip `status: approved` ↔ `rejected`, or edit `target_tags` per family. `run-families --skip-completed` (default) won't re-clean families whose stage8 outputs already exist.
- **One-off `--target-tags T1,T2,T3`** when you don't want a manifest.
- **`--production-tags FILE`** to restrict `discover` to a curated subset of the corpus. Unknown tags hard-fail.

## Known limitations

1. **Subscription-auth tension.** The chronicle-style env-stripping forces
   subscription routing, but if you're not logged in via `claude /login` the
   first call fails INFRA. The startup `probe_auth()` catches this before
   launching a long pipeline.
2. **Subscription rate limits are real.** The Claude Code subscription has
   per-minute quotas. The Stage 5 circuit breaker halts the run when bursts
   trip them rather than silently spinning retries; lower `concurrency` (default
   6) if you see sustained transients in `llm_calls.jsonl`.
3. **`--seed-tag` cluster expansion is broken at corpus scale** (1394-tag
   mega-component issue inherited from the prior design). Use `--target-tags`
   with the explicit list, or use `discover` + `families.yaml`.
4. **Stage 4 ranker scales linearly with full-corpus rows × target families.**
   Each per-family `stage8` walks all corpus rows for kNN evidence; at 264
   family runs this is the wall-clock bottleneck after stage0/1/2 caching.
5. **Stage 9 is chunked but has a memory ceiling.** Default chunk = 4096 rows;
   peak memory ≈ chunk × N × 4 bytes (~640 MB at N=40k). Drop the `CHUNK`
   constant in `run_stage9` if you OOM on smaller machines.
6. **`compose --manifest` warns on missing families by default.** Pass
   `--require-complete` to hard-fail before shipping a partial production CSV.

## Architecture choices

- **Plain Python, no agent framework.** The pipeline is a deterministic DAG of stages with parquet/jsonl artifacts. Each stage is idempotent on its inputs (with hashes that include LLM identity).
- **Claude only at narrow points.** Stage 3 (cluster boundary policy) and Stage 5 (buffer audit). Per-row LLM judging was tried and discarded.
- **Single embedding model.** E5-multilingual-large-instruct. The prior dual-model (E5+Gemma) was simplified out; the cross-validation gate is replaced by E5 cosine + row-level kNN-overlap multi-criteria.
- **Default Bengali normalization.** `language: bn` runs `bnunicodenormalizer`. Pass `--language none` for English/multilingual.
- **Stage 9 is the production-truth gate.** Stage 4 ranking is E5+token-alignment; production inference is E5 alone. Anything that scored well jointly but E5 alone misroutes is caught here.
- **Compose from `top40.csv`, not per-family `production_filtered.csv`.** One global Stage 9 audit on the union, not stacked filters.
- **No backfill in Stage 9.** Hitting a fixed row count by promoting rank-41+ rows defeats the E5-purity guarantee.

## Tests

```bash
pytest -q
```

Offline tests covering helpers (text normalization, MMR ranking, cluster discovery, packet building, audit decisions, Stage 9 audit/drop semantics, compose, claude_cli env stripping + subtype parsing). End-to-end validation is by running the pipeline against your dataset — `stage8/cleaning_report.json` and `stage9/audit_report.json` are the empirical signals; `stageN/llm_calls.jsonl` is the cost/quality drift signal.

## Repo layout

```
src/tagclean/cleaner.py     all stages + CLI
src/tagclean/claude_cli.py  headless `claude -p` subprocess wrapper
configs/example.yaml        starter config
configs/bn_full.yaml        Bengali NID full-corpus config
docs/run_instructions.md    operational runbook (recipes, troubleshooting, costs)
docs/harness_design.md      design rationale (why each stage is shaped this way)
tests/test_cleaner.py       unit tests (offline, no API/CLI)
tools/githooks/pre-commit   refuses staged artifact paths under runs/ or artifacts/
runs/                       gitignored; per-run-id outputs land here
```

## License

MIT.
