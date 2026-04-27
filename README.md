# tagclean

Deterministic dataset cleaner for tagged FAQ corpora. Built for the Bangladesh
Election Commission Bengali NID/voter FAQ domain (1394 tags, 79k GPT-generated
rows), but the harness is language-neutral when you set `language: none`.

You give it `question_tag.csv` (and optionally `tag_answer.json`); it gives back
a cleaned production subset where ambiguous, duplicated, and synthetic-looking
rows are jettisoned and the survivors are filtered against the model
production actually uses at inference (E5).

**This branch (`deterministic-only`):** zero LLM calls. The pipeline runs
purely on E5 geometry plus deterministic heuristics — Stage 6 top-N per tag
followed by a global Stage 9 production-truth audit. The earlier Claude-assisted
boundary-policy and Stage QA passes are gone (see `docs/harness_design.md` for
the rationale). Trade-off, calibrated: a small accuracy delta against the
Claude-assisted output, in exchange for full reproducibility and no rate
limits / no API spend.

## The cleaned corpus

The cleaned production-ready corpus produced by the prior Claude-assisted
branch lives at the repo root for reference:

```
cleaned_bn_nid_corpus.csv
```

Two columns (`question`, `tag`). Regenerate from the deterministic-only
pipeline by following [End-to-end production recipe](#end-to-end-production-recipe)
on this branch.

## The problem

GPT-generated FAQ data has two failure modes:

1. **Within a tag**, rows look fine — they're paraphrases of the same question.
2. **Across close-sibling tags**, rows leak — the model picks the wrong tag for
   a question that legitimately fits multiple intents.

Bengali NID/voter is the hard case. Many tags are slight rewordings of each
other:

- `account_locked` vs `account_locked_retrials` vs `account_locked_unlock_request`
- `name_correction_in_nid_card` vs `parents_name_correction_new` vs `spouse_name_correction_new`
- `otp_not_received` vs `otp_delivery_time` vs `otp_send_button`

Production inference uses E5 alone. Anything ranked well but E5 alone
misroutes is a *production* failure, not a cleaning failure. So the cleaned
set has to pass an E5-only final filter (Stage 9).

## The solution

```
question_tag.csv  →  Stage 0  intake / Bengali normalize / dedup
                  →  Stage 1  E5 embeddings + FAISS
                  →  Stage 2  per-tag medoid + central rows + discriminative phrases
                  →  Stage 3  deterministic merge-candidate scan (E5 cosine + kNN-overlap)
                  →  Stage 4  composite ranker, E5-only
                  →  Stage 6  top-N=40 per tag by composite_score + MMR (λ=0.7)
                  →  Stage 8  leave-one-out validation (FAISS)
                  →  Stage 9  E5-only production-risk audit + filter

For multi-family production composition:

  family_A/stage6/top40.csv ┐
  family_B/stage6/top40.csv ┼─→ tagclean compose → composed.csv → stage9 → production_filtered.csv
  family_C/stage6/top40.csv ┘
```

Stages 5 and 7 are deleted on this branch. There is no LLM call anywhere in
the pipeline.

## Roles

| Component | Role | Weight in ranking |
|---|---|---|
| **E5-multilingual-large-instruct** (1024d) | The single source of geometry. Same model production uses at inference. | 0.55 (own cosine) + 0.35 (margin) = 0.90 |
| Penalties | near_duplicate, cross_tag_duplicate, artifact_score | -0.05, -0.10, -0.10 |
| MMR diversity | λ=0.7 within-tag re-rank for the top-N | 15% post-rerank |

E5 cluster discovery uses cosine threshold + row-level kNN-overlap as a
multi-criteria gate inside Stage 3 (`find_close_tag_clusters`) and reciprocal
top-K + pair-threshold inside `discover`.

## Quickstart

```bash
git clone <this repo> tagclean
cd tagclean
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q                              # offline tests
```

Real run:

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
    --device cpu

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
| `run_manifest.json` | Top-level run identity (version, file hashes, model name, config) |
| `stage0/intake.parquet` | Normalized rows + dedup status (binary; not in git) |
| `stage1/emb_e5.npy` + `faiss_e5.idx` | E5 embeddings + FAISS index (binary; not in git) |
| `stage2/tag_profile.parquet`, `tag_centroids_e5.npy`, `tag_index.json` | Per-tag profile + centroids |
| `stage3/tag_merge_map.csv` | `old_tag → canonical_tag` if any tags were merged |
| `stage3/merge_candidates.jsonl` | Pairs that almost merged but didn't |
| `stage4/row_features.parquet` | Per-row composite scores (binary; not in git) |
| `stage6/question_tag.cleaned.csv` | All in-scope rows, ranked, with composite + MMR scores |
| `stage6/question_tag.top40.csv` | Top-N per tag (`production_recommended` subset), 2-col `question,tag` |
| `stage6/jettisoned_rows.csv` | Below-top-N rows for inspection |
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
  (omit both = corpus-wide; expensive)

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

top_n: 40
near_duplicate_threshold: 0.98
e5_merge_threshold: 0.90
knn_overlap_threshold: 0.30
boundary_policy_threshold: 0.85
boundary_policy_max_cluster_size: 6
concurrency: 6

e5_audit_top_k: 10
e5_audit_drop_on_top1_mismatch: true
```

## Correctness behaviors worth knowing

These are deliberate choices flagged by review:

- **Stage 9 is the production-truth gate.** Stage 4 ranking is composite (E5
  own + margin + penalties); production inference is E5 alone. Anything that
  scored well composite but E5 alone misroutes is a production failure, caught
  here.
- **Compose from `top40.csv`, not per-family `production_filtered.csv`.** A row
  that fails LOO inside a 3-tag family can be safely separable in the larger
  production union (its in-family competitor is no longer a peer). Filter once,
  globally, after composition.
- **No backfill in Stage 9.** Hitting a fixed row count by promoting rank-41+
  rows would defeat the E5-purity guarantee.

## Validation status

The 9-tag, 3-family validation in earlier README revisions (top-1 LOO 0.961
across account_locked / name_correction / otp families, 0/14 cross-family
drops) was done with a prior LLM-assisted geometry. The deterministic-only
branch's accuracy delta against that baseline is expected to be small (sub-1%
to low-single-digit top-1 retrieval) but should be re-measured against your
dataset before shipping.

To re-validate end-to-end, run the production recipe above against a 3-family
test set, compare `stage8/cleaning_report.json` and
`stage9/audit_report.json` against any earlier baselines you have.

## Scaling commands (for cleaning all tags)

```bash
# 1. Bootstrap stage0/1/2 once on the full corpus.
tagclean stage8 --config configs/bn_full.yaml \
    --target-tags account_locked,account_locked_retrials,account_locked_unlock_request \
    --run-id corpus_bootstrap

# 2. Auto-discover close-tag families.
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
    --compose-source top40 \
    --out runs/production/composed.csv

# 6. Cross-family Stage 9 audit.
tagclean stage9 --config configs/bn_full.yaml \
    --e5-audit-input runs/production/composed.csv \
    --run-id production --device cpu
```

Final clean data: `runs/production/stage9/production_filtered.csv`.

## Adjusting which tags to clean

See `docs/run_instructions.md` for the full breakdown. Quick pointer:

- **`families.yaml`** is the canonical place once you've run `tagclean discover`. Flip `status: approved` ↔ `rejected`, or edit `target_tags` per family. `run-families --skip-completed` (default) won't re-clean families whose stage8 outputs already exist.
- **One-off `--target-tags T1,T2,T3`** when you don't want a manifest.
- **`--production-tags FILE`** to restrict `discover` to a curated subset of the corpus. Unknown tags hard-fail.

## Known limitations

1. **`--seed-tag` cluster expansion is broken at corpus scale** (1394-tag
   mega-component issue). Use `--target-tags` with the explicit list, or use
   `discover` + `families.yaml`.
2. **Stage 4 ranker scales with full-corpus rows × target families.** Each
   per-family `stage8` walks all corpus rows for kNN evidence; at hundreds of
   family runs this is the wall-clock bottleneck after stage0/1/2 caching.
3. **Stage 9 is chunked but has a memory ceiling.** Default chunk = 4096 rows;
   peak memory ≈ chunk × N × 4 bytes (~640 MB at N=40k). Drop the `CHUNK`
   constant in `run_stage9` if you OOM on smaller machines.
4. **`compose --manifest` warns on missing families by default.** Pass
   `--require-complete` to hard-fail before shipping a partial production CSV.

## Architecture choices

- **Plain Python, no agent framework.** The pipeline is a deterministic DAG of stages with parquet/jsonl artifacts. Each stage is idempotent on its inputs.
- **No LLM calls.** All stages are deterministic. The previous Claude-assisted
  boundary policy authoring and Stage QA reviewer were removed in favor of
  this branch's pure-geometry approach.
- **Single embedding model.** E5-multilingual-large-instruct. The cross-validation gate uses E5 cosine + row-level kNN-overlap multi-criteria — no second embedding model.
- **Default Bengali normalization.** `language: bn` runs `bnunicodenormalizer`. Pass `--language none` for English/multilingual.

## Tests

```bash
pytest -q
```

Offline tests covering helpers (text normalization, cluster discovery, Stage 0
dedup, Stage 9 audit/drop semantics, compose) plus a deterministic-only
end-to-end smoke (Stage 8 → Stage 9 chain).

## Repo layout

```
src/tagclean/cleaner.py     all stages + CLI
configs/example.yaml        starter config
configs/bn_full.yaml        Bengali NID full-corpus config
docs/run_instructions.md    operational runbook
docs/harness_design.md      design rationale
tests/test_cleaner.py       unit tests (offline)
tools/githooks/pre-commit   refuses staged artifact paths under runs/ or artifacts/
runs/                       gitignored; per-run-id outputs land here
```

## License

MIT.
