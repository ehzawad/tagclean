# tagclean — Run Instructions

How to install, configure, and run the cleaner. For **why** the pipeline is shaped this way, see `docs/harness_design.md`.

## Prerequisites

- Python 3.10+ (3.14 tested)
- `OPENAI_API_KEY` env var (when running with `--judge-mode agents` or `sync`)
- ~3 GB free for cached HuggingFace models (E5 1024d + EmbeddingGemma 768d)
- Optional GPU (CUDA on Linux, MPS on Mac); CPU works too, slower

## Install

```bash
git clone <this repo> ~/tagclean
cd ~/tagclean
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q                              # 13 unit tests, no API
```

Install the pre-commit guard so you don't accidentally check in run artifacts:

```bash
git config core.hooksPath tools/githooks
chmod +x tools/precommit_artifact_check.sh
```

## Quick start — clean three specific tags

The most common workflow: you know which close-sibling tags are leaky, and you want to clean them as a unit.

```bash
export OPENAI_API_KEY=sk-...

tagclean stage8 \
    --input /path/to/question_tag.csv \
    --tag-answer /path/to/tag_answer.json \
    --target-tags account_locked,account_locked_retrials,account_locked_unlock_request \
    --judge-mode agents \
    --openai-model gpt-5.5 \
    --run-id account_locked_clean_v1
```

This runs all stages 0→8 with sensible defaults. Output lands in `artifacts/account_locked_clean_v1/`. Time: ~10 min for ~540 rows; ~$0.50 in GPT spend.

## Three ways to pick which tags to clean

### 1. Manual list — `--target-tags`
You know exactly which tags. Comma-separated, no spaces:

```bash
tagclean stage8 --target-tags T1,T2,T3 ...
```

### 2. Seed expansion — `--seed-tag`
You know one suspect tag. The harness finds its close-tag cluster automatically (cosine ≥ `boundary_policy_threshold` in BOTH E5 and Gemma):

```bash
tagclean stage8 --seed-tag account_locked ...
```

The CLI prints the resolved cluster **and** the top-5 nearest excluded tags with their similarity scores. If a sibling you expected is missing, the diagnostic tells you why:

```
[seed] resolved cluster from 'account_locked': ['account_locked_retrials', 'account_locked_unlock_request', 'account_locked']
[seed] threshold=0.85 (min of E5/Gemma must clear). Nearest excluded tags:
        account_locked_unlock_failed              E5=0.92  Gemma=0.81  min=0.81 ← sibling, just below threshold
```

When that happens, lower `boundary_policy_threshold` in your config or pass `--target-tags` explicitly to include the missing tag.

### 3. Whole corpus — omit both flags
```bash
tagclean stage8 --input ... --tag-answer ...
```

This processes every tag and discovers all close-tag clusters automatically. **High GPT cost** — only do it once you've validated on individual families and the budget is approved.

## Configuration

`configs/example.yaml` is the starter template. Copy and edit:

```bash
cp configs/example.yaml my-config.yaml
```

The full config-knob table lives in the README. The fields you'll actually edit for a Bengali NID run:

```yaml
input_csv: ~/ec-faq-bot/full_dataset/question_tag.csv
tag_answer_json: ~/ec-faq-bot/full_dataset/tag_answer.json
artifact_root: runs                  # gitignored; safe to keep under the repo
run_id: account_locked_clean_v1

language: bn

e5_instruction: |
  You are an expert at matching Bangladeshi National Identity Card (NID) and
  voter registration queries. Identify the most semantically relevant question
  by intent and concrete details, prioritizing exact phrase matches.

embedding_backend: sentence-transformers
judge_mode: agents
openai_model: gpt-5.5

# Tune these only if --seed-tag misses an obvious sibling or the audit looks too aggressive.
boundary_policy_threshold: 0.85
audit_buffer_size: 80
top_n: 40
```

Then run with `--config my-config.yaml`; CLI flags still override individual fields. **Recommended `artifact_root` is `runs/`** — both `runs/` and `artifacts/` are gitignored, and the pre-commit hook refuses staged artifact paths in either.

## Outputs

For a run with `run_id=foo`, in `<artifact_root>/foo/`:

```
foo/
├── run_manifest.json                   ← top-level identity (version, hashes, models)
├── stage0/intake.parquet
├── stage1/{emb_e5,emb_gemma}.npy + faiss indices
├── stage2/{tag_profile.parquet, tag_centroids_*.npy, tag_index.json}
├── stage3/
│   ├── tag_merge_map.csv               ← old_tag → canonical_tag
│   ├── tag_boundary_policy.jsonl       ← GPT-authored discriminative rules
│   └── merge_candidates.jsonl
├── stage4/row_features.parquet         ← composite_score, audit_buffer flag
├── stage5/audit_results.jsonl          ← per-row keep/flag with reasons
├── stage6/
│   ├── question_tag.cleaned.csv        ← all audit-pass rows, ranked
│   ├── question_tag.top40.csv          ← top-N per tag, 2-col `question,tag`
│   └── jettisoned_rows.csv             ← dropped rows + reasons + GPT rationale
└── stage8/cleaning_report.json         ← LOO retrieval accuracy + confusion
```

The two files you usually want:

- `stage6/question_tag.top40.csv` — the production subset (40 per tag, 2-column).
- `stage6/question_tag.cleaned.csv` — all audit-pass rows ranked, useful for inspection.

## Running stage by stage

Each stage chains its predecessors and resumes from cache:

```bash
tagclean stage0 --config my-config.yaml      # intake only
tagclean stage1 --config my-config.yaml      # + embeddings
tagclean stage6 --config my-config.yaml      # → stages 0–6
tagclean stage8 --config my-config.yaml      # → all
tagclean all --config my-config.yaml         # alias for stage8 in non-batch modes
```

Re-running with `--resume` (default) skips any stage whose input hash matches its manifest. `--no-resume` forces a clean rerun of the requested stage and downstream.

## Recipes

### "Clean these 3 specific Bengali tags right now"
```bash
tagclean stage8 \
    --input ~/ec-faq-bot/full_dataset/question_tag.csv \
    --tag-answer ~/ec-faq-bot/full_dataset/tag_answer.json \
    --target-tags name_correction_in_nid_card,parents_name_correction_new,spouse_name_correction_new \
    --max-tags 0 \
    --judge-mode agents --openai-model gpt-5.5 \
    --run-id name_correction_v1
```

### "I have a suspect tag, find its family and clean it"
```bash
tagclean stage8 \
    --input ... --tag-answer ... \
    --seed-tag account_locked \
    --judge-mode agents --openai-model gpt-5.5 \
    --run-id account_locked_v1
```

### "Just rank, no GPT — fast offline pass"
```bash
tagclean stage6 --config my-config.yaml --judge-mode heuristic --no-resume
```

Heuristic mode flags rows that are duplicates / cross-tag duplicates / high artifact score. No semantic intent check — use only to sanity-check the ranker.

### "Run only Stage 5 batch preparation, submit later"
```bash
tagclean stage5 --config my-config.yaml --judge-mode batch_prepare    # writes batch JSONL
tagclean stage5 --config my-config.yaml --judge-mode batch_submit     # uploads to OpenAI Batch API
# ... wait up to 24h ...
tagclean stage5 --config my-config.yaml --judge-mode batch_collect    # ingests results
tagclean stage8 --config my-config.yaml                                # finish the pipeline
```

Batch API is 50% cheaper than sync but has 24h SLA. Worth it for full-corpus runs.

## Troubleshooting

### `--seed-tag` only resolved 2 of my 3 expected tags
The third was just below `boundary_policy_threshold` (default 0.85) on Gemma. The CLI's diagnostic line shows you the exact min(E5, Gemma) value. Either:
- Lower `boundary_policy_threshold` to 0.80 in config, OR
- Use `--target-tags` with the explicit list.

### OpenAI 400 "Invalid schema for response_format ... 'required' is required to be supplied"
A pydantic model used as `output_type` has a field with `default_factory=` or `default=`. OpenAI strict mode requires every property to be in `required`. Remove the default; let GPT supply `[]` if applicable.

### Stage 3 says "boundary policy passes disagree"
You're on an old version with multi-pass self-consistency. The current code uses single-pass + structural validation. `git pull`.

### Process exits with SIGSEGV (139) right after writing artifacts
Joblib's loky parallel backend (used inside sentence-transformers) leaks a semaphore on Python 3.14 / Mac shutdown. The pipeline already wrote everything before the crash. To suppress: `export TOKENIZERS_PARALLELISM=false` before running.

### Mac MPS slow / OOM, or no GPU
- Pass `--device cpu` to force CPU; embedding takes longer but is stable.
- For a 80k-row corpus on Mac CPU, expect ~15–30 min per model in Stage 1; on Linux+L4 it's well under 5.
- Set `TOKENIZERS_PARALLELISM=false` to also avoid the loky semaphore-leak warning.

### I accidentally staged run artifacts
The pre-commit hook should have refused. If you bypassed it or the hook isn't installed:
```bash
git config core.hooksPath tools/githooks    # install
git restore --staged runs/ artifacts/       # unstage
```
History rewrite (squash/reset) is fine while the branch isn't pushed; once pushed, leave it unless artifacts contain anything sensitive.

### Run is slow / hits OpenAI rate limits
- Reduce `audit_rows_per_packet` (default 24) so packets are smaller and GPT is faster per call.
- Lower `concurrency` for sync mode if you're rate-limited.
- For corpus-wide, use `--judge-mode batch_prepare/submit/collect`.

### "I want to verify the cleaned set is actually better"
After Stage 8 runs, `stage8/cleaning_report.json` has leave-one-out top-1 retrieval accuracy. >0.95 = cleanly separable in E5 space. Manually eyeball the top-5 of `cleaned.csv` per tag — they should be textbook examples of each intent.

### "Build the final production CSV from N family runs"
The end-to-end pipeline once you've cleaned several close-tag families:

```bash
# 1. Clean each family separately (one stage8 invocation per family)
tagclean stage8 --target-tags T1,T2,T3 --run-id family_a ...
tagclean stage8 --target-tags U1,U2,U3 --run-id family_b ...
tagclean stage8 --target-tags V1,V2,V3 --run-id family_c ...

# 2. Compose all family top-40s into a single production candidate CSV
tagclean compose \
    --from-runs family_a,family_b,family_c \
    --run-id production \
    --compose-source top40 \
    --out runs/production/composed_top40.csv

# 3. Run cross-family Stage 9 audit on the composed CSV
tagclean stage9 \
    --e5-audit-input runs/production/composed_top40.csv \
    --run-id production
```

The final E5-ready production set is `runs/production/stage9/production_filtered.csv`.

**Why compose from `top40` (not from per-family `stage9/production_filtered.csv`)?** A row that fails LOO inside a 3-tag family can be safely separable in the 9+-tag production union (its in-family rival is no longer a peer). Filter once globally, after composition. Per-family Stage 9 stays useful for per-family diagnostics; compose treats top40s as the production-recommended subset.

### "Filter the cleaned set against production E5 retrieval"
Production inference is E5-only (no Gemma). Stage 9 audits the cleaned set against an E5 leave-one-out retrieval and, by default, drops rows whose top-1 neighbor belongs to a different tag — the rows E5 itself confuses at inference time. Severity per row is reported as "own-share-top-K" (% of K nearest neighbors with the same tag).

```bash
# Audit + drop top-1 mismatches from the current run's cleaned.csv
tagclean stage9 --config my-config.yaml --run-id <existing>

# Report-only mode (no drop)
tagclean stage9 --config my-config.yaml --run-id <existing> --no-e5-drop

# Audit a unioned production set across multiple family runs
tagclean stage9 --config my-config.yaml \
    --e5-audit-input /path/to/combined_top40.csv \
    --run-id production_audit_v1
```

Outputs land in `<run>/stage9/`: `e5_neighbor_audit.csv` (per-row diagnostics), `production_filtered.csv` (the filtered set), `e5_dropped.csv`, `audit_report.json`. Stage 9 is opt-in; existing top40.csv ships unchanged.

## Costs reference

| Scope | GPT calls | $$ (gpt-5.5, high reasoning) | Wall time |
|---|---|---|---|
| 3-tag family (~540 rows) | ~12 | ~$0.50 | ~10 min |
| 6-tag families | ~25 | ~$1 | ~20 min |
| Full Bengali corpus (1394 tags, 79k rows) | ~4500–5500 | ~$50–150 | ~3–6 hours |

Embedding cost is GPU/Mac time, not API. Models are downloaded once (~3 GB).

## What NOT to do

- **Don't `--auto-cluster`** (not implemented, and not advised). Validate one family at a time.
- **Don't enable Stage 7 `review_enabled: true`** unless you have a specific reason; the buffer audit in Stage 5 already covers the same ground.
- **Don't `git add -A`** while a cleaning run is writing. Use the pre-commit hook (`tools/githooks/pre-commit`).
- **Don't commit `runs/` or `artifacts/`** — gitignore covers them; the hook double-protects.
