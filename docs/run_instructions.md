# tagclean — Run Instructions

How to install, configure, and run the cleaner. For **why** the pipeline is shaped this way, see `docs/harness_design.md`.

## Prerequisites

- Python 3.10+ (3.14 tested)
- Claude Code CLI installed and logged in (`claude /login`)
- ~2 GB free for cached HuggingFace models (E5 1024d)
- Optional GPU (CUDA on Linux, MPS on Mac); CPU works too, slower

## Install

```bash
git clone <this repo> ~/tagclean
cd ~/tagclean
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q
```

Install Claude Code CLI separately (see https://claude.com/claude-code) and run `claude /login` once. tagclean spawns `claude -p` per call; it strips `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL` from the subprocess env so calls always route through your subscription, not an API key.

A startup probe runs a trivial `claude -p` call before launching any pipeline that uses Claude — this catches "not logged in" failures in seconds rather than 20 minutes into a run.

Install the pre-commit guard so you don't accidentally check in run artifacts:

```bash
git config core.hooksPath tools/githooks
chmod +x tools/precommit_artifact_check.sh
```

## Quick start — clean three specific tags

The most common workflow: you know which close-sibling tags are leaky, and you want to clean them as a unit.

```bash
tagclean stage8 \
    --input /path/to/question_tag.csv \
    --tag-answer /path/to/tag_answer.json \
    --target-tags account_locked,account_locked_retrials,account_locked_unlock_request \
    --run-id account_locked_clean_v1
```

This runs all stages 0→8 with sensible defaults (Stage 3 = Opus/high, Stage 5 = Sonnet/medium). Output lands in `artifacts/account_locked_clean_v1/`. Time: ~5–10 min for ~540 rows.

## Three ways to pick which tags to clean

### 1. Manual list — `--target-tags`
You know exactly which tags. Comma-separated, no spaces:

```bash
tagclean stage8 --target-tags T1,T2,T3 ...
```

### 2. Seed expansion — `--seed-tag`
You know one suspect tag. The harness finds its close-tag cluster automatically (cosine ≥ `boundary_policy_threshold` in E5):

```bash
tagclean stage8 --seed-tag account_locked ...
```

The CLI prints the resolved cluster **and** the top-5 nearest excluded tags with their similarity scores. Note: at full corpus scale (1394 tags) this can collapse into a mega-component. Use `--target-tags` directly when you see it.

### 3. Whole corpus — omit both flags
```bash
tagclean stage8 --input ... --tag-answer ...
```

This processes every tag and discovers all close-tag clusters automatically. **High Claude-call volume** — only do it once you've validated on individual families.

## Configuration

`configs/example.yaml` is the starter template. Copy and edit:

```bash
cp configs/example.yaml my-config.yaml
```

The fields you'll actually edit for a Bengali NID run:

```yaml
input_csv: ~/ec-faq-bot/full_dataset/question_tag.csv
tag_answer_json: ~/ec-faq-bot/full_dataset/tag_answer.json
artifact_root: runs                  # gitignored
run_id: account_locked_clean_v1

language: bn

e5_instruction: |
  You are an expert at matching Bangladeshi National Identity Card (NID) and
  voter registration queries. Identify the most semantically relevant question
  by intent and concrete details, prioritizing exact phrase matches.

embedding_backend: sentence-transformers
judge_mode: claude
stage3_model: opus
stage3_effort: high
stage5_model: sonnet
stage5_effort: medium

# Tune these only if --seed-tag misses an obvious sibling or the audit looks too aggressive.
boundary_policy_threshold: 0.85
audit_buffer_size: 80
top_n: 40
```

Then run with `--config my-config.yaml`; CLI flags still override individual fields.

## Outputs

For a run with `run_id=foo`, in `<artifact_root>/foo/`:

```
foo/
├── run_manifest.json                   ← top-level identity (version, hashes, models)
├── stage0/intake.parquet
├── stage1/emb_e5.npy + faiss index
├── stage2/{tag_profile.parquet, tag_centroids_e5.npy, tag_index.json}
├── stage3/
│   ├── tag_merge_map.csv               ← old_tag → canonical_tag
│   ├── tag_boundary_policy.jsonl       ← Claude-authored discriminative rules
│   ├── merge_candidates.jsonl
│   └── llm_calls.jsonl                 ← per-call telemetry (cost, cache, subtype)
├── stage4/row_features.parquet         ← composite_score, audit_buffer flag
├── stage5/
│   ├── audit_results.jsonl             ← per-row keep/flag with reasons
│   ├── audit_packets/                  ← raw Claude packet inputs/outputs
│   └── llm_calls.jsonl
├── stage6/
│   ├── question_tag.cleaned.csv        ← all audit-pass rows, ranked
│   ├── question_tag.top40.csv          ← top-N per tag, 2-col `question,tag`
│   └── jettisoned_rows.csv             ← dropped rows + reasons + Claude rationale
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
tagclean stage8 --config my-config.yaml      # → all (skips deleted stage7)
tagclean all --config my-config.yaml         # alias for stage8
```

Re-running with `--resume` (default) skips any stage whose input hash matches its manifest. Stage 3 / Stage 5 hashes include LLM identity (model, effort, prompt-version, schema fingerprint), so a model change forces a re-run rather than reusing stale boundary policies.

`--no-resume` forces a clean rerun of the requested stage and downstream.

## Recipes

### "Clean these 3 specific Bengali tags right now"
```bash
tagclean stage8 \
    --input ~/ec-faq-bot/full_dataset/question_tag.csv \
    --tag-answer ~/ec-faq-bot/full_dataset/tag_answer.json \
    --target-tags name_correction_in_nid_card,parents_name_correction_new,spouse_name_correction_new \
    --run-id name_correction_v1
```

### "I have a suspect tag, find its family and clean it"
```bash
tagclean stage8 \
    --input ... --tag-answer ... \
    --seed-tag account_locked \
    --run-id account_locked_v1
```

### "Just rank, no LLM — fast offline pass"
```bash
tagclean stage6 --config my-config.yaml --judge-mode heuristic --no-resume
```

Heuristic mode flags rows that are duplicates / cross-tag duplicates / high artifact score. No semantic intent check — use only to sanity-check the ranker.

### "Force a different model on Stage 5"
```bash
tagclean stage8 ... --stage5-model opus --stage5-effort high
```

Stage 3 + Stage 5 input hashes include model + effort, so this forces a clean re-run of those two stages without invalidating Stage 1/2/4 caches.

## Troubleshooting

### Startup: "auth probe failed [infra]: ... not logged in"
Run `claude /login` (or `claude auth login`) once and retry. The probe spawns a trivial `claude -p` call under the same env-stripped subprocess that real pipeline calls use, so a successful probe means the real calls will work.

### Startup: "claude binary not found"
Install Claude Code CLI (https://claude.com/claude-code) and ensure `claude` is on PATH (or in `~/.local/bin`, `/opt/homebrew/bin`, or `/usr/local/bin` — the wrapper checks those fallbacks).

### `--seed-tag` only resolved 2 of my 3 expected tags
The third was just below `boundary_policy_threshold` (default 0.85) on E5 cosine. The CLI's diagnostic line shows you the exact value. Either lower `boundary_policy_threshold` to 0.80 in config, OR use `--target-tags` with the explicit list.

### Stage 5 halted with "CIRCUIT BREAKER: last 10 packets all failed"
Sustained Claude failures — probably rate limits or an account issue. Inspect `stage5/llm_calls.jsonl`: look at `subtype` and `error_message` of recent failures. Common causes:
- Rate-limit storm: lower `concurrency` (default 6) and rerun.
- `--effort max` with a long prompt: switch to `medium`.
- Model overload: `--claude-fallback-model` should already kick in; check that the fallback isn't also hitting limits.

### Process exits with SIGSEGV (139) right after writing artifacts
Joblib's loky parallel backend (used inside sentence-transformers) leaks a semaphore on Python 3.14 / Mac shutdown. The pipeline already wrote everything before the crash. To suppress: `export TOKENIZERS_PARALLELISM=false` before running.

### Mac MPS slow / OOM, or no GPU
- Pass `--device cpu` to force CPU; embedding takes longer but is stable.
- For an 80k-row corpus on Mac CPU, expect ~15–30 min in Stage 1; on Linux+L4 it's well under 5.
- Set `TOKENIZERS_PARALLELISM=false` to also avoid the loky semaphore-leak warning.

### I accidentally staged run artifacts
The pre-commit hook should have refused. If you bypassed it or the hook isn't installed:
```bash
git config core.hooksPath tools/githooks    # install
git restore --staged runs/ artifacts/       # unstage
```

### "I want to verify the cleaned set is actually better"
After Stage 8 runs, `stage8/cleaning_report.json` has leave-one-out top-1 retrieval accuracy. >0.95 = cleanly separable in E5 space. Manually eyeball the top-5 of `cleaned.csv` per tag — they should be textbook examples of each intent.

### "I want to track Claude cost / cache hit drift across runs"
Inspect `stageN/llm_calls.jsonl`: each line has `total_cost_usd`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `subtype`, `num_turns`. Sum across runs to see whether prompt drift is busting the cache.

### "I want to change which tags get cleaned"
- Editing `runs/families.yaml` (after `tagclean discover`) is the canonical way. Flip `status: approved` ↔ `rejected`, or edit `target_tags` per family. `run-families --skip-completed` (default) won't re-clean families whose stage8 outputs already exist.
- For one-off runs, pass `--target-tags T1,T2,T3` directly to `tagclean stage8`.
- To restrict `discover` to a curated subset, write a `production_tags.txt` (one tag per line) and pass `--production-tags production_tags.txt`. Unknown tags hard-fail.

### "Build the final production CSV from N family runs"
```bash
# 1. Clean each family separately
tagclean stage8 --target-tags T1,T2,T3 --run-id family_a ...
tagclean stage8 --target-tags U1,U2,U3 --run-id family_b ...
tagclean stage8 --target-tags V1,V2,V3 --run-id family_c ...

# 2. Compose all family top-40s into a single production candidate CSV
tagclean compose \
    --from-runs family_a,family_b,family_c \
    --run-id production \
    --compose-source top40 \
    --out runs/production/composed_top40.csv

# 3. Cross-family Stage 9 audit on the composed CSV
tagclean stage9 \
    --e5-audit-input runs/production/composed_top40.csv \
    --run-id production
```

The final E5-ready production set is `runs/production/stage9/production_filtered.csv`.

**Why compose from `top40` (not from per-family `stage9/production_filtered.csv`)?** A row that fails LOO inside a 3-tag family can be safely separable in the 9+-tag production union (its in-family rival is no longer a peer). Filter once globally, after composition.

### "Filter the cleaned set against production E5 retrieval"
Production inference is E5-only. Stage 9 audits the cleaned set against an E5 leave-one-out retrieval and, by default, drops rows whose top-1 neighbor belongs to a different tag.

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

| Scope | Claude calls | Wall time |
|---|---|---|
| 3-tag family (~540 rows) | ~12 (1 Stage3 + ~10 Stage5) | ~5–10 min |
| 6-tag families | ~25 | ~10–20 min |
| Full Bengali corpus (1394 tags, 79k rows) | ~4500 (~270 Stage3 + ~4200 Stage5) | ~3–6 hours |

All Claude calls bill against the Claude Code subscription, not an API key. Per-call cost is recorded in `stageN/llm_calls.jsonl` for postmortem.

Embedding cost is GPU/Mac time, not API. Models are downloaded once (~2 GB).

## What NOT to do

- **Don't enable Stage 7.** It's been deleted; the dispatcher already skips it. The `review_*` config knobs from prior versions are ignored.
- **Don't `git add -A`** while a cleaning run is writing. Use the pre-commit hook (`tools/githooks/pre-commit`).
- **Don't commit `runs/` or `artifacts/`** — gitignore covers them; the hook double-protects.
- **Don't set `ANTHROPIC_API_KEY` if you want subscription billing.** The wrapper strips it from the subprocess env; if it leaks back via your shell config, calls flip to API-key billing silently.
