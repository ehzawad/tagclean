# tagclean

LLM-assisted dataset cleaner for tagged FAQ corpora — built specifically for the Bangladesh Election Commission NID/voter Bengali FAQ domain. You give it a `question_tag.csv` (and optionally `tag_answer.json`); it gives you back a cleaned subset where ambiguous, duplicated, and synthetic-looking rows are jettisoned and the survivors are ranked by per-tag fit.

## Why this exists

GPT-generated FAQ datasets routinely look fine within a single tag but blur at tag boundaries. Bengali NID/voter data is the hard case: many tags are slight rewordings of each other (`name_correction_in_nid_card` vs `parents_name_correction_new` vs `spouse_name_correction_new`; `account_locked` vs `account_locked_retrials` vs `account_locked_unlock_request`), and the model that handles inference at runtime is fixed — it's E5. Cleaning by hand is slow; throwing GPT at every row is expensive and unreliable.

`tagclean` runs a deterministic embedding ranker for selection and reserves GPT for two narrow jobs only: writing the discriminative rules between close tags, and auditing the top-N buffer per tag.

## Pipeline

```
question_tag.csv  →  Stage 0  intake/normalize/dedup
                  →  Stage 1  E5 + Gemma embeddings
                  →  Stage 2  per-tag medoid + central rows + discriminative phrases
                  →  Stage 3  GPT writes per-tag boundary rules for close-tag clusters
                  →  Stage 4  deterministic ranker → top-N audit buffer per tag
                  →  Stage 5  GPT audits buffer rows (keep / flag with reasons)
                  →  Stage 6  walk audited-pass rows, MMR-diversify, take top K
                  →  Stage 8  leave-one-out validation (optional)
```

GPT call budget for a 3-tag cluster: ~12 calls total (~1 boundary policy + ~10 audit packets).

## Roles

| Component | Role |
|---|---|
| **E5-multilingual-large-instruct** (1024d) | Primary geometry. Defines "tag fit". Owns ~50% of ranking weight. Also what production uses at inference. |
| **EmbeddingGemma-300m** (768d) | Cross-validation. Catches E5 hallucinations via rank disagreement. Owns ~30% of ranking weight. |
| **GPT-5.5 / 5.4** | Tag-boundary rule authoring (Stage 3) + buffer audit (Stage 5). Never scores rows directly. |

## Quickstart

```bash
git clone <this repo> && cd tagclean
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q                              # unit tests, all offline (no API)
```

Real run with GPT-authored boundary policy + audit:

```bash
export OPENAI_API_KEY=sk-...
cp configs/example.yaml my-config.yaml
# edit my-config.yaml: set input_csv, tag_answer_json, run_id, e5_instruction
tagclean stage8 --config my-config.yaml --judge-mode agents --openai-model gpt-5.5
```

Or fully via CLI flags:

```bash
tagclean stage8 \
    --input data/question_tag.csv \
    --tag-answer data/tag_answer.json \
    --seed-tag account_locked \
    --judge-mode agents --openai-model gpt-5.5 \
    --run-id account_locked_clean_v1
```

## Inputs

- **`question_tag.csv`** — required. Columns: `question`, `tag`. UTF-8.
- **`tag_answer.json`** — optional. `{ "<tag>": "<canonical answer>" }`. Used only as a merge-safety gate in Stage 3 so we don't merge two tags whose answers are substantively different.

## Outputs

For a run with `run_id=foo`, in `<artifact_root>/foo/`:

| File | What |
|---|---|
| `run_manifest.json` | Top-level run identity (version, file hashes, model names, config) |
| `stage3/tag_merge_map.csv` | `old_tag → canonical_tag` if any tags were merged |
| `stage3/tag_boundary_policy.jsonl` | GPT-authored discriminative rules per close-tag cluster |
| `stage6/question_tag.cleaned.csv` | All audit-pass rows, ranked, with composite score + rationale |
| `stage6/question_tag.top40.csv` | Top-N per tag (`production_recommended` subset), 2-col `question,tag` |
| `stage6/jettisoned_rows.csv` | Dropped rows with `status_reason` and GPT rationale |
| `stage8/cleaning_report.json` | Leave-one-out retrieval accuracy + confusion matrix |

## CLI

```
tagclean <stage> [--config FILE] [--judge-mode MODE] [--openai-model NAME]
                 [--input PATH] [--tag-answer PATH] [--artifact-root PATH]
                 [--target-tags T1,T2,T3] [--seed-tag X]
                 [--language bn|none] [--no-resume]
```

Stages: `stage0 stage1 stage2 stage3 stage4 stage5 stage6 stage7 stage8 all`.
Judge modes: `heuristic` (no GPT), `sync`, `agents`, `batch_prepare`, `batch_collect`.

### Picking which tags to clean

Three ways, in order of automation:

```bash
# manual: explicit tag list
tagclean stage8 --target-tags name_correction_in_nid_card,parents_name_correction_new,spouse_name_correction_new

# semi-automatic: give one tag, the harness expands to its close-tag cluster
tagclean stage8 --seed-tag name_correction_in_nid_card

# corpus-wide: omit both flags; clean every tag, discover all clusters
tagclean stage8
```

`--seed-tag X` runs Stages 0–2 first to get tag centroids, then walks the close-tag graph (cosine ≥ `boundary_policy_threshold` in BOTH E5 and Gemma) and resolves the connected component containing X. The resolved cluster is printed alongside the top-5 *excluded* tags with their similarity scores, so you immediately see which siblings missed the threshold and can decide whether to lower it or pass `--target-tags` instead. Singletons fall back to cleaning just X.

Each stage chains its predecessors with cached resume; rerunning `stage6` with `--resume` skips already-finished stages.

## Configuration

See `configs/example.yaml`. Key knobs:

| Field | Default | Purpose |
|---|---|---|
| `language` | `bn` | `bn` runs Bengali Unicode normalization via `bnunicodenormalizer`; `none` skips |
| `e5_instruction` | (default) | Domain-specific E5-instruct prompt; override for your dataset |
| `audit_buffer_size` | 80 | Top-N per tag sent to GPT for audit |
| `audit_rows_per_packet` | 24 | Rows per GPT audit call |
| `boundary_policy_threshold` | 0.85 | Centroid sim above which tags form a "close-tag cluster" (used for both Stage 3 policy and `--seed-tag`) |
| `top_n` | 40 | Final per-tag rows marked `production_recommended` |
| `judge_mode` | `sync` | `heuristic` for offline tests, `agents`/`sync` for real GPT |

## Architecture choices

- **Plain Python, no agent framework.** The pipeline is a deterministic DAG of stages with parquet/jsonl artifacts. Each stage is idempotent on its inputs.
- **GPT only at narrow points.** Per-row LLM judging was tried and discarded: it was expensive, unreliable, and dominated the ranking. The current shape uses LLMs as a *policy author* and *auditor*, not a *scorer*.
- **Default Bengali normalization.** This tool was built for Bengali NID/voter FAQ data; English or multilingual users can pass `--language none`.

## Tests

```bash
pytest
```

13 unit tests covering helpers (text normalization, MMR ranking, cluster discovery, packet building, audit decisions). End-to-end validation is done by running the pipeline against your actual dataset — the harness produces `stage8/cleaning_report.json` with leave-one-out retrieval accuracy as the empirical signal.

## License

MIT.
