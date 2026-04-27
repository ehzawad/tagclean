# tagclean

Pure-geometry dataset cleaner for tagged FAQ corpora. Frozen E5 embedding model + deterministic iterative repair. Built for the Bangladesh Election Commission Bengali NID/voter FAQ corpus (≈79k GPT-generated questions across 1,394 close-sibling tags), but the harness is language-neutral when you set `language: none`.

You give it `question_tag.csv`; the pipeline gives back **`xray_cleanup.csv`** — a dataset where every surviving row's top-1 LOO neighbor in E5 space is in its own tag, the property production retrieval needs.

## What's in the repo

```
data/question_tag.csv           — input corpus (question, tag)
xray_cleanup.csv                — cleaned output (question, tag), the deliverable
src/tagclean/repair.py          — the geometric repair loop
src/tagclean/repair_state.py    — RepairState dataclass
src/tagclean/cleaner.py         — pipeline stages 0/1/2/3/4/6/8/9 + CLI
configs/bn_full.yaml            — Bengali NID full-corpus config (1024d E5, thresholds)
docs/repair_design.md           — system design (the repair loop's algorithm)
docs/repair_thresholds.md       — every threshold explained with worked examples
docs/harness_design.md          — older stage-based pipeline (legacy reference)
docs/run_instructions.md        — operational runbook for the legacy stages
tests/test_repair.py            — 29 unit + integration tests
```

## How to clean a corpus (one command)

```bash
PYTHONPATH=src /home/synesis/venv-election-commission/bin/python \
    -m tagclean.cleaner repair \
    --config configs/bn_full.yaml \
    --run-id xray_v1
```

Outputs:

```
runs/xray_v1/repair/
├── xray_cleanup.csv          ← the deliverable (question, tag)
├── final_assignment.parquet  ← per-row (row_id, original_tag, repaired_tag, status, margin)
├── iter_metrics.jsonl        ← per-iter scalars (drops/merges/margin distribution)
├── iter_NN_ops.jsonl         ← per-op log (row drops, merges, reassignments)
└── repair_report.json        ← summary (rows in/kept/dropped, tags alive, loop status)
```

Wall-time: ~12-15 min on CPU once Stage 0/1 embeddings are cached. First run pays an additional ~30 min for the E5 embedding pass.

Resource caps locked into `repair.py`: 4 OMP/MKL/OpenBLAS/FAISS threads, peak working set < 2 GB.

## The result on the Bengali NID corpus

Three sequential revisions at default thresholds:

| Stage | Rows | Tags | Drops this rev | Tag merges this rev | Loop status |
|---|---:|---:|---:|---:|---|
| Raw input | 78,990 | 1,394 | — | — | — |
| **V1** (single repair pass) | 70,445 | 1,344 | 8,545 | 50 | aborted on drop-cap |
| V2 (repair on V1) | 70,172 | 1,286 | 273 | 58 | converged |
| **V3** (repair on V2) | **70,172** | **1,278** | **0** | **8** | **converged** |

`pct_neg_margin` (fraction of rows whose own-tag cosine is below their best other-tag cosine): 21.5% → 17.7% → 16.7% → 16.1% (floor at default thresholds).

The repo's `xray_cleanup.csv` is the V3 output. To reach V3 from raw, run `tagclean repair` three times; each run takes the previous run's output as input.

## What the loop does

| Operation | Effect |
|---|---|
| **Reassign** | A row that geometrically belongs in another tag (margin < threshold AND kNN-majority points at it) gets moved. No data lost. |
| **Drop** | A row whose top-1 LOO neighbor lives in another tag, with no clear "winner" between the pair, is removed. Last resort. |
| **Merge tags** | Two tags whose centroids are E5-indistinguishable AND mutually confused are collapsed into one canonical. Rows survive, just under a unified label. |
| **Dissolve tiny tags** | Tags with < 3 rows after all moves are dissolved (rows drop). |

All decisions are pure E5 geometry: centroid cosine, margin (own − best-other), per-row top-K neighbor majority, mutual directed confusion between tag pairs. **No `tag_answer.json`, no tag-name patterns, no LLM**.

## Why pure geometry

User constraint: *"don't rely on answer or tag names — they will add noise."* Tag names follow brittle conventions (`_followup_[a-d]$` regex hacks); `tag_answer.json` is human-curated and frequently stale. The geometry of the embedding space is the most reliable single source of truth for "which questions production retrieval can route correctly."

Trade-off, made explicit: this pipeline optimizes for **E5-routability**, not truth-preserving taxonomy. If two tags are E5-indistinguishable but have materially different correct answers, the loop will collapse them — production E5 couldn't have distinguished them anyway, so keeping them separate would just make the bot guess wrong. The fix to that case is a different retriever, not a different cleaning pipeline.

For the deeper rationale, design choices, and worked examples of every threshold, see:
- **[docs/repair_design.md](docs/repair_design.md)** — the algorithm
- **[docs/repair_thresholds.md](docs/repair_thresholds.md)** — every knob with concrete examples

## The legacy stage pipeline

The repair loop replaced the older Stage 0→9 deterministic chain. The old stages (cleaner.py:run_stage0, run_stage1, run_stage4, run_stage6, run_stage8, run_stage9) still work — see `docs/run_instructions.md` and `docs/harness_design.md` for the legacy recipe. The legacy pipeline still uses some `tag_answer.json` and `_followup_*` heuristics in Stage 3; the repair loop bypasses Stage 3 entirely.

## Tests

```bash
PYTHONPATH=src /home/synesis/venv-election-commission/bin/python -m pytest tests/ -q
```

29 tests for the repair loop (geometry primitives, reassignment gates, cross-tag triage, merge with hysteresis, absorber recovery, end-to-end convergence on synthetic fixtures), plus the legacy `tests/test_cleaner.py`.

## Repo state

- Branch `geometric-repair`: this work (pure-geometry repair loop).
- Branch `deterministic-only`: the prior stripped-Claude state without the repair loop.
- Branch `claude-cli-stage-qa`: the prior Claude-assisted pipeline (historical reference).

## License

MIT.
