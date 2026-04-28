# scripts/

Standalone scripts that produced the cleaned dataset(s). Useful for reproducibility and as a starting point if you want to chain multiple repair passes manually.

These are wrappers around the `tagclean.repair` API — the canonical entry point is still `python -m tagclean.cleaner repair --config configs/bn_full.yaml --run-id <id>`. Use these scripts when you want to:
- Chain V1 → V2 → V3 revisions explicitly,
- Override defaults inline without editing `RepairConfig`,
- Customize the output naming or post-processing.

## What produced the repo's `xray_cleanup.csv`

- **`repair_aggressive_single_pass.py`** — single pass on raw 78,990 rows with the aggressive defaults shipped in `RepairConfig`. Loop aborts on `drop_cap_strikes` at iter 4 with **37,355 rows / 1,334 tags** surviving. Promotes the result to repo-root `xray_cleanup.csv`.

## Multi-revision chain (historical — for the conservative-threshold runs)

The earlier 70k-row output was produced by chaining three passes:

1. **`repair_v1.py`** — single pass at the (then-conservative) defaults. Equivalent to `tagclean repair`. Output: `runs/<run_id>/repair/xray_cleanup.csv`.
2. **`repair_v2.py`** — reads V1's `final_assignment.parquet`, filters to surviving rows, runs the loop again. Output: `runs/<run_id>/repair_v2/xray_cleanup_v2.csv`.
3. **`repair_v3.py`** — same shape, reads V2's output. Diminishing returns by V3. Output: `runs/<run_id>/repair_v3/xray_cleanup_v3.csv`.

Useful if a run hits `abort_drop_cap` and you want to keep applying gates to the survivors. With the current aggressive defaults, V1 alone hits the target.

## Calling convention

All scripts accept an optional `run_id` arg (defaults to `bn_account_locked_qa_v1`, which is where the cached Stage 0/1 embeddings live):

```bash
PYTHONPATH=src python scripts/repair_v1.py
PYTHONPATH=src python scripts/repair_v2.py            # uses V1's output
PYTHONPATH=src python scripts/repair_v3.py            # uses V2's output
PYTHONPATH=src python scripts/repair_aggressive_single_pass.py
```

Each script auto-detects the repo root from `__file__` and reads embeddings from `runs/<run_id>/stage1/`. They expect Stage 0/1 to have already produced cached embeddings — run the legacy stages or `tagclean repair` once to bootstrap if needed.

## Why these scripts and not just CLI flags

The existing `tagclean repair` CLI does single-pass V1. Multi-revision chaining requires loading the previous run's `final_assignment.parquet` and re-seeding a `RepairState` with the surviving rows + their repaired tags — there's no clean way to express that as a CLI flag without baking the chain logic into the loop itself. The scripts keep the chain logic explicit and editable.

If you find yourself frequently chaining, consider adding a `--from-revision RUN_ID` flag to the CLI.
