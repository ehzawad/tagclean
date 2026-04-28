"""Aggressive-threshold repair: single pass from raw to ~40k rows.

This is the script that produced the repo-root xray_cleanup.csv (37,355 rows).
Reads cached Stage 1 embeddings, runs the repair loop with the aggressive
defaults shipped in RepairConfig, and copies the final output to repo root.

If the loop's first pass leaves > 50k rows alive (e.g. on a less paraphrase-
dense corpus where the aggressive thresholds are still conservative for it),
chains a V2 pass automatically. On Bengali NID one pass is sufficient — the
loop aborts on drop_cap_strikes around iter 4 with ~37k rows.

Usage:
  PYTHONPATH=src python scripts/repair_aggressive_single_pass.py [run_id]
"""
import json, sys, time, traceback, shutil
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tagclean.repair import RepairConfig, run_repair_loop
from tagclean.repair_state import RepairState, STATUS_KEPT


def main(run_id: str = "bn_account_locked_qa_v1") -> int:
    run_dir = REPO_ROOT / "runs" / run_id
    out_dir = run_dir / "repair_aggressive_v1"

    print("[agg] loading raw inputs", flush=True)
    emb = np.load(run_dir / "stage1" / "emb_e5.npy")
    emb_rows = pd.read_parquet(run_dir / "stage1" / "embedding_rows.parquet")
    print(f"[agg] raw: {len(emb_rows)} rows × {emb_rows['tag'].nunique()} tags", flush=True)

    cfg = RepairConfig()  # picks up the aggressive defaults shipped in repair.py
    print(f"[agg] hard_drop_thresh   = {cfg.hard_drop_thresh}", flush=True)
    print(f"[agg] cross_tag_dup_thresh = {cfg.cross_tag_dup_thresh}", flush=True)
    print(f"[agg] merge_cosine_thresh  = {cfg.merge_cosine_thresh}", flush=True)
    print(f"[agg] drop_cap_per_tag     = {cfg.drop_cap_per_tag}", flush=True)

    state = RepairState.from_inputs(
        embeddings=emb,
        row_ids=emb_rows["row_id"].astype(np.int64).to_numpy(),
        question_raw=emb_rows["question_raw"].astype(str).tolist(),
        tags_per_row=emb_rows["tag"].astype(str).tolist(),
        move_budget=cfg.move_budget,
    )

    print(f"[agg] V1 starting on {state.n_rows} rows × {state.n_tags} tags", flush=True)
    t0 = time.time()
    try:
        result = run_repair_loop(state, cfg, artifact_dir=out_dir)
        print(f"[agg] V1 ended: status={result['final_status']}, iters={result['iters_run']}", flush=True)
    except Exception:
        traceback.print_exc()
        return 1

    final = state.final_assignment_frame()
    final.to_parquet(out_dir / "final_assignment.parquet", index=False)
    out = final[final["status"] == STATUS_KEPT][["question", "repaired_tag"]].rename(
        columns={"repaired_tag": "tag"}
    )
    out.to_csv(out_dir / "xray_cleanup.csv", index=False)
    n_kept = len(out)
    print(f"[agg] V1 result: {n_kept} kept / {len(final)} input", flush=True)

    # If many rows still alive, chain a V2 pass automatically.
    if n_kept > 50000:
        print("[agg] running V2 to push further", flush=True)
        out_v2 = run_dir / "repair_aggressive_v2"
        kept_frame = final[final["status"] == STATUS_KEPT].reset_index(drop=True)
        out_rid = kept_frame["row_id"].astype(np.int64)
        rid_to_pos = {int(r): i for i, r in enumerate(emb_rows["row_id"].astype(np.int64))}
        kept_pos = np.array([rid_to_pos[int(r)] for r in out_rid], dtype=np.int64)
        state2 = RepairState.from_inputs(
            embeddings=emb[kept_pos],
            row_ids=out_rid.to_numpy(),
            question_raw=kept_frame["question"].astype(str).tolist(),
            tags_per_row=kept_frame["repaired_tag"].astype(str).tolist(),
            move_budget=cfg.move_budget,
        )
        run_repair_loop(state2, cfg, artifact_dir=out_v2)
        final2 = state2.final_assignment_frame()
        final2.to_parquet(out_v2 / "final_assignment.parquet", index=False)
        out2 = final2[final2["status"] == STATUS_KEPT][["question", "repaired_tag"]].rename(
            columns={"repaired_tag": "tag"}
        )
        out2.to_csv(out_v2 / "xray_cleanup.csv", index=False)
        print(f"[agg] V2 result: {len(out2)} kept", flush=True)
        final_csv = out_v2 / "xray_cleanup.csv"
        final_count = len(out2)
    else:
        final_csv = out_dir / "xray_cleanup.csv"
        final_count = n_kept

    # Promote final to repo root.
    shutil.copy(final_csv, REPO_ROOT / "xray_cleanup.csv")
    print(f"[agg] DONE in {(time.time()-t0)/60:.1f} min — final: {final_count} rows", flush=True)
    print(f"[agg] copied {final_csv} → {REPO_ROOT / 'xray_cleanup.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bn_account_locked_qa_v1"))
