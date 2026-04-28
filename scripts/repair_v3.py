"""Third revision: repair on top of repair_v2's xray_cleanup_v2.csv.

Reads V2's final_assignment.parquet, filters to surviving rows, runs the loop
once more. Diminishing returns by V3 — most of the work is done in V1/V2.

Output: runs/<v1_run_id>/repair_v3/{xray_cleanup_v3.csv, final_assignment.parquet, ...}

Usage:
  PYTHONPATH=src python scripts/repair_v3.py [v1_run_id]
"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tagclean.repair import RepairConfig, run_repair_loop
from tagclean.repair_state import RepairState, STATUS_KEPT


def main(v1_run_id: str = "bn_account_locked_qa_v1") -> int:
    v1_dir = REPO_ROOT / "runs" / v1_run_id
    v2_repair = v1_dir / "repair_v2"
    v3_dir = v1_dir / "repair_v3"

    print("[v3] loading V2 outputs", flush=True)
    emb = np.load(v1_dir / "stage1" / "emb_e5.npy")
    emb_rows = pd.read_parquet(v1_dir / "stage1" / "embedding_rows.parquet")
    v2_final = pd.read_parquet(v2_repair / "final_assignment.parquet")

    row_id_to_pos = {int(r): i for i, r in enumerate(emb_rows["row_id"].astype(np.int64))}
    kept = v2_final[v2_final["status"] == STATUS_KEPT].reset_index(drop=True)
    print(f"[v3] V2 kept {len(kept)} rows", flush=True)

    positions = np.array([row_id_to_pos[int(r)] for r in kept["row_id"]], dtype=np.int64)
    cfg = RepairConfig()
    state = RepairState.from_inputs(
        embeddings=emb[positions],
        row_ids=kept["row_id"].astype(np.int64).to_numpy(),
        question_raw=kept["question"].astype(str).tolist(),
        tags_per_row=kept["repaired_tag"].astype(str).tolist(),
        move_budget=cfg.move_budget,
    )

    print(f"[v3] starting loop on {state.n_rows} rows × {state.n_tags} tags", flush=True)
    t0 = time.time()
    try:
        result = run_repair_loop(state, cfg, artifact_dir=v3_dir)
        print(f"[v3] loop ended: status={result['final_status']}, iters={result['iters_run']}", flush=True)
    except Exception:
        traceback.print_exc()
        return 1

    final = state.final_assignment_frame()
    final.to_parquet(v3_dir / "final_assignment.parquet", index=False)
    out = final[final["status"] == STATUS_KEPT][["question", "repaired_tag"]].rename(
        columns={"repaired_tag": "tag"}
    )
    out.to_csv(v3_dir / "xray_cleanup_v3.csv", index=False)

    n_in = len(final)
    report = {
        "rows_in": n_in,
        "rows_kept": len(out),
        "rows_dropped": int((final["status"] == "dropped").sum()),
        "rows_reassigned_or_merged": int((final["original_tag"] != final["repaired_tag"]).sum()),
        "tags_in": state.n_tags,
        "tags_alive_final": int(state.alive_tags.sum()),
        "loop_status": result["final_status"],
        "iters_run": result["iters_run"],
    }
    with (v3_dir / "repair_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[v3] {len(out)}/{n_in} rows kept; {state.alive_tags.sum()}/{state.n_tags} tags alive", flush=True)
    print(f"[v3] DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[v3] wrote {v3_dir / 'xray_cleanup_v3.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bn_account_locked_qa_v1"))
