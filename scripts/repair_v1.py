"""Single-pass repair on raw corpus.

Equivalent to `python -m tagclean.cleaner repair --config configs/bn_full.yaml
--run-id <id>`, but as a standalone script so it can be edited / extended.

Usage:
  PYTHONPATH=src python scripts/repair_v1.py [run_id]
  (run_id defaults to bn_account_locked_qa_v1; the cached embeddings live there)
"""
import sys, time, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tagclean.cleaner import load_config
from tagclean.repair import run_repair


def main(run_id: str = "bn_account_locked_qa_v1") -> int:
    cfg = load_config(REPO_ROOT / "configs" / "bn_full.yaml")
    cfg.run_id = run_id
    cfg.device = "cpu"
    cfg.input_csv = (REPO_ROOT / "data" / "question_tag.csv").resolve()
    cfg.tag_answer_json = (REPO_ROOT / "data" / "tag_answer.json").resolve()
    cfg.artifact_root = (REPO_ROOT / "runs").resolve()

    print(f"[v1] launching repair, run_id={cfg.resolved_run_id()}", flush=True)
    t0 = time.time()
    try:
        run_repair(cfg, resume=True)
        print(f"[v1] DONE in {(time.time()-t0)/60:.1f} min", flush=True)
        return 0
    except Exception:
        print(f"[v1] FAILED after {(time.time()-t0)/60:.1f} min", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bn_account_locked_qa_v1"))
