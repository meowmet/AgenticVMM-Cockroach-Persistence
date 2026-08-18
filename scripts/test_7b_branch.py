# scripts/test_7b_branch.py
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.engine.kv_ops import seq_copy, seq_rm, KVSequenceError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
N_SEQ_MAX = 8


def main():
    engine = LlamaEngine(
        model_path=MODEL_PATH,
        n_ctx=2048,      # leave as is if VRAM is limited, increase to 4096 if no issues
        n_seq_max=N_SEQ_MAX,
        n_gpu_layers=-1,
    )

    assert engine.n_seq_max() == N_SEQ_MAX, (
        f"expected {N_SEQ_MAX}, got {engine.n_seq_max()} — init fix did not apply"
    )
    print(f"[OK] n_seq_max at C layer = {engine.n_seq_max()}")

    # valid branch: 0 -> 1
    seq_copy(engine, src_seq_id=0, dst_seq_id=1)
    print("[OK] seq_copy(0, 1) succeeded")

    # deliberately trigger the guard instead of a segfault
    try:
        seq_copy(engine, src_seq_id=0, dst_seq_id=99)
        print("[FAIL] out-of-range dst_seq_id was not rejected")
    except KVSequenceError as e:
        print(f"[OK] out-of-range dst_seq_id correctly rejected: {e}")

    seq_rm(engine, seq_id=1)
    print("[OK] seq_rm(1) succeeded")

    engine.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
