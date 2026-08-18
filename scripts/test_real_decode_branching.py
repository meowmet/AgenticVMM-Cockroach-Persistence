# scripts/test_real_decode_branching.py
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
N_SEQ_MAX = 4

def main():
    engine = LlamaEngine(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_seq_max=N_SEQ_MAX,
        n_gpu_layers=-1,
    )
    bm = BranchManager(engine)

    root = bm.active_node()
    sys_prompt = "You are a cybersecurity expert assistant."
    sys_tokens = engine.tokenize(sys_prompt, add_bos=True)
    engine.generate(seq_id=root.seq_id, prompt_tokens=sys_tokens, pos_start=0, max_new_tokens=0)
    root.kv_pos = len(sys_tokens)

    base = bm.commit_and_generate("Hello, are you ready for cybersecurity analysis?", max_new_tokens=30)
    print(f"[base] {base.generated_text}")

    sql_branch = bm.create_branch(from_node_id=base.node_id)  # NO decode, instant
    sql_commit = bm.commit_and_generate("Start SQL Injection scan.", max_new_tokens=40)
    print(f"[SQL] {sql_commit.generated_text}")

    bm.checkout(base.node_id)
    xss_branch = bm.create_branch(from_node_id=base.node_id)  # NO decode, instant
    xss_commit = bm.commit_and_generate("Start XSS scan.", max_new_tokens=40)
    print(f"[XSS] {xss_commit.generated_text}")

    print(bm.render_tree())
    engine.close()

if __name__ == "__main__":
    main()
