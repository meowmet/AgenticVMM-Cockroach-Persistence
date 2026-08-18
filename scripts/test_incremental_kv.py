import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

print(">>> Initializing Engine and BranchManager...")
engine = LlamaEngine(model_path=MODEL_PATH, n_ctx=2048, n_seq_max=4, n_gpu_layers=-1)
bm = BranchManager(engine)

# 1. System Prompt seed decoding
root = bm.active_node()
sys_tokens = engine.tokenize("You are a cybersecurity assistant.", add_bos=True)
engine.generate(seq_id=root.seq_id, prompt_tokens=sys_tokens, pos_start=0, max_new_tokens=0)
root.kv_pos = len(sys_tokens)
print(f"[ROOT SEED] System prompt decode edildi. kv_pos={root.kv_pos}")

# 2. Delta Commit 1
c1 = bm.commit_and_generate("Merhaba!", max_new_tokens=15)
print(f"[C1] kv_pos={c1.kv_pos}, seq={c1.seq_id}")

# 3. Instant Branching (No decode!)
b1 = bm.create_branch(c1.node_id)
print(f"[BRANCH B1] Copied. seq={b1.seq_id}, kv_pos={b1.kv_pos}")

# 4. Hard Reset Testi
bm.checkout(c1.node_id)
bm.reset_hard(root.node_id)
print(f"[HARD RESET] Returned to Root. KV pruned at C level.")

print("\n=== FINAL TREE STATE ===")
print(bm.render_tree())
