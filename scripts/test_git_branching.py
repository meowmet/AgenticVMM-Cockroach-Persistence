import sys
from pathlib import Path

# src dizinini import yoluna ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

print(">>> 1. Initializing LlamaEngine & BranchManager...")
engine = LlamaEngine(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_seq_max=4,
    verbose=False,
)
mgr = BranchManager(engine)

print("\n>>> 2. First Commit on Main Branch...")
c1 = mgr.commit("Hello, are you ready for cybersecurity analysis?", "Yes, please specify targets and rules.")
print(f"[OK] Commit 1: {c1.node_id[:8]} (seq={c1.seq_id})")

print("\n>>> 3. First Branching: Scenario A (SQL Injection)...")
b_sql = mgr.create_branch(c1.node_id)
c2_sql = mgr.commit("Start SQL Injection scan.", "' OR 1=1 -- parameter tried.")
print(f"[OK] Branch SQL Commit: {c2_sql.node_id[:8]} (seq={c2_sql.seq_id})")

print("\n>>> 4. Checkout to Root Commit & Scenario B Branching (XSS)...")
mgr.checkout(c1.node_id)
b_xss = mgr.create_branch(c1.node_id)
c2_xss = mgr.commit("Start XSS scan.", "<script>alert(1)</script> injected.")
print(f"[OK] Branch XSS Commit: {c2_xss.node_id[:8]} (seq={c2_xss.seq_id})")

print("\n=== CURRENT GIT-LIKE KV TREE ===")
print(mgr.render_tree())

print("\n>>> 5. Active Branch Context (HEAD / XSS Branch):")
print("-" * 40)
print(mgr.get_active_context())
print("-" * 40)

print("\n>>> 6. Cleaning SQL Branch (Drop Subtree)...")
mgr.drop(b_sql.node_id)

print("\n=== TREE AFTER DROP ===")
print(mgr.render_tree())
print("\n[SUCCESS] All Git-KV logical operations completed successfully!")
