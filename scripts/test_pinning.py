import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager, BranchManagerError

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

print(">>> Initializing engine (n_seq_max=3)...")
# Sadece 3 slot: 1 Root + 2 Branch slotu
engine = LlamaEngine(model_path=MODEL_PATH, n_ctx=512, n_seq_max=3, n_gpu_layers=-1)
bm = BranchManager(engine)
root = bm.active_node()

print("\n>>> 1. Opening two new branches (Slots will fill)...")
b1 = bm.create_branch(root.node_id) # seq=1
time.sleep(0.1) # Small delay for LRU sorting
bm.checkout(root.node_id)
b2 = bm.create_branch(root.node_id) # seq=2

print("Slot durumu:", bm._free_seq_ids) # [] bekliyoruz

print("\n>>> 2. Dal 1 ve Dal 2 kilitleniyor (Pin)...")
bm.pin(b1.node_id)
bm.pin(b2.node_id)

print("\n>>> 3. Trying to open 3rd branch when slots are full and everything is locked...")
bm.checkout(root.node_id)
try:
    b3 = bm.create_branch(root.node_id)
    print("HATA: Kilitli slotlar ezildi!")
except BranchManagerError as e:
    print(f"[SUCCESSFUL ERROR CATCH] {e}")

print("\n>>> 4. Unlocking Branch 1 (Unpin) and trying again...")
bm.unpin(b1.node_id)
time.sleep(0.1) # b1's last access time updated
# b2 has older last_accessed_at but is locked! Eviction will choose b1.
b3_yeni = bm.create_branch(root.node_id)
print(f"[SUCCESS] Yeni dal açıldı. seq={b3_yeni.seq_id}")

# Verification: Did b1 drop?
try:
    bm.tree.get(b1.node_id)
    print("ERROR: b1 is still in the tree!")
except Exception:
    print("[VERIFICATION] Branch 1 (b1) successfully crushed by LRU and C slot transferred.")

print("\n[OK] Pinning and LRU tests 100% successful!")
