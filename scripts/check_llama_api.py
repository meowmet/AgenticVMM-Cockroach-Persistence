from llama_cpp import Llama
import inspect

print("=== llama-cpp-python KV API Check ===\n")

# List kv related methods in Llama class
methods = [m for m in dir(Llama) if "kv" in m.lower() or "seq" in m.lower() or "cache" in m.lower()]
print("Related methods on Llama:")
for m in sorted(methods):
    print(f"  - {m}")

print("\n--- Detay ---")
for name in ["kv_cache_seq_cp", "kv_cache_seq_rm", "kv_cache_clear", "kv_cache_seq_keep", "kv_cache_seq_add"]:
    if hasattr(Llama, name):
        print(f"[VAR]  {name}")
        try:
            print(f"       {inspect.signature(getattr(Llama, name))}")
        except:
            pass
    else:
        print(f"[YOK] {name}")
