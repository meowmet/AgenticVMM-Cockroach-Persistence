#!/usr/bin/env python3
import sys
import llama_cpp

def check_kv_api():
    print("--- llama.cpp Alt Seviye KV-Cache API Kontrolü ---")
    
    # Systemin kalbini oluşturacak o kritik C++ fonksiyonları
    critical_funcs = [
        "llama_kv_cache_seq_cp",   # Branch (Dal oluşturma / Copy-on-Write)
        "llama_kv_cache_seq_rm",   # Reset (Dalı VRAM'den silme)
        "llama_kv_cache_seq_keep", # Sadece belirli bir dalı tutup diğerlerini çöpe atma
    ]
    
    missing_funcs = []
    
    for func in critical_funcs:
        # llama_cpp kütüphanesi bu C fonksiyonlarını ctypes ile dışarı aktarmış mı?
        if hasattr(llama_cpp, func):
            print(f"[OK] {func} erişilebilir durumda.")
        else:
            print(f"[HATA] {func} BULUNAMADI!")
            missing_funcs.append(func)
            
    print("-" * 50)
    if missing_funcs:
        print("[KRİTİK] Gerekli ameliyat aletleri eksik! llama-cpp-python sürümü uyumsuz olabilir.")
        sys.exit(1)
    else:
        print("[SUCCESS] Bütün KV-Cache fonksiyonları masada! VRAM'i hacklemeye hazırız.")
        sys.exit(0)

if __name__ == "__main__":
    check_kv_api()
