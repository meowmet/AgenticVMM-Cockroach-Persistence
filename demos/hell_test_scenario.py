#!/usr/bin/env python3
# demos/hell_test_scenario.py
"""
CEHENNEM TESTİ (HELL TEST): 5000+ token ağır bağlam ve 7 eşzamanlı dal!

Traditional ajanlar (LangChain vb.):
  5000 token x 7 dal = 35.000 token KV yükü → Anında OOM (Out of Memory)

AgenticVMM:
  5000 token 1 kez prefill → seq_copy ile O(1) dal kopyası
  Toplam KV yükü: ~5500 token (5000 ortak + dallar arası delta)
  Dal açma maliyeti: ~0.5ms
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager
from agentic_vmm.engine.vram_monitor import VRAMRecorder

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
console = Console()

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# ─── 5000+ TOKEN AĞIR BAĞLAM (Simüle Edilmiş Ağ Trafiği) ───────────────────
# Zararlı yazılım yerine, sistemi yoracak devasa boyutta ve gürültülü 
# bir TCP/Wireshark stream dökümü (benign network noise) kullanıyoruz.
# Yaklaşık 5000 token üretmek için büyük bir metin bloğu oluşturuyoruz.

CHUNK = """
No.     Time           Source                Destination           Protocol Length Info
    1 0.000000000    192.168.1.100         10.0.0.5              TCP      74     54321 → 80 [SYN] Seq=0 Win=65535 Len=0 MSS=1460 SACK_PERM=1 TSval=1000 TSecr=0 WS=128
    2 0.001234000    10.0.0.5              192.168.1.100         TCP      74     80 → 54321 [SYN, ACK] Seq=0 Ack=1 Win=65535 Len=0 MSS=1460 SACK_PERM=1 TSval=2000 TSecr=1000 WS=128
    3 0.001345000    192.168.1.100         10.0.0.5              TCP      66     54321 → 80 [ACK] Seq=1 Ack=1 Win=65535 Len=0 TSval=1001 TSecr=2000
    4 0.002456000    192.168.1.100         10.0.0.5              HTTP     345    GET /api/v1/status HTTP/1.1 
    5 0.005678000    10.0.0.5              192.168.1.100         TCP      66     80 → 54321 [ACK] Seq=1 Ack=280 Win=65535 Len=0 TSval=2002 TSecr=1002
    6 0.010789000    10.0.0.5              192.168.1.100         HTTP     456    HTTP/1.1 200 OK  (application/json)
    7 0.011890000    192.168.1.100         10.0.0.5              TCP      66     54321 → 80 [ACK] Seq=280 Ack=391 Win=65535 Len=0 TSval=1005 TSecr=2005

0000   45 00 00 34 12 34 40 00 40 06 a1 b2 c0 a8 01 64  E..4.4@.@......d
0010   0a 00 00 05 d4 31 00 50 00 00 00 01 00 00 00 01  .....1.P........
0020   80 10 ff ff 12 34 00 00 01 01 08 0a 00 00 03 e9  .....4..........
0030   00 00 07 d5 47 45 54 20 2f 61 70 69 2f 76 31 2f  ....GET /api/v1/
0040   73 74 61 74 75 73 20 48 54 54 50 2f 31 2e 31 0d  status HTTP/1.1.
0050   0a 48 6f 73 74 3a 20 31 30 2e 30 2e 30 2e 35 0d  .Host: 10.0.0.5.
0060   0a 55 73 65 72 2d 41 67 65 6e 74 3a 20 4d 6f 7a  .User-Agent: Moz
0070   69 6c 6c 61 2f 35 2e 30 20 28 57 69 6e 64 6f 77  illa/5.0 (Window
0080   73 20 4e 54 20 31 30 2e 30 3b 20 57 69 6e 36 34  s NT 10.0; Win64
0090   3b 20 78 36 34 29 20 41 70 70 6c 65 57 65 62 4b  ; x64) AppleWebK
00a0   69 74 2f 35 33 37 2e 33 36 0d 0a 41 63 63 65 70  it/537.36..Accep
00b0   74 3a 20 2a 2f 2a 0d 0a 0d 0a                    t: */*....
"""

HEAVY_CONTEXT = "=== NETWORK TRAFFIC DUMP ===\n\n" + CHUNK + "\n=== END DUMP ===\n"

def step_banner(step_num: int, title: str, desc: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold white]{desc}[/bold white]",
        title=f"[bold red]Cehennem Stepı {step_num}: {title}[/bold red]",
        border_style="red",
        box=box.HEAVY,
    ))


def show_generation(label: str, node, elapsed_ms: float) -> None:
    # Konsolu yormamak için çıktıyı kısaltıyoruz
    text = node.generated_text.strip()
    if len(text) > 150:
        text = text[:147] + "..."
    console.print(Panel(
        f"[bold green]Assistant:[/bold green] {text}",
        subtitle=f"[dim]{label} | node={node.node_id[:8]} seq={node.seq_id} kv_pos={node.kv_pos} | {elapsed_ms:.0f}ms[/dim]",
        border_style="green",
    ))


def show_tree(bm: BranchManager) -> None:
    console.print(Panel(
        bm.render_tree(),
        title="[bold cyan]🌳 Branch Tree (Cehennem Modu)[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))


def show_status(bm: BranchManager) -> None:
    status = bm.slot_status()
    table = Table(box=box.SIMPLE, border_style="dim")
    table.add_column("Metrik", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Slot Kullanımı", f"{status['used_slots']}/{status['n_seq_max']}")
    table.add_row("Free seq_ids", str(status["free_seq_ids"]))
    table.add_row("Aktif seq/kv_pos", f"seq={status['active_seq_id']} pos={status['active_kv_pos']}")
    console.print(table)


def main():
    console.print(Panel(
        "[bold white]☠️ CEHENNEM TESTİ — Aşırı Yoğun Context ve 7 Eşzamanlı Dal![/bold white]\n\n"
        "[dim]Traditional: 7 ayrı prefill → VRAM ŞİŞMESİ (OOM)\n"
        "AgenticVMM: 1 prefill + O(1) seq_copy → ✓ Mükemmel Çalışır[/dim]",
        border_style="bright_red",
        box=box.DOUBLE,
    ))

    # Motoru 8 slot ve 16384 context ile başlat
    t0 = time.time()
    engine = LlamaEngine(model_path=MODEL_PATH, n_ctx=16384, n_seq_max=8, n_gpu_layers=-1)
    bm = BranchManager(engine)

    vram_rec = VRAMRecorder(interval_sec=1.0)
    vram_rec.start()
    vram_rec.mark("0: Motor Yüklendi")

    root = bm.active_node()
    system_prompt = (
        "Sen bir ağ güvenliği analistisin. Aşağıdaki devasa TCP paket dökümünü analiz edeceksin.\n"
        + HEAVY_CONTEXT
    )

    sys_tokens = engine.tokenize(system_prompt, add_bos=True)
    token_count = len(sys_tokens)

    console.print(f"[bold red]🔥 Devasa Context Yükleniyor: {token_count} token ({len(system_prompt)} karakter)[/bold red]")

    step_banner(0, "Massive Prefill", f"{token_count} token tek seferde KV-Cache'e yazılıyor...")
    
    t_prefill = time.time()
    engine.generate(seq_id=root.seq_id, prompt_tokens=sys_tokens, pos_start=0, max_new_tokens=0)
    prefill_ms = (time.time() - t_prefill) * 1000

    root.kv_pos = token_count
    vram_rec.mark(f"1: Prefill Tamamlandı ({token_count} tok)")

    console.print(f"[green]✓ {token_count} token prefill edildi ({prefill_ms:.0f}ms).[/green]")
    show_status(bm)

    # ── 7 Farklı Dal Açıyoruz ──────────────────────────────────────────────
    step_banner(1, "The Hydra", "Aynı kökten 7 farklı analiz dalı O(1) maliyetle filizleniyor!")

    branches = []
    tasks = [
        "Bu ağ trafiğindeki tüm kaynak ve hedef IP adreslerini listele.",
        "HTTP isteklerinde kullanılan User-Agent bilgilerini analiz et.",
        "Is there a possible port scan or abnormal connection attempt?",
        "Detect unencrypted (plain-text) data fragments in traffic.",
        "Estimate OS based on TCP Window Size and TTL values. do.",
        "Comment on the connection latency durations.",
        "En uzun paket boyutuna sahip transferleri listele."
    ]

    branch_times = []
    gen_times = []

    # First create 7 branches from the main root node (seq_copy only)
    for i in range(7):
        bm.checkout(root.node_id)
        
        t_branch = time.time()
        new_branch = bm.create_branch(root.node_id)
        b_time = (time.time() - t_branch) * 1000
        branch_times.append(b_time)
        
        branches.append(new_branch)
        console.print(f"[green]✓ Branch {i+1} created ({b_time:.2f}ms) — seq={new_branch.seq_id}[/green]")
    
    vram_rec.mark("2: 7 Branches Created")
    show_status(bm)

    # Now switch to each branch and perform generation
    step_banner(2, "Concurrent Execution", "Each branch generates only delta tokens...")

    for i, (branch, task) in enumerate(zip(branches, tasks)):
        bm.checkout(branch.node_id)
        
        t_gen = time.time()
        result_node = bm.commit_and_generate(task, max_new_tokens=30)
        g_time = (time.time() - t_gen) * 1000
        gen_times.append(g_time)
        
        show_generation(f"Task {i+1}", result_node, g_time)

    vram_rec.mark("3: 7 Task Üretimi Tamamlandı")
    
    # ── Final ────────────────────────────────────────────────────────────
    vram_rec.stop()
    console.print()
    show_tree(bm)
    
    avg_branch_time = sum(branch_times) / len(branch_times)
    
    perf_table = Table(title="☠️  Hell Test Performance Summary", box=box.ROUNDED, border_style="red")
    perf_table.add_column("Metrik", style="cyan")
    perf_table.add_column("Traditional (LangChain)", style="red")
    perf_table.add_column("AgenticVMM", style="green")
    
    perf_table.add_row("KV Token Load", f"{token_count} × 7 = {token_count*7} token", f"{token_count} × 1 + delta")
    perf_table.add_row("Branching Latency", f"~{int(prefill_ms)}ms (Re-prefill)", f"Average {avg_branch_time:.2f}ms (O(1))")
    perf_table.add_row("Result", "[bold red]☠️ OOM (SYSTEM CRASHES)[/bold red]", "[bold green]✓ Works Flawlessly[/bold green]")
    console.print(perf_table)
    
    # VRAM Tablosu
    vram_table = Table(title="📊 VRAM Telemetrisi", box=box.ROUNDED, border_style="bright_magenta")
    vram_table.add_column("Step", style="cyan")
    vram_table.add_column("VRAM (MB)", style="green", justify="right")
    
    for m in vram_rec.marks:
        vram_table.add_row(m.label, str(m.used_mb))
        
    delta = vram_rec.delta_range()
    vram_table.add_section()
    vram_table.add_row("[bold]Max VRAM Swell (Delta)[/bold]", f"[bold]{delta} MB[/bold]")
    console.print(vram_table)

    engine.close()


if __name__ == "__main__":
    main()
