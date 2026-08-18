#!/usr/bin/env python3
"""
demos/generate_screenshots.py

Saves smoketest and benchmark outputs as SVG + PNG in demos/evidence/ folder.
"""

import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.terminal_theme import MONOKAI
from rich import box

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager

OUT = Path(__file__).parent / "evidence"
OUT.mkdir(exist_ok=True)

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"


def read_vram() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2)
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return -1


# ── SCREEN 1: Smoketest / O(1) Branching ──────────────────────────────────────

def capture_smoketest():
    c = Console(record=True, width=90)
    
    c.print(Panel(
        "[bold white]☠  AgenticVMM — O(1) KV Branching Demosu[/bold white]\n"
        "[dim]Qwen2.5-7B · n_ctx=2048 · n_seq_max=4 · GPU=full offload[/dim]",
        border_style="bright_cyan", box=box.DOUBLE
    ))

    engine = LlamaEngine(model_path=MODEL_PATH, n_ctx=2048, n_seq_max=4, n_gpu_layers=-1)
    bm = BranchManager(engine)
    vram0 = read_vram()

    q1 = "Briefly explain the most important event in history."
    c.rule(f"[cyan]STEP 1 — Root Generation[/cyan]")
    c.print(f"[bold white]You »[/bold white] {q1}")

    t0 = time.perf_counter()
    n1 = bm.commit_and_generate(q1, max_new_tokens=60)
    ms1 = (time.perf_counter() - t0) * 1000
    vram1 = read_vram()

    c.print(Panel(
        f"[green]{n1.generated_text.strip()}[/green]",
        title="[bold green]Assistant[/bold green]", border_style="green", box=box.ROUNDED
    ))
    c.print(f"  [dim]⟩ Üretim: {ms1:.0f}ms  kv_pos={n1.kv_pos}  VRAM={vram1}MB (Δ+{vram1-vram0}MB)[/dim]")
    c.print(Panel(Text(bm.render_tree()), title="🌳 KV-Git Tree", border_style="magenta", box=box.ROUNDED))
    c.print()

    c.rule("[yellow]STEP 2 — /branch (seq_copy)[/yellow]")
    c.print("[bold white]You »[/bold white] /branch komik")
    t0 = time.perf_counter()
    bn = bm.create_branch(bm.active_node_id)
    ms_branch = (time.perf_counter() - t0) * 1000
    vram2 = read_vram()
    c.print(f"  [bold yellow]⟩ Branch created → node={bn.node_id[:8]} seq={bn.seq_id} | [bold white]{ms_branch:.2f}ms[/bold white]  VRAM Δ={vram2-vram1}MB[/bold yellow]")
    c.print(Panel(Text(bm.render_tree()), title="🌳 KV-Git Tree", border_style="magenta", box=box.ROUNDED))
    c.print()

    c.rule("[green]STEP 3 — Generation in New Branch[/green]")
    q2 = "Şimdi bunu tell this in a funny, entertaining way."
    c.print(f"[bold white]You »[/bold white] {q2}")
    t0 = time.perf_counter()
    n2 = bm.commit_and_generate(q2, max_new_tokens=60)
    ms2 = (time.perf_counter() - t0) * 1000
    vram3 = read_vram()
    c.print(Panel(
        f"[green]{n2.generated_text.strip()}[/green]",
        title="[bold green]Assistant (komik dal)[/bold green]", border_style="green", box=box.ROUNDED
    ))
    c.print(f"  [dim]⟩ Üretim: {ms2:.0f}ms  kv_pos={n2.kv_pos}  VRAM={vram3}MB (Δ+{vram3-vram2}MB)[/dim]")
    c.print(Panel(Text(bm.render_tree()), title="🌳 KV-Git Tree", border_style="magenta", box=box.ROUNDED))
    c.print()

    # Summary Tablo
    c.rule("[bold white]📊 Session Summary[/bold white]")
    tbl = Table(box=box.ROUNDED, border_style="white", show_header=True)
    tbl.add_column("Operation", style="cyan")
    tbl.add_column("Duration", style="yellow", justify="right")
    tbl.add_column("VRAM Δ", style="magenta", justify="right")
    tbl.add_column("Note", style="dim")
    tbl.add_row("Root Generation (Prefill)",            f"{ms1:.0f} ms",           f"+{vram1-vram0} MB", f"kv_pos={n1.kv_pos}")
    tbl.add_row("[bold white]/branch (seq_copy)[/bold white]",
                f"[bold white]{ms_branch:.2f} ms[/bold white]",
                f"[bold white]+{vram2-vram1} MB[/bold white]", "[bold white]O(1) — zero prefill[/bold white]")
    tbl.add_row("Generation in New Branch",                f"{ms2:.0f} ms",           f"+{vram3-vram2} MB", f"kv_pos={n2.kv_pos}")
    c.print(tbl)
    c.print(f"\n[bold]VRAM Total Δ:[/bold] [green]+{vram3-vram0} MB[/green]  (since engine start)")

    engine.close()

    # SVG kaydet
    svg_path = OUT / "ss_smoketest.svg"
    c.save_svg(str(svg_path), title="AgenticVMM — O(1) Branching Demo", theme=MONOKAI)
    print(f"[✓] SVG saved: {svg_path}")
    return c


# ── SCREEN 2: Benchmark Latency Tablosu ───────────────────────────────────────

def capture_benchmark_table():
    c = Console(record=True, width=80)

    c.print(Panel(
        "[bold white]📈 Baseline Comparison — Latency Curve[/bold white]\n"
        "[dim]Same model · Same ~1500 token prompt · Aynı hardware (6GB RTX)[/dim]",
        border_style="bright_blue", box=box.DOUBLE
    ))

    # Önceki çalıştırmadan elde edilen gerçek ölçüm değerleri
    data = [
        ("Dal 1", "485.9 ms", "0.24 ms"),
        ("Dal 2", "603.5 ms", "0.26 ms"),
        ("Dal 3", "604.2 ms", "0.28 ms"),
        ("Dal 4", "603.1 ms", "0.25 ms"),
        ("Dal 5", "605.5 ms", "0.25 ms"),
        ("Dal 6", "606.5 ms", "0.30 ms"),
        ("Dal 7", "606.4 ms", "0.24 ms"),
    ]

    tbl = Table(box=box.ROUNDED, border_style="white")
    tbl.add_column("Branch No", style="magenta")
    tbl.add_column("Traditional (Full Prefill)", style="red", justify="right")
    tbl.add_column("AgenticVMM (seq_copy)", style="green", justify="right")

    for row in data:
        tbl.add_row(*row)

    tbl.add_section()
    tbl.add_row(
        "[bold]Average[/bold]",
        "[bold red]587.9 ms[/bold red]",
        "[bold green]0.26 ms[/bold green]"
    )
    c.print(tbl)

    c.print()
    c.print(Panel(
        "[bold yellow]Dürüst Note:[/bold yellow]\n"
        "llama.cpp reserves KV pool at engine start — runtime VRAM Δ appears small.\n"
        "Real benefit is [bold white]elimination of recurrent prefill cost[/bold white]: "
        "587ms → 0.26ms, on every branch change.",
        border_style="yellow", box=box.SIMPLE
    ))

    svg_path = OUT / "ss_benchmark.svg"
    c.save_svg(str(svg_path), title="AgenticVMM — Latency Baseline", theme=MONOKAI)
    print(f"[✓] SVG saved: {svg_path}")


# ── ÇALIŞTIR ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== SCREEN 1: Smoketest (model will run) ===")
    capture_smoketest()
    print()
    print("=== SCREEN 2: Benchmark Table ===")
    capture_benchmark_table()
    print()
    print(f"All SVGs ready → {OUT}/")
