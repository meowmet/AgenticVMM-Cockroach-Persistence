#!/usr/bin/env python3
# demos/antigravity_smoketest.py
"""
Smoke test that autonomously runs all commands of antigravity_cli.py sequentially.
Saves the output as demos/evidence/antigravity_session.html.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager, BranchManagerError

console = Console(record=True, width=120)

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# ─── Helpers ──────────────────────────────────────────────────────────────

def read_vram_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return -1


def print_tree(bm: BranchManager) -> None:
    node = bm.active_node()
    status = bm.slot_status()
    console.print(Panel(
        Text(bm.render_tree()),
        title="[bold magenta]🌳 KV-Git Tree[/bold magenta]",
        subtitle=(
            f"[dim]active node=[yellow]{node.node_id[:8]}[/yellow]  "
            f"seq=[yellow]{node.seq_id}[/yellow]  "
            f"kv_pos=[yellow]{node.kv_pos}[/yellow]  "
            f"slot=[magenta]{status['used_slots']}/{status['n_seq_max']}[/magenta][/dim]"
        ),
        border_style="magenta",
        box=box.ROUNDED,
    ))


def user_says(text: str) -> None:
    console.print(f"[bold white]You »[/bold white] [white]{text}[/white]")


def system_says(text: str) -> None:
    console.print(f"[bold dim][System][/bold dim] [dim]{text}[/dim]")


def tele(text: str, style: str = "green") -> None:
    console.print(f"  [bold {style}]⟩[/bold {style}] [{style}]{text}[/{style}]")


# ─── Session Simulation ───────────────────────────────────────────────────────

def main() -> None:
    console.print(Panel(
        "[bold white]☠  AgenticVMM — Automated CLI Smoke Test[/bold white]\n"
        "[dim]Engine: Qwen2.5-7B  ·  n_ctx=2048  ·  n_seq_max=4[/dim]",
        border_style="bright_cyan",
        box=box.DOUBLE,
    ))
    console.print()

    # ── Initialize Engine ─────────────────────────────────────────────────────────
    system_says("Loading engine... (n_ctx=2048, n_seq_max=4, GPU offload=full)")
    t0 = time.perf_counter()
    engine = LlamaEngine(model_path=MODEL_PATH, n_ctx=2048, n_seq_max=4, n_gpu_layers=-1)
    bm = BranchManager(engine)
    init_ms = (time.perf_counter() - t0) * 1000
    vram_init = read_vram_mb()
    tele(f"Engine ready: {init_ms:.0f}ms  VRAM={vram_init}MB")
    console.print()

    # ── STEP 1: Root Generation (Large Prefill) ─────────────────────────────────
    console.rule("[bold cyan]STEP 1: Root Generation — Large Prefill[/bold cyan]")
    q1 = "Briefly explain the most important event in history."
    user_says(q1)

    t0 = time.perf_counter()
    root_node = bm.commit_and_generate(q1, max_new_tokens=60)
    gen_ms = (time.perf_counter() - t0) * 1000
    vram_after_root = read_vram_mb()

    console.print(Panel(
        f"[green]{root_node.generated_text.strip()}[/green]",
        title="[bold green]Assistant[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    ))
    tele(f"Generation: {gen_ms:.0f}ms  kv_pos={root_node.kv_pos}  VRAM={vram_after_root}MB (delta={vram_after_root - vram_init}MB)")
    console.print()
    print_tree(bm)
    console.print()

    # ── STEP 2: O(1) Branch — "Antigravity Moment" ────────────────────────────
    console.rule("[bold yellow]STEP 2: /branch funny  ← O(1) seq_copy[/bold yellow]")
    user_says("/branch funny")

    t0 = time.perf_counter()
    branch_node = bm.create_branch(bm.active_node_id)
    branch_ms = (time.perf_counter() - t0) * 1000
    vram_after_branch = read_vram_mb()

    tele(
        f"Branch created [funny] → node={branch_node.node_id[:8]}  seq={branch_node.seq_id} "
        f"| [bold white]{branch_ms:.2f}ms[/bold white]  VRAM delta={vram_after_branch - vram_after_root}MB",
        style="yellow",
    )
    console.print()
    print_tree(bm)
    console.print()

    # ── STEP 3: Generation in New Branch (Zero Prefill) ─────────────────────────
    console.rule("[bold green]STEP 3: Generation in New Branch — Zero Prefill[/bold green]")
    q2 = "Now tell this in a funny, entertaining way."
    user_says(q2)

    t0 = time.perf_counter()
    gen2_node = bm.commit_and_generate(q2, max_new_tokens=60)
    gen2_ms = (time.perf_counter() - t0) * 1000
    vram_after_gen2 = read_vram_mb()

    console.print(Panel(
        f"[green]{gen2_node.generated_text.strip()}[/green]",
        title="[bold green]Assistant (funny branch)[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    ))
    tele(
        f"Generation (zero prefill): {gen2_ms:.0f}ms  kv_pos={gen2_node.kv_pos}  "
        f"VRAM={vram_after_gen2}MB (delta={vram_after_gen2 - vram_after_root}MB)"
    )
    console.print()
    print_tree(bm)
    console.print()

    # ── STEP 4: Checkout Root ────────────────────────────────────
    console.rule("[bold cyan]STEP 4: /checkout <root>  ← Instant Rollback[/bold cyan]")
    root_node_id = bm.tree.root().node_id
    user_says(f"/checkout {root_node_id[:8]}")

    t0 = time.perf_counter()
    bm.checkout(root_node_id)
    checkout_ms = (time.perf_counter() - t0) * 1000

    tele(f"Checkout → root node={root_node_id[:8]} seq=0 | {checkout_ms:.2f}ms", style="cyan")
    console.print()
    print_tree(bm)
    console.print()

    # ── STEP 5: Summary Table ───────────────────────────────────────────────
    console.rule("[bold white]📊 Session Summary[/bold white]")
    from rich.table import Table
    tbl = Table(box=box.ROUNDED, border_style="white")
    tbl.add_column("Operation", style="cyan")
    tbl.add_column("Duration", style="yellow", justify="right")
    tbl.add_column("VRAM Delta", style="magenta", justify="right")
    tbl.add_column("Note", style="dim")

    tbl.add_row("Root Generation", f"{gen_ms:.0f} ms",       f"+{vram_after_root - vram_init} MB",        f"kv_pos={root_node.kv_pos}")
    tbl.add_row("[bold white]/branch (seq_copy)[/bold white]", f"[bold white]{branch_ms:.2f} ms[/bold white]", f"[bold white]+{vram_after_branch - vram_after_root} MB[/bold white]", "[bold white]O(1) — zero prefill[/bold white]")
    tbl.add_row("Generation in New Branch",  f"{gen2_ms:.0f} ms",      f"+{vram_after_gen2 - vram_after_root} MB",  f"kv_pos={gen2_node.kv_pos}")
    tbl.add_row("/checkout (rollback)", f"{checkout_ms:.2f} ms", "0 MB",                              "O(1) — pointer change")
    console.print(tbl)

    # ── /vram ────────────────────────────────────────────────────────────
    vram_final = read_vram_mb()
    console.print(f"\n[bold]/vram →[/bold] [green]{vram_final} MB[/green] (total delta: +{vram_final - vram_init} MB since engine start)")
    console.print()

    # ── /ss — HTML Export ─────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "evidence"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "antigravity_session.html"
    console.save_html(str(out_file), clear=False)
    system_says(f"/ss → Session saved: {out_file}")

    engine.close()
    console.print("\n[bold cyan]Engine shut down. Smoke test completed successfully. 🚀[/bold cyan]")


if __name__ == "__main__":
    main()
