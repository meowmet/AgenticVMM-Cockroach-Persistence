# src/agentic_vmm/cli/app.py
"""
Interactive CLI REPL for the Agentic VMM branching system.
Uses rich for polished terminal output. 

Commands:
  /tree           - Display the branch tree with HEAD marker
  /branch [name]  - Fork a new branch from the active node
  /checkout <id>  - Switch HEAD to an existing node (prefix match)
  /reset <id>     - Hard reset to a node (in-place KV prune, no new seq)
  /drop <id>      - Drop a branch subtree and free its C-level sequence
  /status         - Show VRAM slot usage and active context info
  /context        - Show the conversation history for the active branch
  /help           - Show available commands
  /quit           - Exit the REPL

Any other input is treated as a user message and triggers
commit_and_generate() on the active branch.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager, BranchManagerError
from agentic_vmm.engine.vram_monitor import vram_snapshot, VRAMRecorder

logger = logging.getLogger(__name__)
console = Console()


def _find_node_by_prefix(bm: BranchManager, prefix: str) -> str | None:
    """Find a node_id by its prefix (at least 4 chars)."""
    prefix = prefix.strip()
    matches = [n.node_id for n in bm.tree if n.node_id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Multiple matches: {[m[:8] for m in matches]}. Enter a longer prefix.[/yellow]")
        return None
    console.print(f"[red]'{prefix}' matching node not found.[/red]")
    return None


def _print_tree(bm: BranchManager) -> None:
    tree_text = bm.render_tree()
    console.print(Panel(
        tree_text,
        title="[bold cyan]🌳 Branch Tree[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))


def _print_status(bm: BranchManager) -> None:
    status = bm.slot_status()
    table = Table(
        title="⚡ VRAM Sequence Slot Durumu",
        box=box.ROUNDED,
        border_style="magenta",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metrik", style="cyan", width=22)
    table.add_column("Value", style="green")

    table.add_row("n_seq_max", str(status["n_seq_max"]))
    table.add_row("Used Slots", f"{status['used_slots']} / {status['n_seq_max']}")
    table.add_row("Free Slots", str(status["free_slots"]))
    table.add_row("Used seq_ids", str(status["used_seq_ids"]))
    table.add_row("Free seq_ids", str(status["free_seq_ids"]))
    table.add_row("Aktif seq_id", str(status["active_seq_id"]))
    table.add_row("Aktif kv_pos", str(status["active_kv_pos"]))
    table.add_row("Toplam Node", str(status["total_nodes"]))

    console.print(table)


def _print_vram(label: str = "") -> None:
    snap = vram_snapshot(label)
    if snap is None:
        console.print("[red]nvidia-smi could not be accessed.[/red]")
        return

    bar_width = 30
    filled = int(snap.used_pct / 100 * bar_width)
    empty = bar_width - filled

    if snap.used_pct < 60:
        color = "green"
    elif snap.used_pct < 85:
        color = "yellow"
    else:
        color = "red"

    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
    console.print(Panel(
        f"  VRAM  [{bar}] {snap.used_mb}/{snap.total_mb} MB ({snap.used_pct:.1f}%)\n"
        f"  GPU   {snap.gpu_util_pct}% utilization\n"
        f"  Free  {snap.free_mb} MB"
        f"{'  📌 ' + label if label else ''}",
        title="[bold magenta]⚡ GPU VRAM[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    ))


def _print_help() -> None:
    help_text = """[bold cyan]Commands:[/bold cyan]
  [green]/tree[/green]              Show branch tree
  [green]/branch[/green]            Create new branch from active node
  [green]/checkout <id>[/green]     Switch to another node (with prefix)
  [green]/reset <id>[/green]        Yerinde geri sar (hard reset, slot harcamaz)
  [green]/drop <id>[/green]         Delete a branch and its subtree
  [green]/status[/green]            Show VRAM slot status
  [green]/vram[/green]              Show current GPU VRAM usage
  [green]/context[/green]           Show conversation history of active branch
  [green]/help[/green]              Show this help
  [green]/quit[/green]              Exit

  [dim]All other inputs are processed as user messages and
  commit_and_generate() is called on the active branch.[/dim]"""
    console.print(Panel(help_text, title="[bold]Help[/bold]", border_style="blue"))


def run_repl(
    model_path: str,
    n_ctx: int = 2048,
    n_seq_max: int = 4,
    n_gpu_layers: int = -1,
    system_prompt: str = "You are a cybersecurity expert assistant.",
    max_new_tokens: int = 80,
) -> None:
    """Launch the interactive REPL."""

    console.print(Panel(
        "[bold white]Agentic VMM — Git-like KV-Cache Branching REPL[/bold white]\n"
        "[dim]Tip /help for available commands[/dim]",
        border_style="bright_green",
        box=box.DOUBLE,
    ))

    console.print("[dim]Loading model...[/dim]")
    engine = LlamaEngine(
        model_path=model_path,
        n_ctx=n_ctx,
        n_seq_max=n_seq_max,
        n_gpu_layers=n_gpu_layers,
    )

    bm = BranchManager(engine)

    # System prompt seed
    root = bm.active_node()
    sys_tokens = engine.tokenize(system_prompt, add_bos=True)
    engine.generate(seq_id=root.seq_id, prompt_tokens=sys_tokens, pos_start=0, max_new_tokens=0)
    root.kv_pos = len(sys_tokens)

    console.print(f"[green]✓ Model loaded. n_seq_max={n_seq_max}, system prompt ({len(sys_tokens)} token) seeded.[/green]")
    _print_help()

    while True:
        try:
            user_input = console.input("\n[bold cyan]vmm>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exit...[/dim]")
            break

        if not user_input:
            continue

        # -- command dispatch --
        if user_input == "/quit" or user_input == "/exit":
            break

        elif user_input == "/help":
            _print_help()

        elif user_input == "/tree":
            _print_tree(bm)

        elif user_input == "/status":
            _print_status(bm)

        elif user_input == "/vram":
            _print_vram()

        elif user_input == "/context":
            ctx = bm.get_active_context()
            if ctx:
                console.print(Panel(ctx, title="[bold]Active Branch Context[/bold]", border_style="yellow"))
            else:
                console.print("[dim]No conversation yet.[/dim]")

        elif user_input.startswith("/branch"):
            try:
                branch_node = bm.create_branch(bm.active_node_id)
                console.print(
                    f"[green]✓ Yeni dal created:[/green] "
                    f"node={branch_node.node_id[:8]}, seq={branch_node.seq_id}, "
                    f"kv_pos={branch_node.kv_pos}"
                )
            except BranchManagerError as e:
                console.print(f"[red]✗ Branch error: {e}[/red]")

        elif user_input.startswith("/checkout"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]Usage: /checkout <node_id_prefix>[/red]")
                continue
            node_id = _find_node_by_prefix(bm, parts[1])
            if node_id:
                node = bm.checkout(node_id)
                console.print(
                    f"[green]✓ HEAD moved:[/green] node={node.node_id[:8]}, "
                    f"seq={node.seq_id}, kv_pos={node.kv_pos}"
                )

        elif user_input.startswith("/reset"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]Usage: /reset <node_id_prefix>[/red]")
                continue
            node_id = _find_node_by_prefix(bm, parts[1])
            if node_id:
                try:
                    node = bm.reset_hard(node_id)
                    console.print(
                        f"[green]✓ Hard reset completed:[/green] node={node.node_id[:8]}, "
                        f"seq={node.seq_id}, kv_pos={node.kv_pos}"
                    )
                except BranchManagerError as e:
                    console.print(f"[red]✗ Reset error: {e}[/red]")

        elif user_input.startswith("/drop"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]Usage: /drop <node_id_prefix>[/red]")
                continue
            node_id = _find_node_by_prefix(bm, parts[1])
            if node_id:
                try:
                    bm.drop(node_id)
                    console.print(f"[green]✓ Branch silindi: {node_id[:8]}[/green]")
                except BranchManagerError as e:
                    console.print(f"[red]✗ Drop error: {e}[/red]")

        else:
            # Regular user message -> commit_and_generate
            try:
                console.print("[dim]Generating...[/dim]")
                node = bm.commit_and_generate(user_input, max_new_tokens=max_new_tokens)
                console.print(Panel(
                    f"[bold green]Assistant:[/bold green] {node.generated_text}",
                    subtitle=f"[dim]node={node.node_id[:8]} seq={node.seq_id} kv_pos={node.kv_pos}[/dim]",
                    border_style="green",
                ))
            except Exception as e:
                console.print(f"[red]✗ Generation error: {e}[/red]")

    engine.close()
    console.print("[green]Engine shut down. Goodbye![/green]")


def main():
    """Entry point for CLI."""
    import argparse
    parser = argparse.ArgumentParser(description="Agentic VMM Interactive REPL")
    parser.add_argument("--model", type=str,
                        default="/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                        help="Path to GGUF model")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    parser.add_argument("--n-seq-max", type=int, default=4, help="Max KV sequences")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens per turn")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a cybersecurity expert assistant.",
                        help="System prompt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_repl(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_seq_max=args.n_seq_max,
        n_gpu_layers=args.n_gpu_layers,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
