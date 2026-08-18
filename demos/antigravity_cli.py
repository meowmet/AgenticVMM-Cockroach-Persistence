#!/usr/bin/env python3
# demos/antigravity_cli.py
"""
AgenticVMM Interactive CLI Demo (TUI)

Left Panel : "Agentic Chat" — user input and LLM output
Right Panel : "KV-Git Tree & Telemetry" — branch tree, active node, durations

Commands:
  /branch [isim]    → O(1) seq_copy to instantly create a new branch
  /checkout <id>    → Revert to an existing node_id
  /list             → List all nodes (short ID + seq_id)
  /vram             → Read current nvidia-smi VRAM (optional)
  /ss               → Save screen as HTML under demos/evidence/
  /exit             → Exit
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich import box
import questionary
from questionary import Style as QStyle

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager, BranchManagerError

logging.basicConfig(level=logging.WARNING)

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# ─── Terminal Record Buffer ────────────────────────────────────────────────────
chat_log: list[tuple[str, str]] = []        # (style, text)
telemetry_log: list[tuple[str, str]] = []   # (style, text)

console = Console(record=True)


# ─── Helpers ──────────────────────────────────────────────────────────────

def _read_vram_mb() -> int:
    """nvidia-smi ile Returns VRAM usage in MB using nvidia-smi. 0 if no GPU."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return 0


def chat_append(style: str, text: str) -> None:
    chat_log.append((style, text))
    # Keep buffer short
    if len(chat_log) > 200:
        chat_log.pop(0)


def tele_append(style: str, text: str) -> None:
    telemetry_log.append((style, text))
    if len(telemetry_log) > 100:
        telemetry_log.pop(0)


# ─── Panel Builders ───────────────────────────────────────────────────────

def build_chat_panel(width: int = 80) -> Panel:
    text = Text()
    for style, line in chat_log:
        text.append(line + "\n", style=style)
    return Panel(
        text,
        title="[bold cyan]💬 Agentic Chat[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        expand=True,
    )


def build_tree_panel(bm: BranchManager) -> Panel:
    node = bm.active_node()
    status = bm.slot_status()

    # Branch ağacı (Text olarak)
    tree_text = Text(bm.render_tree() + "\n")

    # Telemetry satırları
    tele_text = Text()
    for style, line in telemetry_log[-20:]:
        tele_text.append(line + "\n", style=style)

    # Stats tablosu
    stat_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    stat_table.add_row("[dim]Active Node[/dim]",  f"[green]{node.node_id[:8]}[/green]")
    stat_table.add_row("[dim]seq_id[/dim]",       f"[yellow]{node.seq_id}[/yellow]")
    stat_table.add_row("[dim]kv_pos[/dim]",       f"[yellow]{node.kv_pos}[/yellow]")
    stat_table.add_row("[dim]Slot[/dim]",          f"[magenta]{status['used_slots']}/{status['n_seq_max']}[/magenta]")
    stat_table.add_row("[dim]VRAM (MB)[/dim]",    f"[white]{_read_vram_mb()}[/white]")

    content = Group(tree_text, tele_text, stat_table)

    return Panel(
        content,
        title="[bold magenta]🌳 KV-Git Tree & Telemetry[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
        expand=True,
    )


# ─── Command Handlers ────────────────────────────────────────────────────────

def cmd_branch(bm: BranchManager, name: str | None) -> None:
    label = name or "branch"
    t0 = time.perf_counter()
    try:
        new_node = bm.create_branch(bm.active_node_id)
        elapsed = (time.perf_counter() - t0) * 1000
        tele_append("bold green", f"✓ Branch created [{label}] → node={new_node.node_id[:8]} seq={new_node.seq_id} | {elapsed:.2f}ms")
        chat_append("bold green", f"[System] New branch: {new_node.node_id[:8]} (seq={new_node.seq_id}) — {elapsed:.2f}ms")
    except BranchManagerError as e:
        tele_append("bold red", f"✗ Branch ERROR: {e}")
        chat_append("bold red", f"[Error] {e}")


def cmd_checkout(bm: BranchManager, node_id_prefix: str) -> None:
    # Short prefix matching
    matches = [n for n in bm.tree if n.node_id.startswith(node_id_prefix)]
    if not matches:
        chat_append("bold red", f"[Error] '{node_id_prefix}' starting node not found.")
        return
    if len(matches) > 1:
        chat_append("bold red", f"[Error] Ambiguous prefix, {len(matches)} matches. Enter a longer ID.")
        return
    node = matches[0]
    t0 = time.perf_counter()
    bm.checkout(node.node_id)
    elapsed = (time.perf_counter() - t0) * 1000
    tele_append("bold yellow", f"↩ Checkout: node={node.node_id[:8]} seq={node.seq_id} | {elapsed:.2f}ms")
    chat_append("bold yellow", f"[System] Switched branch → {node.node_id[:8]} (seq={node.seq_id}) | {elapsed:.2f}ms")


def cmd_list(bm: BranchManager) -> None:
    active_id = bm.active_node_id
    lines = []
    for n in bm.tree:
        marker = " ●" if n.node_id == active_id else "  "
        snippet = (n.prompt_text[:30] + "…") if n.prompt_text else "(root)"
        lines.append(f"{marker} {n.node_id[:8]}  seq={n.seq_id}  kv={n.kv_pos}  '{snippet}'")
    chat_append("dim", "\n".join(lines))


def cmd_screenshot(label: str = "antigravity_screenshot") -> None:
    out_dir = Path(__file__).parent / "evidence"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{label}.html"
    console.save_html(str(out_file), clear=False)
    chat_append("bold white", f"[System] Screenshot saved: {out_file}")
    tele_append("bold white", f"📸 Saved: {out_file.name}")


def cmd_generate(bm: BranchManager, user_text: str, max_tokens: int = 80) -> None:
    chat_append("bold white", f"You: {user_text}")

    t0 = time.perf_counter()
    try:
        node = bm.commit_and_generate(user_text, max_new_tokens=max_tokens)
        elapsed = (time.perf_counter() - t0) * 1000
        response = node.generated_text.strip()
        chat_append("green", f"Assistant: {response}")
        tele_append("dim", f"generate: {len(bm.engine.tokenize(user_text, add_bos=False))} tok → {elapsed:.0f}ms | kv_pos={node.kv_pos}")
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        chat_append("bold red", f"[Error] {e} ({elapsed:.0f}ms)")
        tele_append("bold red", f"✗ generate ERROR: {e}")


def cmd_vram() -> None:
    mb = _read_vram_mb()
    chat_append("dim", f"[VRAM] {mb} MB in use")
    tele_append("white", f"VRAM current: {mb} MB")


_MENU_STYLE = QStyle([
    ("selected", "fg:#00d7ff bold"),
    ("pointer",  "fg:#00d7ff bold"),
    ("highlighted", "fg:#ffffff"),
    ("answer",   "fg:#98ff98 bold"),
])


def cmd_interactive_checkout(bm: BranchManager) -> None:
    """interactive checkout menu navigated with arrow keys (questionary)."""
    nodes = list(bm.tree)
    if len(nodes) <= 1:
        chat_append("yellow", "[Menu] Only root node exists, no branches yet.")
        return

    active_id = bm.active_node_id
    choices = []
    for n in nodes:
        marker = " ● " if n.node_id == active_id else "   "
        snippet = (n.prompt_text[:40] + "…") if n.prompt_text else "(root)"
        label = f"{marker}{n.node_id[:8]}  seq={n.seq_id}  kv={n.kv_pos}  “{snippet}”"
        choices.append(questionary.Choice(title=label, value=n.node_id))

    # Rich Live screen is already frozen during input,
    # questionary writes directly to stdin/stdout — no conflict.
    try:
        selected_id = questionary.select(
            "Select the branch to checkout (↑/↓ + Enter):",
            choices=choices,
            style=_MENU_STYLE,
            use_shortcuts=False,
        ).ask()  # Returns None if canceled with Ctrl+C
    except KeyboardInterrupt:
        selected_id = None

    if selected_id is None:
        chat_append("yellow", "[Menu] Canceled.")
        return

    cmd_checkout(bm, selected_id[:8])


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run() -> None:
    console.print(Panel(
        "[bold white]AgenticVMM Demo CLI[/bold white]\n"
        "[dim]n_ctx=2048 · n_seq_max=4 · model=Qwen2.5-7B[/dim]",
        border_style="bright_cyan",
        box=box.DOUBLE,
    ))
    console.print("[dim]Loading engine...[/dim]")

    engine = LlamaEngine(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_seq_max=4,
        n_gpu_layers=-1,
    )
    bm = BranchManager(engine)

    # No root prefill; system context ready for user's first message.
    tele_append("bold cyan", "✓ Engine ready. Root node created.")
    chat_append("dim", "System başlatıldı. Commands: /branch [isim] /checkout <id> /list /vram /ss /exit")
    chat_append("dim", "You can chat with the LLM by typing any text.")

    # ── Render loop ────────────────────────────────────────────────────────
    while True:
        # Draw updated panels on each turn
        console.print()
        console.print(Panel(
            build_chat_panel().renderable,
            title="[bold cyan]💬 Agentic Chat[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print(build_tree_panel(bm))
        console.rule("[dim]Enter command (any text or /command)[/dim]")

        try:
            user_input = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting...[/yellow]")
            break

        if not user_input:
            continue

        # ── Command parsing ─────────────────────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "/exit":
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif cmd in ("/branch", "/b"):
                cmd_branch(bm, arg)

            elif cmd in ("/checkout", "/c"):
                if arg:
                    cmd_checkout(bm, arg)
                else:
                    # Open interactive menu if no ID
                    cmd_interactive_checkout(bm)

            elif cmd in ("/menu", "/m"):
                cmd_interactive_checkout(bm)

            elif cmd == "/list":
                cmd_list(bm)

            elif cmd == "/vram":
                cmd_vram()

            elif cmd in ("/ss", "/screenshot"):
                cmd_screenshot()

            elif cmd == "/help":
                chat_append("cyan",
                    "/b [name]   → create branch (alias: /branch)\n"
                    "/c [id]     → checkout (alias: /checkout) — opens menu if no ID\n"
                    "/m          → interactive branch menu (alias: /menu)\n"
                    "/list       → list all nodes\n"
                    "/vram       → read current VRAM\n"
                    "/ss         → save session to HTML\n"
                    "/exit       → exit"
                )

            else:
                chat_append("bold red", f"[Error] Unknown command: {cmd}  (type /help for help)")
        else:
            # Normal text → Send to LLM
            cmd_generate(bm, user_input)

    engine.close()


if __name__ == "__main__":
    run()
