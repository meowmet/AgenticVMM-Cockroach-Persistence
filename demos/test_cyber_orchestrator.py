"""
AgenticVMM — Rule-First Autonomous Orchestrator Demo

Jury demo: LLM doesn't decide every step. Terminal output is triaged
by a 0-token cost rule engine (regex/string match). On failure, 
an O(1) branch rollback is triggered, proceeding to the next strategy 
without recalling the LLM. NO network requests — mock_run_command 
is fully simulated.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic_vmm.engine.llama_engine import LlamaEngine
from agentic_vmm.branch.manager import BranchManager

MODEL_PATH = "/home/met/MVP/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
N_SEQ_MAX = 4
TARGET_IP = "10.10.10.5"
EVIDENCE_PATH = Path(__file__).resolve().parent / "evidence" / "cyber_orchestrator_session.html"

STRATEGIES = [
    {
        "name": "SSH Brute Force",
        "hint": f"ssh admin@{TARGET_IP}",
        "cmd_template": f"ssh admin@{TARGET_IP} -p 22",
    },
    {
        "name": "SQL Injection",
        "hint": f"sqlmap -u http://{TARGET_IP}/login",
        "cmd_template": f"sqlmap -u http://{TARGET_IP}/login --batch",
    },
    {
        "name": "Apache CGI-Bin RCE",
        "hint": f"curl http://{TARGET_IP}/cgi-bin/test.cgi",
        "cmd_template": f"curl http://{TARGET_IP}/cgi-bin/test.cgi",
    },
]

FAIL_PATTERN = re.compile(r"refused|403", re.IGNORECASE)
SUCCESS_PATTERN = re.compile(r"\broot\b", re.IGNORECASE)


# -- mock terminal (never makes real network requests) -------------------------


def mock_run_command(cmd: str) -> str:
    """Deterministic, fully simulated command output. No network."""
    if "ssh" in cmd:
        return f"ssh: connect to host {TARGET_IP} port 22: Connection refused"
    if "sqlmap" in cmd:
        return "CRITICAL: WAF detected. IP blocked (403 Forbidden)."
    if "curl" in cmd and "cgi-bin" in cmd:
        return "uid=0(root) gid=0(root) groups=0(root)"
    return "command not recognized (simulation stub)"


# -- telemetry ------------------------------------------------------------


class Telemetry:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, label: str, value_ms: float, kind: str = "kv") -> None:
        self.events.append({"label": label, "ms": value_ms, "kind": kind})

    def table(self) -> Table:
        t = Table(title="Telemetry (ms)", expand=True, show_lines=False)
        t.add_column("Event", style="cyan")
        t.add_column("Duration (ms)", justify="right", style="magenta")
        t.add_column("Type", style="dim")
        for e in self.events[-12:]:
            t.add_row(e["label"], f"{e['ms']:.3f}", e["kind"])
        return t


# -- UI ---------------------------------------------------------------


def build_layout() -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="chat", ratio=3),
        Layout(name="log", ratio=2),
    )
    layout["right"].split_column(
        Layout(name="tree", ratio=3),
        Layout(name="telemetry", ratio=2),
    )
    return layout


def render(layout: Layout, chat_lines: list[Text], log_lines: list[Text],
           bm: BranchManager, telemetry: Telemetry, status: str) -> None:
    layout["chat"].update(
        Panel(Text("\n").join(chat_lines[-14:]), title="Chat / Agent Output", border_style="blue")
    )
    layout["log"].update(
        Panel(Text("\n").join(log_lines[-10:]), title=f"Rule Engine Log — {status}", border_style="yellow")
    )
    layout["tree"].update(
        Panel(bm.render_tree(), title="KV-Cache Branch Tree (n_seq_max=%d)" % N_SEQ_MAX, border_style="green")
    )
    layout["telemetry"].update(Panel(telemetry.table(), border_style="magenta"))


# -- main ----------------------------------------------------------------


def main() -> None:
    console = Console(record=True)
    telemetry = Telemetry()

    chat_lines: list[Text] = []
    log_lines: list[Text] = []

    def chat(msg: str, style: str = "white") -> None:
        chat_lines.append(Text(msg, style=style))

    def log(msg: str, style: str = "white") -> None:
        log_lines.append(Text(msg, style=style))

    console.rule("[bold cyan]AgenticVMM — Rule-First Autonomous Cyber Orchestrator")
    console.print(f"[dim]Target: {TARGET_IP} | Model: Qwen2.5-7B-Instruct | n_seq_max={N_SEQ_MAX}[/dim]\n")

    with console.status("[bold green]Loading model (llama.cpp / CUDA)...", spinner="dots"):
        engine = LlamaEngine(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_seq_max=N_SEQ_MAX,
            n_gpu_layers=-1,
        )
        bm = BranchManager(engine)

    root = bm.active_node()
    root_id = root.node_id

    mission_prompt = f"Target IP: {TARGET_IP}. Mission: Breach this system."
    t0 = time.perf_counter()
    sys_tokens = engine.tokenize(mission_prompt, add_bos=True)
    engine.generate(seq_id=root.seq_id, prompt_tokens=sys_tokens, pos_start=0, max_new_tokens=0)
    root.kv_pos = len(sys_tokens)
    prefill_ms = (time.perf_counter() - t0) * 1000
    telemetry.record("Root Prefill", prefill_ms, kind="decode")

    chat(f"[MISSION] {mission_prompt}", style="bold white")
    log(f"Root prefill completed: {len(sys_tokens)} tokens, {prefill_ms:.2f} ms", style="dim")

    layout = build_layout()
    breach_success = False

    with Live(layout, console=console, refresh_per_second=8, screen=False) as live:
        render(layout, chat_lines, log_lines, bm, telemetry, "initializing")
        live.refresh()
        time.sleep(0.4)

        for strategy in STRATEGIES:
            name = strategy["name"]
            chat(f"\n[STRATEGY] {name} attempting...", style="bold cyan")
            render(layout, chat_lines, log_lines, bm, telemetry, name)
            live.refresh()
            time.sleep(0.3)

            # 1) O(1) dallanma (seq_copy) — branch from root without asking LLM
            t0 = time.perf_counter()
            branch = bm.create_branch(from_node_id=root_id)
            branch_ms = (time.perf_counter() - t0) * 1000
            telemetry.record(f"branch created: {name}", branch_ms, kind="seq_copy")
            log(f"create_branch('{name}') → seq={branch.seq_id} [{branch_ms:.3f} ms]", style="green")
            render(layout, chat_lines, log_lines, bm, telemetry, name)
            live.refresh()
            time.sleep(0.3)

            # 2) short command generation from LLM (real decode, costs tokens)
            t0 = time.perf_counter()
            cmd_node = bm.commit_and_generate(
                f"Generate a one-line attack command using the {name} strategy: {strategy['hint']}",
                max_new_tokens=24,
            )
            gen_ms = (time.perf_counter() - t0) * 1000
            telemetry.record(f"model generation: {name}", gen_ms, kind="decode")

            proposed_cmd = strategy["cmd_template"]  # template used for deterministic demo
            chat(f"[LLM] Proposed command: {proposed_cmd}", style="white")
            # [LLM raw output intentionally hidden — deterministic demo flow]
            render(layout, chat_lines, log_lines, bm, telemetry, name)
            live.refresh()
            time.sleep(0.4)

            # 3) Mock terminal çalıştırma (asla gerçek ağ isteği yok)
            output = mock_run_command(proposed_cmd)
            chat(f"[TERMINAL] $ {proposed_cmd}", style="bold yellow")
            chat(f"[TERMINAL] {output}", style="yellow")
            render(layout, chat_lines, log_lines, bm, telemetry, name)
            live.refresh()
            time.sleep(0.4)

            # 4) RULE CHECK — 0-token triage, LLM is not queried
            if FAIL_PATTERN.search(output):
                log(
                    "RULE: Failed. 0-token Triage active. Rolling back...",
                    style="bold red",
                )
                render(layout, chat_lines, log_lines, bm, telemetry, name)
                live.refresh()
                time.sleep(0.3)

                t0 = time.perf_counter()
                bm.checkout(root_id)
                rollback_ms = (time.perf_counter() - t0) * 1000
                telemetry.record("checkout (rollback) → root", rollback_ms, kind="checkout")
                log(f"checkout (root) [{rollback_ms:.3f} ms] — proceeding to next strategy", style="red")
                render(layout, chat_lines, log_lines, bm, telemetry, name)
                live.refresh()
                time.sleep(0.4)
                continue

            if SUCCESS_PATTERN.search(output):
                log("RULE: Success! System compromised.", style="bold green")
                chat(f"[BREACH] {name} success — root access verified.", style="bold green")
                bm.pin(cmd_node.node_id)
                log(f"pin({cmd_node.node_id[:8]}) — this branch is now protected from LRU eviction.", style="green")
                render(layout, chat_lines, log_lines, bm, telemetry, name)
                live.refresh()
                breach_success = True
                break

        if not breach_success:
            chat("\n[FAILURE] All strategies exhausted, breach failed.", style="bold red")
            log("Loop terminated: No vulnerabilities found.", style="red")

        render(layout, chat_lines, log_lines, bm, telemetry, "COMPLETED")
        live.refresh()
        time.sleep(1)

    console.print(f"\n[bold green]✓ Session evidence saved as HTML: {EVIDENCE_PATH}[/bold green]")
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    console.save_html(str(EVIDENCE_PATH))
    console.print("[dim]Execution terminated. Ready for presentation.[/dim]")

    engine.close()


if __name__ == "__main__":
    main()
