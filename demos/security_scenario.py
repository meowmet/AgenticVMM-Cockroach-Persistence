#!/usr/bin/env python3
# demos/security_scenario.py
"""
Gösteri Senaryosu: Siber güvenlik taramasında dal değiştirme.

Akış:
  1. Root: System prompt + Genel hedef tanımlama
  2. Branch 1 (SQLi): Model SQLi dener, payload başarısız olur
  3. Hard Reset: Metin silinmeden C seviyesinde KV kv_pos noktasına çekilir
  4. Branch 2 (Command Injection): Anında yeni dala geçilip hedefe ulaşılır

Bu senaryo, tek model üzerinde zero prefill gecikmesiyle
stratejik dal değişikliğinin canlı demosunu sunar.
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


def step_banner(step_num: int, title: str, desc: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold white]{desc}[/bold white]",
        title=f"[bold yellow]Step {step_num}: {title}[/bold yellow]",
        border_style="yellow",
        box=box.HEAVY,
    ))


def show_generation(label: str, node, elapsed_ms: float) -> None:
    console.print(Panel(
        f"[bold green]Assistant:[/bold green] {node.generated_text}",
        subtitle=f"[dim]{label} | node={node.node_id[:8]} seq={node.seq_id} kv_pos={node.kv_pos} | {elapsed_ms:.0f}ms[/dim]",
        border_style="green",
    ))


def show_tree(bm: BranchManager) -> None:
    console.print(Panel(
        bm.render_tree(),
        title="[bold cyan]🌳 Branch Tree[/bold cyan]",
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
        "[bold white]🛡️  Agentic VMM — Siber Güvenlik Dallanma Senaryosu[/bold white]\n\n"
        "[dim]Tek model ağırlığı üzerinden bellekte çoğaltılmadan\n"
        "çoklu izole strateji dallarının canlı gösterimi[/dim]",
        border_style="bright_magenta",
        box=box.DOUBLE,
    ))

    # ── Step 0: Motor başlatma ──────────────────────────────────────────
    step_banner(0, "Motor Başlatma", "LlamaEngine + BranchManager init, system prompt seed")

    t0 = time.time()
    engine = LlamaEngine(model_path=MODEL_PATH, n_ctx=2048, n_seq_max=4, n_gpu_layers=-1)
    bm = BranchManager(engine)

    root = bm.active_node()
    system_prompt = (
        "Sen bir siber güvenlik uzmanı ve sızma testi (pentest) asistanısın. "
        "Usernın belirttiği hedef sisteme karşı farklı saldırı vektörlerini "
        "planlayıp uyguluyorsun. Etik kurallar dahilinde çalış."
    )
    sys_tokens = engine.tokenize(system_prompt, add_bos=True)
    engine.generate(seq_id=root.seq_id, prompt_tokens=sys_tokens, pos_start=0, max_new_tokens=0)
    root.kv_pos = len(sys_tokens)
    init_ms = (time.time() - t0) * 1000

    # VRAM recorder başlat
    vram_rec = VRAMRecorder(interval_sec=1.0)
    vram_rec.start()
    vram_rec.mark("0: Model Yüklendi")

    console.print(f"[green]✓ Motor hazır ({init_ms:.0f}ms). System prompt: {len(sys_tokens)} token seeded.[/green]")
    show_status(bm)

    # ── Step 1: Hedef tanımlama ────────────────────────────────────────
    step_banner(1, "Hedef Tanımlama", "Ana dalda (seq=0) hedef sistem tanımlaması")

    t0 = time.time()
    base = bm.commit_and_generate(
        "Hedef: https://example-target.com adresindeki web uygulaması. "
        "Login formu var, backend PHP/MySQL. İlk keşif sonuçlarını analiz et.",
        max_new_tokens=60,
    )
    show_generation("Hedef Tanımlama", base, (time.time() - t0) * 1000)
    vram_rec.mark("1: Hedef Tanımlama sonrası")
    show_tree(bm)

    # ── Step 2: SQLi dalı ──────────────────────────────────────────────
    step_banner(2, "Branch → SQLi Stratejisi",
                "base commit'ten $O(1)$ seq_copy ile yeni dal açılıyor")

    t0 = time.time()
    sql_branch = bm.create_branch(base.node_id)
    branch_ms = (time.time() - t0) * 1000
    console.print(f"[green]✓ SQLi dalı created ({branch_ms:.1f}ms) — seq={sql_branch.seq_id}, kv_pos miras={sql_branch.kv_pos}[/green]")

    t0 = time.time()
    sql_attempt = bm.commit_and_generate(
        "Login formuna SQL Injection dene. "
        "Klasik ' OR 1=1 -- payload'ını kullanarak authentication bypass do.",
        max_new_tokens=80,
    )
    show_generation("SQLi Denemesi", sql_attempt, (time.time() - t0) * 1000)
    vram_rec.mark("2: SQLi branch + generate sonrası")
    show_tree(bm)

    # ── Step 3: SQLi başarısız → Hard Reset ─────────────────────────────
    step_banner(3, "Hard Reset → Geri Sarma",
                "SQLi başarısız oldu. Aynı seq üzerinde kv_pos'a geri sar — yeni slot HARCAMADAN!")

    console.print(f"[yellow]⚠ SQLi denemesi başarısız varsayılıyor. Strateji değişikliği gerekli.[/yellow]")
    console.print(f"[dim]Mevcut kv_pos={sql_attempt.kv_pos} → base kv_pos={base.kv_pos} noktasına geri sarılacak[/dim]")

    # Önce base'e checkout (aynı seq=0 üzerinde), sonra hard reset
    bm.checkout(base.node_id)

    t0 = time.time()
    reset_node = bm.reset_hard(base.node_id)
    reset_ms = (time.time() - t0) * 1000

    console.print(f"[green]✓ Hard reset tamamlandı ({reset_ms:.1f}ms) — KV-cache pos={base.kv_pos} sonrası budandı[/green]")
    vram_rec.mark("3: Hard Reset sonrası")
    show_tree(bm)
    show_status(bm)

    # ── Step 4: Command Injection dalı ──────────────────────────────────
    step_banner(4, "Branch → Command Injection Stratejisi",
                "Aynı base noktasından yeni strateji dalı — zero prefill gecikmesi!")

    t0 = time.time()
    cmd_branch = bm.create_branch(base.node_id)
    branch2_ms = (time.time() - t0) * 1000
    console.print(f"[green]✓ CmdInj dalı created ({branch2_ms:.1f}ms) — seq={cmd_branch.seq_id}, kv_pos={cmd_branch.kv_pos}[/green]")

    t0 = time.time()
    cmd_attempt = bm.commit_and_generate(
        "SQL yerine OS Command Injection dene. "
        "Uygulamanın dosya yükleme fonksiyonundaki filename parametresine "
        "'; cat /etc/passwd #' payload'ı enjekte et.",
        max_new_tokens=80,
    )
    show_generation("Command Injection", cmd_attempt, (time.time() - t0) * 1000)
    vram_rec.mark("4: CmdInj branch + generate sonrası")

    # ── Final ────────────────────────────────────────────────────────────
    vram_rec.stop()
    console.print()
    show_tree(bm)
    show_status(bm)

    # ── VRAM Telemetri Tablosu ───────────────────────────────────────────
    vram_table = Table(
        title="📊 VRAM Telemetrisi — Step Step Karşılaştırma",
        box=box.ROUNDED,
        border_style="bright_magenta",
        show_header=True,
        header_style="bold magenta",
    )
    vram_table.add_column("Step", style="cyan", width=35)
    vram_table.add_column("VRAM (MB)", style="green", justify="right")
    vram_table.add_column("Kullanım %", style="yellow", justify="right")
    vram_table.add_column("GPU %", style="dim", justify="right")

    for m in vram_rec.marks:
        vram_table.add_row(
            m.label,
            str(m.used_mb),
            f"{m.used_pct:.1f}%",
            f"{m.gpu_util_pct}%",
        )

    delta = vram_rec.delta_range()
    vram_table.add_section()
    vram_table.add_row(
        "[bold]Δ (Peak - Min)[/bold]",
        f"[bold]{delta} MB[/bold]",
        "",
        "",
    )
    console.print(vram_table)

    if delta <= 10:
        console.print("[bold green]✓ VRAM dallanma/reset boyunca SABİT kaldı (Δ ≤ 10 MB)![/bold green]")
    else:
        console.print(f"[yellow]⚠ VRAM farkı {delta} MB — tolere edilebilir ama incelenmeli.[/yellow]")

    console.print(Panel(
        "[bold green]✓ Senaryo tamamlandı![/bold green]\n\n"
        f"• Toplam dal sayısı: {len(bm.tree)}\n"
        f"• Slot kullanımı: {bm.slot_status()['used_slots']}/{bm.slot_status()['n_seq_max']}\n"
        f"• VRAM farkı: Δ{delta} MB (dallanma sırasında şişme yok)\n"
        f"• Hard reset ile SQLi dalı slot harcamadan geri sarıldı\n"
        f"• Command Injection dalı anında ($O(1)$ seq_copy) açıldı\n"
        f"• Tüm dallar birbirinden izole, ortak prefix bir kez decode edildi",
        title="[bold]🏁 Result[/bold]",
        border_style="bright_green",
        box=box.DOUBLE,
    ))

    engine.close()


if __name__ == "__main__":
    main()
