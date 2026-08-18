import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

# 1500 token'lik uzun bir baseline metni
LONG_PROMPT = "The quick brown fox jumps over the lazy dog.\n" * 150

def run_benchmark():
    console.print(Panel(
        "[bold white]📊 GERÇEK BASELINE KARŞILAŞTIRMASI[/bold white]\n\n"
        "[dim]A Yolu: Traditional (Her dal için tam Prefill)\n"
        "B Yolu: AgenticVMM (seq_copy ile O(1) dal açma)[/dim]",
        border_style="bright_blue",
        box=box.DOUBLE,
    ))

    num_branches = 7
    
    # --- B YOLU: AGENTICVMM (seq_copy) ---
    console.print("\n[bold cyan]▶ B Yolu: AgenticVMM (seq_copy) Başlıyor...[/bold cyan]")
    engine_b = LlamaEngine(model_path=MODEL_PATH, n_ctx=16384, n_seq_max=8, n_gpu_layers=-1)
    bm = BranchManager(engine_b)
    root = bm.active_node()
    tokens = engine_b.tokenize(LONG_PROMPT, add_bos=True)
    
    vram_rec_b = VRAMRecorder(interval_sec=0.2)
    vram_rec_b.start()
    vram_rec_b.mark("B_Init")
    
    t0 = time.time()
    engine_b.generate(seq_id=root.seq_id, prompt_tokens=tokens, pos_start=0, max_new_tokens=0)
    root.kv_pos = len(tokens)
    prefill_time_b = (time.time() - t0) * 1000
    vram_rec_b.mark("B_Prefill")
    
    b_branch_times = []
    branches = []
    
    for i in range(num_branches):
        bm.checkout(root.node_id)
        t_branch = time.time()
        new_branch = bm.create_branch(root.node_id)
        b_time = (time.time() - t_branch) * 1000
        b_branch_times.append(b_time)
        branches.append(new_branch)
        time.sleep(0.1) # VRAM örneklemesi için ufak gecikme
        
    vram_rec_b.mark("B_Branches_Created")
    vram_rec_b.stop()
    engine_b.close()
    time.sleep(1) # VRAM'in tam boşaldığından emin ol
    
    
    # --- A YOLU: GELENEKSEL (Full Prefill) ---
    console.print("\n[bold yellow]▶ A Yolu: Traditional (Full Prefill) Başlıyor...[/bold yellow]")
    engine_a = LlamaEngine(model_path=MODEL_PATH, n_ctx=16384, n_seq_max=8, n_gpu_layers=-1)
    # Traditional yöntemde her dal bağımsız bir seq_id kullanır ve sıfırdan prefill edilir.
    vram_rec_a = VRAMRecorder(interval_sec=0.2)
    vram_rec_a.start()
    vram_rec_a.mark("A_Init")
    
    a_branch_times = []
    try:
        for i in range(num_branches):
            t_branch = time.time()
            seq_id = i + 1  # 1'den başla
            engine_a.generate(seq_id=seq_id, prompt_tokens=tokens, pos_start=0, max_new_tokens=0)
            a_time = (time.time() - t_branch) * 1000
            a_branch_times.append(a_time)
            vram_rec_a.mark(f"A_Branch_{i+1}")
            time.sleep(0.1)
    except Exception as e:
        console.print(f"[bold red]HATA: Traditional yöntem {i+1}. dalda çöktü! ({e})[/bold red]")
        
    vram_rec_a.stop()
    engine_a.close()


    # --- SONUÇLAR ---
    console.print()
    table = Table(title="📈 Baseline Karşılaştırması (Latency Eğrisi)", box=box.ROUNDED, border_style="white")
    table.add_column("Branch No", style="magenta")
    table.add_column("Traditional (Full Prefill) Latency", style="red")
    table.add_column("AgenticVMM (seq_copy) Latency", style="green")
    
    for i in range(num_branches):
        a_time = f"{a_branch_times[i]:.1f} ms" if i < len(a_branch_times) else "ÇÖKTÜ"
        b_time = f"{b_branch_times[i]:.2f} ms" if i < len(b_branch_times) else "HATA"
        table.add_row(f"Dal {i+1}", a_time, b_time)
        
    avg_a_time = (sum(a_branch_times) / len(a_branch_times)) if a_branch_times else 0
    avg_b_time = (sum(b_branch_times) / len(b_branch_times)) if b_branch_times else 0
    table.add_section()
    table.add_row("[bold]Average Latency[/bold]", f"[bold]{avg_a_time:.1f} ms[/bold]", f"[bold]{avg_b_time:.2f} ms[/bold]")
    
    console.print(table)
    
    console.print(Panel(
        "[bold yellow]Dürüst VRAM Metrikleri ve Açıklama:[/bold yellow]\n"
        "llama.cpp motoru `n_ctx` boyutundaki KV-Cache havuzunu (bu testte ~896 MB) motor başlarken blok halinde "
        "rezerve eder. Bu yüzden çalışma anındaki 'VRAM Şişmesi (Delta)' geleneksel yöntemde de, AgenticVMM'de de "
        "düşük (10-20 MB) görünür. Ancak asıl kazanç bellekte değil, [bold white]TEKRAR HESAPLAMA (Prefill) MALİYETİNİN YOK OLMASIDIR[/bold white].\n\n"
        "Traditional yöntem aynı bağlamı her dal için sıfırdan prefill ederken (bağlam uzadıkça süre artar), "
        "AgenticVMM pointer seviyesinde kopyalama yaparak O(1) maliyetle anında çoğaltır.",
        border_style="yellow",
        box=box.SIMPLE
    ))

if __name__ == "__main__":
    run_benchmark()
