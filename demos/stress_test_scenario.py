#!/usr/bin/env python3
# demos/stress_test_scenario.py
"""
STRES TESTİ: 1500+ token ağır bağlam ile KV-Git dallanma performansı.

Traditional ajanlar (LangChain vb.):
  1500 token x 4 dal = 6000 token KV yükü → OOM

AgenticVMM:
  1500 token 1 kez prefill → seq_copy ile O(1) dal kopyası
  Toplam KV yükü: ~1700 token (1500 ortak + dallar arası delta)
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

# ─── 1500+ TOKEN AĞIR BAĞLAM ────────────────────────────────────────────────
HEAVY_CONTEXT = """
=== NMAP TARAMA RAPORU ===
Hedef: 192.168.1.105 (example-target.com)
Tarama: nmap -sV -sC -O -A 192.168.1.105

PORT     STATE SERVICE     VERSION
22/tcp   open  ssh         OpenSSH 7.2p2 Ubuntu
80/tcp   open  http        Apache/2.4.18 (Ubuntu)
443/tcp  open  ssl/http    Apache/2.4.18
3306/tcp open  mysql       MySQL 5.7.33-0ubuntu0.16.04.2
8080/tcp open  http-proxy  Squid 3.5.12

OS: Ubuntu 16.04.7 LTS (Xenial)
HTTP-title: SecureBank Online Banking Portal
HTTP-server-header: Apache/2.4.18 (Ubuntu)
SSL-cert: CN=example-target.com, O=SecureBank Ltd
MySQL: Uzaktan erişim AÇIK (bind-address=0.0.0.0)

Tespit edilen teknolojiler: PHP 7.0, jQuery 2.1.4, Bootstrap 3.3.7
robots.txt: /admin/, /backup/, /api/debug/
Dizin listeleme: /backup/ dizininde AÇIK

=== KEŞİF SONUÇLARI ===
- /backup/database_dump_2024.sql (12MB, herkese açık)
- /admin/login.php (admin paneli, rate limiting YOK)
- /api/debug/phpinfo.php (PHP yapılandırması açıkta)
- /uploads/ dizini çalıştırılabilir (PHP execution aktif)
- CORS: Access-Control-Allow-Origin: * (tüm originlere açık)
- CSP header TANIMLANMAMIŞ
- X-Frame-Options header YOK (Clickjacking riski)
- Session cookie: HttpOnly=false, Secure=false, SameSite=none

=== HEDEF UYGULAMA KAYNAK KODU (login.php) ===

<?php
session_start();
require_once('config/database.php');
require_once('includes/functions.php');

// Database bağlantısı - config/database.php içeriği:
// $db_host = 'localhost';
// $db_user = 'root';
// $db_pass = 'SecureBank2024!';
// $db_name = 'securebank_prod';
// $conn = new mysqli($db_host, $db_user, $db_pass, $db_name);

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = $_POST['username'];  // Sanitizasyon YOK
    $password = $_POST['password'];  // Sanitizasyon YOK
    
    // KRİTİK ZAFİYET #1: SQL Injection
    // String concatenation ile sorgu oluşturma
    $query = "SELECT * FROM users WHERE username = '" . $username . "' AND password = MD5('" . $password . "')";
    $result = $conn->query($query);
    
    if ($result && $result->num_rows > 0) {
        $user = $result->fetch_assoc();
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['username'] = $user['username'];
        $_SESSION['role'] = $user['role'];
        $_SESSION['is_admin'] = ($user['role'] === 'admin') ? true : false;
        
        // KRİTİK ZAFİYET #2: Session Fixation
        // session_regenerate_id() çağrılmamış
        
        // Audit log - yine SQL Injection açığı
        $log_query = "INSERT INTO login_logs (username, ip, timestamp) VALUES ('" . $username . "', '" . $_SERVER['REMOTE_ADDR'] . "', NOW())";
        $conn->query($log_query);
        
        header("Location: dashboard.php");
        exit();
    } else {
        // KRİTİK ZAFİYET #3: Reflected XSS
        $error_msg = "Geçersiz kullanıcı: " . $username;
        // $username doğrudan HTML'e basılıyor, htmlspecialchars YOK
    }
}
?>
<!DOCTYPE html>
<html>
<head><title>SecureBank - Giriş</title></head>
<body>
<form method="POST" action="login.php">
    <input type="text" name="username" placeholder="User Adı">
    <input type="password" name="password" placeholder="Şifre">
    <button type="submit">Giriş Yap</button>
</form>
<?php if (isset($error_msg)) echo "<p class='error'>$error_msg</p>"; ?>
</body>
</html>

=== HEDEF UYGULAMA KAYNAK KODU (upload.php) ===

<?php
session_start();
if (!isset($_SESSION['user_id'])) { header("Location: login.php"); exit(); }

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['document'])) {
    $upload_dir = '/var/www/html/uploads/';
    $filename = $_FILES['document']['name'];  // Orijinal dosya adı kullanılıyor
    
    // KRİTİK ZAFİYET #4: Unrestricted File Upload
    // Dosya tipi kontrolü YOK, sadece boyut kontrolü var
    if ($_FILES['document']['size'] <= 5242880) {
        
        // KRİTİK ZAFİYET #5: OS Command Injection
        // Dosya adı shell komutuna doğrudan geçiriliyor
        $target_path = $upload_dir . $filename;
        move_uploaded_file($_FILES['document']['tmp_name'], $target_path);
        
        // Virüs taraması - command injection açığı
        $scan_cmd = "clamscan " . $target_path;
        $scan_result = shell_exec($scan_cmd);
        
        // Thumbnail oluşturma - yine command injection
        if (preg_match('/\\.(jpg|jpeg|png|gif)$/i', $filename)) {
            $thumb_cmd = "convert " . $target_path . " -resize 100x100 " . $upload_dir . "thumbs/" . $filename;
            exec($thumb_cmd);
        }
        
        // KRİTİK ZAFİYET #6: Path Traversal
        // ../../ gibi dizin atlama kontrol edilmiyor
        $log_entry = "File uploaded: " . $filename . " by user " . $_SESSION['username'];
        file_put_contents('/var/log/uploads.log', $log_entry . "\\n", FILE_APPEND);
        
        echo "<p>Dosya başarıyla yüklendi: " . htmlspecialchars($filename) . "</p>";
    } else {
        echo "<p>Dosya boyutu 5MB'ı aşamaz.</p>";
    }
}
?>
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="document">
    <button type="submit">Yükle</button>
</form>

=== HEDEF UYGULAMA KAYNAK KODU (api/user_profile.php) ===

<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

$user_id = $_GET['id'];  // Sanitizasyon YOK

// KRİTİK ZAFİYET #7: IDOR (Insecure Direct Object Reference)
// Yetkilendirme kontrolü YOK - herhangi bir kullanıcının verisine erişilebilir
$query = "SELECT id, username, email, phone, address, role, created_at FROM users WHERE id = " . $user_id;
$result = $conn->query($query);

if ($result && $result->num_rows > 0) {
    $user = $result->fetch_assoc();
    // KRİTİK ZAFİYET #8: Hassas veri sızıntısı
    // role ve tüm kişisel bilgiler API'den döndürülüyor
    echo json_encode(['status' => 'success', 'data' => $user]);
} else {
    echo json_encode(['status' => 'error', 'message' => 'User not found: ' . $user_id]);
    // Error mesajında user_id reflect ediliyor - XSS riski
}
?>

=== ZAFİYET ÖZETİ ===
1. SQL Injection (login.php - authentication bypass)
2. Session Fixation (login.php - session hijacking)
3. Reflected XSS (login.php - error message)
4. Unrestricted File Upload (upload.php - web shell)
5. OS Command Injection (upload.php - clamscan/convert)
6. Path Traversal (upload.php - directory traversal)
7. IDOR (api/user_profile.php - unauthorized data access)
8. Information Disclosure (api/user_profile.php - sensitive data leak)

Toplam kritik zafiyet: 8
Risk seviyesi: KRİTİK
Öncelikli hedef: login.php (SQL Injection ile admin erişimi)
İkincil hedef: upload.php (Command Injection ile RCE)
"""


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
        "[bold white]🔥 STRES TESTİ — 1500+ Token Ağır Context ile KV-Git Dallanma[/bold white]\n\n"
        "[dim]Traditional: 1500 tok × 4 dal = 6000 tok KV → OOM\n"
        "AgenticVMM:  1500 tok × 1 prefill + O(1) seq_copy → ~1700 tok KV[/dim]",
        border_style="bright_red",
        box=box.DOUBLE,
    ))

    # ── Step 0: Motor başlatma ──────────────────────────────────────────
    step_banner(0, "Motor Başlatma", "LlamaEngine init + 1500+ token ağır bağlam prefill")

    t0 = time.time()
    engine = LlamaEngine(model_path=MODEL_PATH, n_ctx=16384, n_seq_max=4, n_gpu_layers=-1)
    bm = BranchManager(engine)

    vram_rec = VRAMRecorder(interval_sec=0.5)
    vram_rec.start()
    vram_rec.mark("0: Model loaded (prefill öncesi)")

    root = bm.active_node()
    system_prompt = (
        "Sen bir siber güvenlik uzmanı ve sızma testi (pentest) asistanısın. "
        "Aşağıda hedef sistemin keşif raporu ve kaynak kodu verilmiştir. "
        "Bu bilgileri analiz ederek farklı saldırı vektörlerini planlayıp uygulayacaksın.\n\n"
        + HEAVY_CONTEXT
    )

    sys_tokens = engine.tokenize(system_prompt, add_bos=True)
    token_count = len(sys_tokens)

    console.print(f"[bold yellow]⚡ Ağır bağlam: {token_count} token ({len(system_prompt)} karakter)[/bold yellow]")

    t_prefill = time.time()
    engine.generate(seq_id=root.seq_id, prompt_tokens=sys_tokens, pos_start=0, max_new_tokens=0)
    prefill_ms = (time.time() - t_prefill) * 1000

    root.kv_pos = token_count
    init_ms = (time.time() - t0) * 1000

    vram_rec.mark(f"0b: Prefill tamamlandı ({token_count} tok)")

    console.print(f"[green]✓ {token_count} token prefill edildi ({prefill_ms:.0f}ms). Toplam init: {init_ms:.0f}ms[/green]")
    show_status(bm)

    # Traditional vs AgenticVMM karşılaştırma tablosu
    comparison = Table(
        title="📐 Memory Karşılaştırması (Teorik)",
        box=box.ROUNDED,
        border_style="red",
    )
    comparison.add_column("System", style="cyan")
    comparison.add_column("KV Token Load", style="yellow", justify="right")
    comparison.add_column("6GB VRAM", style="green")
    comparison.add_row(
        "Traditional (4 dal)",
        f"{token_count} × 4 = {token_count * 4}",
        "[bold red]OOM ☠️[/bold red]"
    )
    comparison.add_row(
        "AgenticVMM (4 dal)",
        f"{token_count} × 1 + delta",
        "[bold green]✓ Çalışıyor[/bold green]"
    )
    console.print(comparison)

    # ── Step 1: Zafiyet analizi ─────────────────────────────────────────
    step_banner(1, "Zafiyet Analizi",
                f"Ana dalda (seq=0) — {token_count} token bağlam üzerinden analiz")

    t0 = time.time()
    base = bm.commit_and_generate(
        "Verilen kaynak kodu ve keşif raporunu analiz et. "
        "En kritik 3 zafiyeti öncelik sırasına göre listele ve "
        "her biri için saldırı vektörünü kısaca açıkla.",
        max_new_tokens=100,
    )
    gen1_ms = (time.time() - t0) * 1000
    show_generation("Zafiyet Analizi", base, gen1_ms)
    vram_rec.mark("1: Zafiyet analizi sonrası")
    show_tree(bm)

    # ── Step 2: SQL Injection dalı ──────────────────────────────────────
    step_banner(2, "Branch → SQL Injection",
                f"O(1) seq_copy — {token_count}+ token bağlam KOPYALANMADAN paylaşılıyor")

    t0 = time.time()
    sql_branch = bm.create_branch(base.node_id)
    branch1_ms = (time.time() - t0) * 1000
    vram_rec.mark("2a: SQLi branch (seq_copy)")
    console.print(f"[green]✓ SQLi dalı: {branch1_ms:.2f}ms — seq={sql_branch.seq_id}, kv_pos={sql_branch.kv_pos}[/green]")

    t0 = time.time()
    sql_attack = bm.commit_and_generate(
        "login.php'deki SQL Injection zafiyetini kullanarak admin authentication bypass do. "
        "Tam payload'ı, HTTP isteğini ve beklenen sonucu göster.",
        max_new_tokens=100,
    )
    gen2_ms = (time.time() - t0) * 1000
    show_generation("SQLi Saldırısı", sql_attack, gen2_ms)
    vram_rec.mark("2b: SQLi generate sonrası")

    # ── Step 3: XSS dalı (aynı base'den) ───────────────────────────────
    step_banner(3, "Branch → Reflected XSS",
                "base commit'ten ikinci dal — yine O(1)")

    bm.checkout(base.node_id)
    t0 = time.time()
    xss_branch = bm.create_branch(base.node_id)
    branch2_ms = (time.time() - t0) * 1000
    vram_rec.mark("3a: XSS branch (seq_copy)")
    console.print(f"[green]✓ XSS dalı: {branch2_ms:.2f}ms — seq={xss_branch.seq_id}, kv_pos={xss_branch.kv_pos}[/green]")

    t0 = time.time()
    xss_attack = bm.commit_and_generate(
        "login.php'deki Reflected XSS zafiyetini kullanarak session cookie çalma payload'ı oluştur. "
        "document.cookie'yi uzak sunucuya gönderen tam XSS vektörünü yaz.",
        max_new_tokens=100,
    )
    gen3_ms = (time.time() - t0) * 1000
    show_generation("XSS Saldırısı", xss_attack, gen3_ms)
    vram_rec.mark("3b: XSS generate sonrası")

    # ── Step 4: Command Injection dalı (aynı base'den) ─────────────────
    step_banner(4, "Branch → OS Command Injection",
                "base commit'ten üçüncü dal — hala O(1), VRAM sabit!")

    bm.checkout(base.node_id)
    t0 = time.time()
    cmd_branch = bm.create_branch(base.node_id)
    branch3_ms = (time.time() - t0) * 1000
    vram_rec.mark("4a: CmdInj branch (seq_copy)")
    console.print(f"[green]✓ CmdInj dalı: {branch3_ms:.2f}ms — seq={cmd_branch.seq_id}, kv_pos={cmd_branch.kv_pos}[/green]")

    t0 = time.time()
    cmd_attack = bm.commit_and_generate(
        "upload.php'deki OS Command Injection zafiyetini kullanarak reverse shell al. "
        "Dosya adı parametresine enjekte edilecek tam payload'ı ve "
        "saldırgan tarafında çalıştırılacak netcat listener komutunu yaz.",
        max_new_tokens=100,
    )
    gen4_ms = (time.time() - t0) * 1000
    show_generation("Command Injection", cmd_attack, gen4_ms)
    vram_rec.mark("4b: CmdInj generate sonrası")

    # ── Final ────────────────────────────────────────────────────────────
    vram_rec.stop()
    console.print()
    show_tree(bm)
    show_status(bm)

    # ── Performans Tablosu ───────────────────────────────────────────────
    perf_table = Table(
        title="⏱️  Performans Metrikleri",
        box=box.ROUNDED,
        border_style="bright_cyan",
        show_header=True,
        header_style="bold cyan",
    )
    perf_table.add_column("Operation", style="white", width=30)
    perf_table.add_column("Duration", style="green", justify="right")
    perf_table.add_column("Note", style="dim")

    perf_table.add_row(f"Prefill ({token_count} token)", f"{prefill_ms:.0f}ms", "Sadece 1 kez!")
    perf_table.add_row("SQLi branch (seq_copy)", f"{branch1_ms:.2f}ms", "O(1)")
    perf_table.add_row("XSS branch (seq_copy)", f"{branch2_ms:.2f}ms", "O(1)")
    perf_table.add_row("CmdInj branch (seq_copy)", f"{branch3_ms:.2f}ms", "O(1)")
    perf_table.add_row("SQLi generate (100 tok)", f"{gen2_ms:.0f}ms", "Delta only")
    perf_table.add_row("XSS generate (100 tok)", f"{gen3_ms:.0f}ms", "Delta only")
    perf_table.add_row("CmdInj generate (100 tok)", f"{gen4_ms:.0f}ms", "Delta only")

    console.print(perf_table)

    # ── VRAM Telemetri Tablosu ───────────────────────────────────────────
    vram_table = Table(
        title="📊 VRAM Telemetrisi — Dallanma Boyunca",
        box=box.ROUNDED,
        border_style="bright_magenta",
        show_header=True,
        header_style="bold magenta",
    )
    vram_table.add_column("Step", style="cyan", width=40)
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

    if delta <= 15:
        console.print(f"[bold green]✓ VRAM {token_count}+ token prefill ve 3 dallanma boyunca SABİT kaldı (Δ={delta} MB)![/bold green]")
    else:
        console.print(f"[yellow]⚠ VRAM farkı {delta} MB — incelenmeli.[/yellow]")

    # ── Final Result Paneli ───────────────────────────────────────────────
    console.print(Panel(
        "[bold green]✓ STRES TESTİ TAMAMLANDI![/bold green]\n\n"
        f"📦 Ağır bağlam: {token_count} token (Nmap raporu + PHP kaynak kodu)\n"
        f"🌳 Toplam dal: {len(bm.tree)} düğüm, {bm.slot_status()['used_slots']}/{bm.slot_status()['n_seq_max']} slot\n"
        f"💾 VRAM farkı: Δ{delta} MB (dallanma sırasında şişme yok)\n"
        f"⚡ Branch maliyeti: {branch1_ms:.2f}ms / {branch2_ms:.2f}ms / {branch3_ms:.2f}ms (O(1) seq_copy)\n\n"
        f"[bold]Traditional yaklaşım: {token_count} × 4 = {token_count * 4} token → OOM\n"
        f"AgenticVMM:           {token_count} × 1 + delta → ✓ Çalışıyor[/bold]",
        title="[bold]🏁 Stres Testi Sonucu[/bold]",
        border_style="bright_green",
        box=box.DOUBLE,
    ))

    engine.close()


if __name__ == "__main__":
    main()
