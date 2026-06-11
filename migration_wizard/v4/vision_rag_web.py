#!/usr/bin/env python3
"""
Vision RAG — Complete Migration Wizard v4.0 (Web UI)
Flask + SSE backend with premium dark HTML/CSS/JS frontend.

Self-contained: all logic in this single file. No .md or .sh files needed.
"""

import os, sys, json, hashlib, shutil, subprocess, threading, time, re
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify, Response


# ── Design tokens (shared with frontend) ───────────────────────────────────
class T:
    BG1 = "#0a0d12"
    BG2 = "#0f1219"
    BG3 = "#141820"
    CARD = "#161b24"
    CARD2 = "#1c2230"
    HOVER = "#1e2536"
    INPUT = "#111620"
    INPUT_B = "#1a2030"
    ACCENT = "#5b9aff"
    AC2 = "#7db4ff"
    AC3 = "#3d7be0"
    ACCENT_G1 = "#4a8af4"
    ACCENT_G2 = "#7c5cf5"
    OK = "#34d399"
    OK2 = "#2ab385"
    OK_G1 = "#22c997"
    OK_G2 = "#34d399"
    WARN = "#fbbf24"
    WARN2 = "#f59e0b"
    WARN_G1 = "#f59e0b"
    WARN_G2 = "#fbbf24"
    ERR = "#f43f5e"
    ERR2 = "#e11d48"
    ERR_G1 = "#ef4444"
    ERR_G2 = "#f43f5e"
    TXT = "#f0f4f8"
    TXT2 = "#94a3b8"
    MUTE = "#475569"
    BORDER = "#1e293b"
    B2 = "#334155"
    B3 = "#0f172a"


# ── Utilities ──────────────────────────────────────────────────────────────
def hsize(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(65536)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def win2wsl(wp):
    if len(wp) >= 2 and wp[1:3] == ":\\":
        d, r = wp[0].lower(), wp[2:].replace("\\", "/")
        m = f"/mnt/{d}{r}"
        if Path(m).exists():
            return m
    try:
        r = subprocess.run(
            ["wslpath", "-u", wp], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def is_valid_project(p):
    p = Path(p)
    return p.exists() and (p / "docker-compose.yml").exists()


def get_prefix():
    try:
        r = subprocess.run(
            ["docker", "volume", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        for line in r.stdout.splitlines():
            if "_hf-cache" in line:
                return line.replace("_hf-cache", "")
    except Exception:
        pass
    return "vision_rag_git"


# ── SSE Log Stream ─────────────────────────────────────────────────────────
# ── Log polling instead of SSE ──────────────────────────────────────────────
_log_data = {"source": [], "target": []}
_log_pos = {"source": 0, "target": 0}


def log_emit(channel, msg, level="INFO"):
    colors = {"INFO": T.TXT2, "OK": T.OK, "WARN": T.WARN, "ERROR": T.ERR, "CMD": T.AC2}
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"ts": ts, "level": level, "msg": msg}
    _log_data[channel].append(entry)
    if len(_log_data[channel]) > 2000:
        _log_data[channel] = _log_data[channel][-2000:]
    print(f"[{ts}] {level}: {msg}")


# ── Wizard State ───────────────────────────────────────────────────────────
class State:
    def __init__(s):
        s.proj = ""
        s.dok = ""
        s.mode = ""
        s.bkp_done = False
        s.bkp_path = ""
        s.vol_prefix = "vision_rag_git"
        s.imgs = []
        s.need_sudo_pw = False
        s.sudo_pw = ""
        s.verify = {}
        s.manifest_images = []
        s.manifest_code = {}
        s.source_steps = {
            "disc": "pending",
            "exp": "pending",
            "code": "pending",
            "bun": "pending",
        }
        s.target_steps = {
            "detect": "pending",
            "backup": "pending",
            "import": "pending",
            "verify": "pending",
        }
        s.export_dir = ""


def get_export_dir():
    ts = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"VisionRAG_Update_{ts}"
    dok = state.dok
    dp = Path(dok) / folder_name
    try:
        dp.mkdir(parents=True, exist_ok=True)
        state.export_dir = str(dp)
        return dp
    except OSError:
        pass
    if dok.startswith("/mnt/"):
        parts = dok.split("/")
        if len(parts) >= 3:
            drive_path = Path(f"/mnt/{parts[2]}")
            if _ensure_mounted(drive_path):
                try:
                    dp.mkdir(parents=True, exist_ok=True)
                    state.export_dir = str(dp)
                    return dp
                except OSError:
                    pass
    try:
        dok_win = dok.replace("/mnt/", "").replace("/", "\\")
        if len(dok_win) >= 2 and dok_win[1] == "\\":
            dok_win = dok_win[0].upper() + ":" + dok_win[1:]
        subprocess.run(
            ["cmd.exe", "/c", "mkdir", dok_win],
            capture_output=True,
            timeout=10,
            check=True,
        )
        dp.mkdir(parents=True, exist_ok=True)
        state.export_dir = str(dp)
        return dp
    except Exception:
        pass
    log_emit(
        "source",
        f"Cannot access {dok}. Ensure the drive is connected and mounted.",
        "ERROR",
    )
    raise OSError(f"Cannot access {dok}")


def _try_mount_all_drives():
    try:
        subprocess.run(["sudo", "mount", "-a"], capture_output=True, timeout=10)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["cmd.exe", "/c", "wmic", "logicaldisk", "get", "caption"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if len(line) == 2 and line[1] == ":" and line[0].isalpha():
                letter = line[0].lower()
                mnt = Path(f"/mnt/{letter}")
                try:
                    list(mnt.iterdir())
                    continue
                except Exception:
                    pass
                try:
                    subprocess.run(
                        [
                            "sudo",
                            "mount",
                            "-t",
                            "drvfs",
                            f"{letter.upper()}:",
                            str(mnt),
                        ],
                        capture_output=True,
                        timeout=5,
                    )
                except Exception:
                    pass
    except Exception:
        pass


def _mnt_drives():
    drives = []
    mounted = set()
    try:
        with open("/proc/mounts") as f:
            for line in f:
                if "drvfs" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        mp = parts[1]
                        try:
                            for e in Path(mp).iterdir():
                                if e.is_dir() and e.name.isalpha() and len(e.name) == 1:
                                    mounted.add(e.name.lower())
                        except Exception:
                            pass
    except Exception:
        pass
    try:
        for e in sorted(Path("/mnt").iterdir()):
            if (
                e.is_dir()
                and e.name not in ("c", "wsl", "wslg", "wslg-bind")
                and e.name.isalpha()
                and len(e.name) == 1
            ):
                if e.name in mounted:
                    drives.append(e)
                else:
                    drives.append(e)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["cmd.exe", "/c", "wmic", "logicaldisk", "get", "caption"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if len(line) == 2 and line[1] == ":" and line[0].isalpha():
                p = Path(f"/mnt/{line[0].lower()}")
                if p not in drives:
                    drives.append(p)
    except Exception:
        pass
    return drives


def _is_drvfs_mounted(mnt_path):
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4 and parts[1] == str(mnt_path) and "drvfs" in line:
                    return True
    except Exception:
        pass
    return False


def _ensure_mounted(drive_path):
    if _is_drvfs_mounted(drive_path):
        return True
    letter = drive_path.name
    try:
        subprocess.run(
            ["sudo", "mount", "-t", "drvfs", f"{letter.upper()}:", str(drive_path)],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
    return _is_drvfs_mounted(drive_path)


state = State()


# ── Auto-detect ────────────────────────────────────────────────────────────
def auto_detect_project():
    home = Path.home()
    for p in [
        home / "projects" / "Vision_RAG_Git",
        home / "projects" / "Vision_RAG",
        home / "Vision_RAG_Git",
        home / "Vision_RAG",
    ]:
        if is_valid_project(p):
            return str(p)
    proj_dir = home / "projects"
    candidates = []
    if proj_dir.exists():
        try:
            for c in proj_dir.iterdir():
                if c.is_dir() and is_valid_project(c):
                    candidates.append(c)
        except Exception:
            pass
    for drive in _mnt_drives():
        _ensure_mounted(drive)
        try:
            for c in drive.iterdir():
                if (
                    c.is_dir()
                    and ("Vision_RAG" in c.name or "vision_rag" in c.name.lower())
                    and is_valid_project(c)
                ):
                    candidates.append(c)
        except Exception:
            pass
    for c in candidates:
        if "vision_rag" in c.name.lower():
            return str(c)
    if candidates:
        return str(candidates[0])
    return None


def auto_detect_dok():
    for drive in _mnt_drives():
        if _ensure_mounted(drive):
            p = drive / "VisionRAG_Update"
            if p.exists():
                return str(p)
    for drive in _mnt_drives():
        letter = drive.name.upper()
        try:
            r = subprocess.run(
                ["cmd.exe", "/c", f"if exist {letter}:\\VisionRAG_Update echo YES"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "YES" in r.stdout:
                _ensure_mounted(drive)
                p = drive / "VisionRAG_Update"
                if p.exists():
                    return str(p)
                return str(drive / "VisionRAG_Update")
        except Exception:
            pass
    for drive in _mnt_drives():
        if _ensure_mounted(drive):
            try:
                for d in sorted(drive.iterdir(), reverse=True):
                    if d.is_dir() and d.name.startswith("VisionRAG_Update_"):
                        return str(d)
            except Exception:
                pass
    return None


# ── Restore script generator ──────────────────────────────────────────────
def gen_restore():
    return r"""#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${{BLUE}}[INFO]${{NC}} $*"; }
ok() { echo -e "${{GREEN}}[OK]${{NC}} $*"; }
warn() { echo -e "${{YELLOW}}[WARN]${{NC}} $*"; }
error() { echo -e "${{RED}}[ERROR]${{NC}} $*"; }

echo "========================================"
echo "  Vision RAG — RESTORE FROM BACKUP"
echo "========================================"
read -p "Type 'yes' to continue: " confirm
if [ "${{confirm}}" != "yes" ]; then echo "Aborted."; exit 0; fi

PROJECT_DIR=""
if [ -f "${{SCRIPT_DIR}}/PROJECT_DIR.txt" ]; then PROJECT_DIR=$(cat "${{SCRIPT_DIR}}/PROJECT_DIR.txt"); fi
if [ -z "${{PROJECT_DIR}}" ] || [ ! -d "${{PROJECT_DIR}}" ]; then read -p "Enter project directory: " PROJECT_DIR; fi
if [ ! -d "${{PROJECT_DIR}}" ]; then error "Not found"; exit 1; fi

VOLUME_PREFIX=""
if [ -f "${{SCRIPT_DIR}}/docker/volumes-list.txt" ]; then VOLUME_PREFIX=$(head -n 1 "${{SCRIPT_DIR}}/docker/volumes-list.txt" | sed 's/_hf-cache//'); fi
if [ -z "${{VOLUME_PREFIX}}" ]; then VOLUME_PREFIX="vision_rag_git"; fi

info "Restoring from: ${{SCRIPT_DIR}}"
info "Project: ${{PROJECT_DIR}}"
info "Prefix: ${{VOLUME_PREFIX}}"

info "[1/4] Stopping containers..."
cd "${{PROJECT_DIR}}"
docker compose down 2>/dev/null || true
ok "Stopped"

info ""
info "[2/4] Restoring volumes..."
if [ -d "${{SCRIPT_DIR}}/volumes" ]; then
    for archive in "${{SCRIPT_DIR}}"/volumes/*.tar.gz; do
        [ -f "${{archive}}" ] || continue
        suffix=$(basename "${{archive}}" .tar.gz)
        name="${{VOLUME_PREFIX}}_${{suffix}}"
        info "  Restoring ${{name}}..."
        docker volume create "${{name}}" >/dev/null 2>&1 || true
        docker run --rm -v "${{name}}:/dest" -v "${{archive}}:/source.tar.gz:ro" busybox sh -c "cd /dest && tar xzf /source.tar.gz" 2>/dev/null || \
        docker run --rm -v "${{name}}:/dest" -v "${{archive}}:/source.tar.gz:ro" alpine sh -c "cd /dest && tar xzf /source.tar.gz" 2>/dev/null || { warn "Failed ${{name}}"; continue; }
        ok "    ${{name}} restored"
    done
else
    warn "No volume backups"
fi

info ""
info "[3/4] Restoring code..."
if [ -d "${{SCRIPT_DIR}}/code" ]; then
    info "  Restoring full project tree..."
    rsync -rl --delete --no-perms --no-times --chmod=ugo=rwX --exclude='vision-rag-backups' --exclude='my-pdfs' --exclude='.git' "${{SCRIPT_DIR}}/code/" "${{PROJECT_DIR}}/" 2>/dev/null || \
    cp -rf "${{SCRIPT_DIR}}/code/"* "${{PROJECT_DIR}}/" 2>/dev/null || true
    ok "  Code restored"
else
    warn "  No code backup found"
fi

info ""
info "[4/4] Restarting..."
cd "${{PROJECT_DIR}}"
docker compose up -d 2>/dev/null || true
ok "Restarted"

echo ""
echo "========================================"
echo "  RESTORE COMPLETE"
echo "========================================"
ok "Reverted to backup state."
echo "Verify: cd ${{PROJECT_DIR}} && docker compose ps && curl http://localhost:8082/status"
"""


# ── Source actions (threaded) ─────────────────────────────────────────────
def _safe_copy(src, dst):
    try:
        shutil.copy(src, dst)
    except OSError:
        with open(src, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())


def _docker_save_gzip(img, out_path, timeout=3600, channel="source"):
    proc = subprocess.Popen(
        ["docker", "save", img], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stderr_thread = threading.Thread(
        target=lambda p: p.read(), args=(proc.stderr,), daemon=True
    )
    stderr_thread.start()
    gz = None
    try:
        with open(out_path, "wb") as f:
            gz = subprocess.Popen(
                ["gzip", "-c"], stdin=proc.stdout, stdout=f, stderr=subprocess.PIPE
            )
            proc.stdout.close()
            last_log = time.time()
            while gz.poll() is None:
                time.sleep(5)
                now = time.time()
                if now - last_log >= 15:
                    try:
                        sz = out_path.stat().st_size
                    except OSError:
                        sz = 0
                    log_emit(channel, f"  ... {img} {hsize(sz)} written", "INFO")
                    last_log = now
            gz_stderr = (
                gz.stderr.read().decode(errors="replace").strip() if gz.stderr else ""
            )
        proc.wait(timeout=timeout)
    except Exception:
        proc.kill()
        if gz:
            gz.kill()
            gz.wait()
        proc.wait()
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr.close()
        except Exception:
            pass
    stderr_thread.join(timeout=5)
    docker_err = ""
    try:
        docker_err = proc.stderr.read().decode(errors="replace").strip()
    except Exception:
        pass
    err_msg = docker_err or gz_stderr
    ok = proc.returncode == 0 and gz.returncode == 0
    if not ok and out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass
    return ok, err_msg


def src_disc():
    if not state.proj:
        log_emit(
            "source",
            "No project path set. Please select a project folder first.",
            "ERROR",
        )
        state.source_steps["disc"] = "error"
        return
    proj = Path(state.proj)
    if not proj.exists():
        log_emit("source", f"Project path does not exist: {state.proj}", "ERROR")
        state.source_steps["disc"] = "error"
        return
    c = proj / "docker-compose.yml"
    imgs = []
    if c.exists():
        try:
            for line in c.read_text().splitlines():
                t = line.strip()
                if t.startswith("image:"):
                    img = t.split("image:", 1)[-1].strip().strip('"').strip("'")
                    if img and img not in imgs:
                        imgs.append(img)
        except Exception as e:
            log_emit("source", f"Parse error: {e}", "WARN")
    if not imgs:
        imgs = ["ollama/ollama:latest", "qdrant/qdrant:latest", "python:3.11-alpine"]
        log_emit("source", "Using fallback image list", "WARN")
    UTILITY_IMAGES = ["busybox:latest", "alpine:latest"]
    for util in UTILITY_IMAGES:
        if util not in imgs:
            try:
                r = subprocess.run(
                    ["docker", "image", "inspect", util],
                    capture_output=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    imgs.append(util)
                    log_emit("source", f"  Utility image {util} found locally", "INFO")
                else:
                    log_emit("source", f"  Pulling utility image {util}...", "INFO")
                    r = subprocess.run(
                        ["docker", "pull", util],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if r.returncode == 0:
                        imgs.append(util)
                        log_emit("source", f"  ✓ {util} pulled successfully", "OK")
                    else:
                        log_emit(
                            "source",
                            f"  ⚠ {util} not available locally and pull failed — skipped",
                            "WARN",
                        )
            except Exception as e:
                log_emit("source", f"  ⚠ {util} check failed: {e} — skipped", "WARN")
    state.imgs = imgs
    proj_imgs = [i for i in imgs if i not in UTILITY_IMAGES]
    util_imgs = [i for i in imgs if i in UTILITY_IMAGES]
    log_emit(
        "source",
        f"Discovered {len(proj_imgs)} project + {len(util_imgs)} utility images ({len(imgs)} total):",
        "OK",
    )
    for img in imgs:
        tag = "utility" if img in UTILITY_IMAGES else "project"
        log_emit("source", f"  • {img} [{tag}]", "INFO")
    state.source_steps["disc"] = "done"


def src_exp():
    if not state.dok:
        log_emit(
            "source",
            "Cannot export images: no DOK path set. Set the export destination first.",
            "ERROR",
        )
        state.source_steps["exp"] = "error"
        return
    try:
        dp = get_export_dir()
    except Exception as e:
        log_emit("source", f"Cannot create export directory: {e}", "ERROR")
        state.source_steps["exp"] = "error"
        return
    idir = dp / "images"
    idir.mkdir(parents=True, exist_ok=True)
    log_emit("source", f"Exporting {len(state.imgs)} images to {idir}...", "INFO")
    m = []
    for img in state.imgs:
        safe = img.replace("/", "-").replace(":", "-") + ".tar.gz"
        out = idir / safe
        log_emit("source", f"Exporting {img}...", "INFO")
        try:
            ok, err = _docker_save_gzip(img, out, channel="source")
            if ok and out.stat().st_size > 0:
                sz = out.stat().st_size
                sh = sha256f(out)
                m.append(
                    {
                        "name": img,
                        "file": f"images/{safe}",
                        "sha256": sh,
                        "size_bytes": sz,
                    }
                )
                log_emit("source", f"✓ {safe} ({hsize(sz)})", "OK")
            else:
                if out.exists():
                    out.unlink(missing_ok=True)
                log_emit(
                    "source",
                    f"Failed {img}: {err or 'docker save returned empty or error'}",
                    "ERROR",
                )
        except Exception as e:
            if out.exists():
                out.unlink(missing_ok=True)
            log_emit("source", f"Exception {img}: {e}", "ERROR")
    state.manifest_images = m
    if m:
        log_emit("source", f"Exported {len(m)}/{len(state.imgs)} images.", "OK")
    else:
        log_emit("source", f"Exported 0/{len(state.imgs)} images.", "ERROR")
    state.source_steps["exp"] = "done"


def src_code():
    if not state.export_dir:
        if state.dok:
            get_export_dir()
        else:
            log_emit(
                "source",
                "Cannot export code: no DOK path set. Set the export destination first.",
                "ERROR",
            )
            state.source_steps["code"] = "error"
            return
    dp = Path(state.export_dir)
    cd = dp / "code"
    if cd.exists():
        shutil.rmtree(cd, ignore_errors=True)
    cd.mkdir(parents=True)
    proj = Path(state.proj)
    log_emit("source", f"Exporting code from {proj}...", "INFO")
    skip = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        "vision-rag-backups",
        "FULL_BackUp",
    }
    skip_ext = {".pyc", ".egg-info"}

    def _copy_tree(src, dst):
        if src.name in skip or src.suffix in skip_ext:
            return
        if src.name == ".env" and src.is_file():
            return
        if src.parent.name == "cache" and src.parent.parent.name == "pipelines":
            return
        try:
            if src.is_dir():
                dst.mkdir(exist_ok=True)
                for item in src.iterdir():
                    _copy_tree(item, dst / item.name)
            else:
                try:
                    shutil.copy(src, dst)
                except OSError:
                    try:
                        with open(src, "rb") as fi, open(dst, "wb") as fo:
                            fo.write(fi.read())
                    except Exception:
                        pass
        except OSError:
            pass

    try:
        _copy_tree(proj, cd)
        count = sum(1 for _ in cd.rglob("*") if _.is_file())
        sz = sum(_.stat().st_size for _ in cd.rglob("*") if _.is_file())
        log_emit("source", f"Code export complete: {count} files, {hsize(sz)}", "OK")
        state.manifest_code = {"file_count": count, "size_human": hsize(sz)}
        state.source_steps["code"] = "done"
    except Exception as e:
        log_emit("source", f"Code export failed: {e}", "ERROR")
        state.source_steps["code"] = "error"


def src_bun():
    if not state.export_dir:
        if state.dok:
            get_export_dir()
        else:
            log_emit("source", "Cannot bundle: no DOK path set.", "ERROR")
            state.source_steps["bun"] = "error"
            return
    dp = Path(state.export_dir)
    log_emit("source", "Bundling wizard + scripts + manifest...", "INFO")
    try:
        manifest = {
            "export_type": "vision-rag-update",
            "export_date": datetime.utcnow().isoformat() + "Z",
            "source_repo": str(Path(state.proj).resolve()),
            "images": state.manifest_images,
            "code": state.manifest_code,
        }
        with open(dp / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log_emit("source", "✓ manifest.json written", "OK")
        myself = Path(__file__)
        _safe_copy(myself, dp / "vision_rag_web.py")
        log_emit("source", "✓ vision_rag_web.py copied", "OK")
        gd = myself.parent
        for fn in ["requirements.txt", "run-wizard.sh", "VisionRAG-Wizard.desktop"]:
            src = gd / fn
            if src.exists():
                _safe_copy(src, dp / fn)
        wsl_path = str(dp / "vision_rag_web.py").replace("\\", "/")
        wsl_dir = str(dp).replace("\\", "/")
        bat = rf"""@echo off
REM Windows launcher for Vision RAG Migration Wizard (Web UI)
REM Double-click this file to start the wizard and open Chrome.

echo ============================================================
echo  Vision RAG Migration Wizard v4.0 (Web UI)
echo ============================================================
echo.
echo  Starting server... Chrome will open in a moment.
echo  If it doesn't open, go to: http://localhost:5555
echo  Close this window to stop the server.
echo.

start "" wsl.exe -d Ubuntu -e bash -c "cd {wsl_dir} && python3 vision_rag_web.py"

echo  Waiting 8 seconds for server to start...
timeout /t 8 /nobreak >nul

start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" "http://localhost:5555"
REM explorer "http://localhost:5555"
"""
        with open(dp / "Start-Wizard.bat", "w", newline="\r\n") as f:
            f.write(bat)
        log_emit("source", "✓ Start-Wizard.bat generated", "OK")
        parent = gd.parent
        for fn in ["import-update.sh", "backup-target.sh"]:
            src = parent / fn
            if src.exists():
                _safe_copy(src, dp / fn)
        for fn in [
            "UPDATE_GUIDE.md",
            "TARGET_MACHINE_SAFETY_RUNBOOK.md",
            "TARGET_QUICK_CHECKLIST.md",
        ]:
            src = parent / fn
            if src.exists():
                _safe_copy(src, dp / fn)
        log_emit("source", "═══════════════════════════════════════", "OK")
        log_emit("source", " PACKAGE COMPLETE — DOK IS SELF-CONTAINED", "OK")
        log_emit("source", f" Location: {dp}", "OK")
        log_emit("source", " Next: Eject DOK and plug into target.", "INFO")
        state.source_steps["bun"] = "done"
    except Exception as e:
        log_emit("source", f"Bundling failed: {e}", "ERROR")
        state.source_steps["bun"] = "error"


# ── Target actions (threaded) ─────────────────────────────────────────────
def tgt_detect():
    log_emit("target", "Scanning for VisionRAG_Update...", "INFO")
    cands = []
    for base in _mnt_drives():
        _ensure_mounted(base)
        vu = base / "VisionRAG_Update"
        if vu.exists():
            try:
                for d in sorted(vu.iterdir(), reverse=True):
                    if (
                        d.is_dir()
                        and d.name.startswith("VisionRAG_Update_")
                        and (d / "images").exists()
                    ):
                        cands.append(d)
                        break
            except Exception:
                pass
        try:
            for d in sorted(base.iterdir(), reverse=True):
                if (
                    d.is_dir()
                    and d.name.startswith("VisionRAG_Update_")
                    and (d / "images").exists()
                ):
                    cands.append(d)
                    break
        except Exception:
            pass
        p = base / "VisionRAG_Update"
        if p.exists() and (p / "images").exists() and p not in cands:
            cands.append(p)
    if cands:
        state.dok = str(cands[0])
        log_emit("target", f"Found: {state.dok}", "OK")
        state.target_steps["detect"] = "done"
    else:
        log_emit("target", "No package found.", "WARN")
        state.target_steps["detect"] = "error"


def tgt_backup():
    if not state.proj:
        return
    log_emit(
        "target", "╔══════════════════════════════════════════════════════════╗", "WARN"
    )
    log_emit(
        "target", "║  STARTING FULL SAFETY BACKUP                             ║", "WARN"
    )
    log_emit(
        "target", "╚══════════════════════════════════════════════════════════╝", "WARN"
    )
    try:
        proj = Path(state.proj)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bd = proj / "vision-rag-backups" / f"backup_{ts}"
        bd.mkdir(parents=True, exist_ok=True)
        for sub in ["volumes", "code", "docker", "docker-images"]:
            (bd / sub).mkdir(exist_ok=True)
        (bd / "restore-from-backup.sh").write_text(gen_restore())
        (bd / "PROJECT_DIR.txt").write_text(str(proj))
        prefix = get_prefix()
        (bd / "VOLUME_PREFIX.txt").write_text(prefix)
        log_emit("target", f"Backup dir: {bd}", "INFO")
        log_emit("target", f"Volume prefix: {prefix}", "INFO")
        # Images
        log_emit("target", "Backing up Docker images...", "WARN")
        try:
            r = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            imgs = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
            log_emit("target", f"Found {len(imgs)} images", "INFO")
            for img in imgs:
                safe = img.replace("/", "-").replace(":", "-") + ".tar.gz"
                out = bd / "docker-images" / safe
                try:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    ok, err = _docker_save_gzip(img, out, channel="target")
                    if ok:
                        log_emit("target", f"  ✓ {safe}", "OK")
                    else:
                        log_emit(
                            "target", f"  ✗ {img}: {err or 'docker save error'}", "WARN"
                        )
                except Exception as e:
                    log_emit("target", f"  ✗ {img}: {e}", "WARN")
        except Exception as e:
            log_emit("target", f"Could not list images: {e}", "WARN")
        # Volumes
        log_emit("target", "Backing up Docker volumes...", "WARN")
        _CRITICAL_VOLS = ["hf-cache", "qdrant_data", "ollama", "open-webui"]
        for suffix in _CRITICAL_VOLS:
            vol = f"{prefix}_{suffix}"
            try:
                res = subprocess.run(
                    ["docker", "volume", "inspect", vol], capture_output=True, timeout=5
                )
                if res.returncode != 0:
                    log_emit("target", f"  Skipped {vol} (not found)", "WARN")
                    continue
            except Exception:
                continue
            out = bd / "volumes" / f"{suffix}.tar.gz"
            log_emit("target", f"  Backing up {vol}...", "INFO")
            try:
                for tool in ["busybox", "alpine"]:
                    result = subprocess.run(
                        [
                            "docker",
                            "run",
                            "--rm",
                            "-v",
                            f"{vol}:/source:ro",
                            "-v",
                            f"{bd / 'volumes'}:/dest",
                            tool,
                            "tar",
                            "czf",
                            f"/dest/{suffix}.tar.gz",
                            "-C",
                            "/source",
                            ".",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2700,
                    )
                    if result.returncode == 0 and out.exists():
                        sz = out.stat().st_size
                        log_emit("target", f"    ✓ {suffix}.tar.gz ({hsize(sz)})", "OK")
                        break
                else:
                    log_emit("target", f"    ✗ Failed to backup {vol}", "ERROR")
            except Exception as e:
                log_emit("target", f"    ✗ {vol}: {e}", "ERROR")
        # Code
        log_emit("target", "Backing up project code...", "WARN")
        _IGNORE_DIRS = {
            "vision-rag-backups",
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "my-pdfs",
        }

        def _ignore_fn(d, contents):
            return [c for c in contents if c in _IGNORE_DIRS]

        code_dst = bd / "code"
        try:
            shutil.copytree(proj, code_dst, ignore=_ignore_fn, dirs_exist_ok=True)
            log_emit("target", "  ✓ Full project tree backed up", "OK")
        except Exception as e:
            log_emit(
                "target",
                f"  ✗ copytree failed: {e}, falling back to essential files",
                "WARN",
            )
            for fn in ["docker-compose.yml", "Dockerfile", ".env", ".env.example"]:
                src = proj / fn
                if src.exists():
                    shutil.copy2(src, code_dst)
                    log_emit("target", f"  ✓ {fn}", "OK")
        try:
            subprocess.run(
                [
                    "docker",
                    "ps",
                    "--format",
                    "table {{.Names}}\t{{.Image}}\t{{.Status}}",
                ],
                stdout=open(bd / "docker" / "containers-list.txt", "w"),
                timeout=10,
                check=True,
            )
            subprocess.run(
                ["docker", "volume", "ls", "--format", "{{.Name}}"],
                stdout=open(bd / "docker" / "volumes-list.txt", "w"),
                timeout=10,
                check=True,
            )
        except Exception:
            pass
        total = sum(f.stat().st_size for f in bd.rglob("*") if f.is_file())
        log_emit("target", "═══════════════════════════════════════", "OK")
        log_emit("target", "BACKUP COMPLETE", "OK")
        log_emit("target", f"Location: {bd}", "WARN")
        log_emit("target", f"Total Size: {hsize(total)}", "OK")
        log_emit("target", "═══════════════════════════════════════", "OK")
        state.bkp_path = str(bd)
        state.bkp_done = True
        state.target_steps["backup"] = "done"
    except Exception as e:
        log_emit("target", f"Backup failed: {e}", "ERROR")
        state.target_steps["backup"] = "error"


def tgt_import():
    dok = Path(state.dok)
    tgt = Path(state.proj)
    prefix = get_prefix()
    idir = dok / "images"
    if not idir.exists():
        log_emit("target", "No images/ on DOK!", "ERROR")
        return
    tars = sorted(idir.glob("*.tar.gz"))
    log_emit("target", f"Found {len(tars)} images to load", "INFO")
    loaded = 0
    for tb in tars:
        log_emit("target", f"Loading {tb.name}...", "INFO")
        try:
            proc = subprocess.Popen(
                ["docker", "load", "-i", str(tb)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            last_log = time.time()
            while proc.poll() is None:
                time.sleep(5)
                now = time.time()
                if now - last_log >= 15:
                    try:
                        sz = tb.stat().st_size
                    except OSError:
                        sz = 0
                    log_emit(
                        "target",
                        f"  ... loading {tb.name} ({hsize(sz)} image file)",
                        "INFO",
                    )
                    last_log = now
            stdout = (
                proc.stdout.read().decode(errors="replace").strip()
                if proc.stdout
                else ""
            )
            stderr = (
                proc.stderr.read().decode(errors="replace").strip()
                if proc.stderr
                else ""
            )
            if proc.returncode == 0:
                log_emit("target", f"  ✓ {tb.name}", "OK")
                loaded += 1
            else:
                log_emit("target", f"  ✗ {tb.name}: {stderr}", "ERROR")
        except Exception as e:
            log_emit("target", f"  ✗ {tb.name}: {e}", "ERROR")
    log_emit("target", f"Loaded {loaded}/{len(tars)} images", "OK")
    ef = tgt / ".env"
    if ef.exists():
        shutil.copy2(
            ef, tgt / f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        log_emit("target", "Backed up existing .env", "OK")
    sc = dok / "code"
    if not sc.exists():
        log_emit("target", "No code/ on DOK!", "ERROR")
        return
    log_emit("target", "Stopping containers before code copy...", "INFO")
    try:
        subprocess.run(
            ["docker", "compose", "down"],
            cwd=str(tgt),
            capture_output=True,
            text=True,
            timeout=60,
        )
        log_emit("target", "✓ Containers stopped", "OK")
    except Exception as e:
        log_emit(
            "target", f"docker compose down failed: {e} (continuing anyway)", "WARN"
        )
    log_emit("target", "Fixing file ownership...", "INFO")
    chown_path = str(tgt)
    if shutil.which("wsl"):
        _wsl_path = win2wsl(str(tgt))
        if _wsl_path:
            chown_path = _wsl_path
        chown = subprocess.run(
            ["wsl", "-e", "bash", "-c", f"sudo chown -R $(whoami): '{chown_path}'"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    else:
        chown = subprocess.run(
            [
                "sudo",
                "-n",
                "chown",
                "-R",
                f"{os.environ.get('USER', 'root')}:",
                chown_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    if chown.returncode != 0:
        log_emit(
            "target",
            f"⚠ Could not fix ownership (sudo needed): {chown.stderr.strip()[:200]}",
            "WARN",
        )
    else:
        log_emit("target", "✓ Ownership fixed", "OK")
    log_emit("target", "Copying updated code...", "INFO")
    _skip_names = {"vision-rag-backups", ".git", "__pycache__"}
    perm_errors = []
    for item in sc.iterdir():
        if item.name in _skip_names:
            continue
        di = tgt / item.name
        try:
            if item.is_dir():
                if di.exists():
                    shutil.rmtree(di)
                shutil.copytree(item, di)
            else:
                if di.exists():
                    di.unlink()
                with open(item, "rb") as fi, open(di, "wb") as fo:
                    fo.write(fi.read())
        except PermissionError:
            perm_errors.append(item.name)
            continue
        except Exception:
            continue
    if perm_errors:
        log_emit(
            "target",
            f"Permission denied on {len(perm_errors)} item(s): {', '.join(perm_errors[:5])}",
            "WARN",
        )
        log_emit(
            "target", "Enter your sudo password to fix ownership and retry.", "WARN"
        )
        state.need_sudo_pw = True
    log_emit("target", "Verifying all files (SHA256)...", "INFO")
    _skip_dirs = {
        "vision-rag-backups",
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "my-pdfs",
    }
    _verify_ok = 0
    _verify_fail = 0
    _verify_missing = 0
    _verify_perm = 0
    for src_f in sorted(sc.rglob("*")):
        if not src_f.is_file():
            continue
        rel = src_f.relative_to(sc)
        if any(p in _skip_dirs for p in rel.parts):
            continue
        tgt_f = tgt / rel
        if not tgt_f.exists():
            log_emit("target", f"  ✗ {rel} — missing on target", "ERROR")
            _verify_missing += 1
            continue
        try:
            src_hash = hashlib.sha256(src_f.read_bytes()).hexdigest()[:12]
            tgt_hash = hashlib.sha256(tgt_f.read_bytes()).hexdigest()[:12]
            if src_hash == tgt_hash:
                _verify_ok += 1
            else:
                log_emit(
                    "target",
                    f"  ✗ {rel} — hash mismatch (src={src_hash} tgt={tgt_hash})",
                    "ERROR",
                )
                _verify_fail += 1
        except PermissionError:
            log_emit("target", f"  ⚠ {rel} — permission denied (root-owned)", "WARN")
            _verify_perm += 1
        except Exception as e:
            log_emit("target", f"  ⚠ {rel} — verify error: {e}", "WARN")
    total = _verify_ok + _verify_fail + _verify_missing + _verify_perm
    if _verify_fail == 0 and _verify_missing == 0 and _verify_perm == 0:
        log_emit("target", f"✓ All {total} files verified (SHA256 match)", "OK")
    else:
        parts = []
        if _verify_fail:
            parts.append(f"{_verify_fail} mismatched")
        if _verify_missing:
            parts.append(f"{_verify_missing} missing")
        if _verify_perm:
            parts.append(f"{_verify_perm} permission denied")
        log_emit("target", f"⚠ {', '.join(parts)} — {_verify_ok}/{total} ok", "WARN")
    log_emit("target", "Code updated", "OK")
    ep = tgt / ".env"
    if ep.exists():
        content = ep.read_text()
        if "COMPOSE_PROJECT_NAME=" in content:
            lines = content.splitlines()
            nl = []
            for line in lines:
                if line.startswith("COMPOSE_PROJECT_NAME="):
                    nl.append(f"COMPOSE_PROJECT_NAME={prefix}")
                else:
                    nl.append(line)
            ep.write_text("\n".join(nl) + "\n")
            log_emit("target", f"Updated COMPOSE_PROJECT_NAME={prefix}", "OK")
        else:
            with open(ep, "a") as f:
                f.write(
                    f"\n# Docker Compose project name\nCOMPOSE_PROJECT_NAME={prefix}\n"
                )
            log_emit("target", f"Added COMPOSE_PROJECT_NAME={prefix}", "OK")
    else:
        ex = tgt / ".env.example"
        if ex.exists():
            shutil.copy2(ex, ep)
            open(ep, "a").write(f"\nCOMPOSE_PROJECT_NAME={prefix}\n")
            log_emit("target", "Created .env from .env.example", "OK")
    if _verify_fail > 0 or perm_errors:
        log_emit("target", "═══════════════════════════════════════", "WARN")
        log_emit("target", "IMPORT PARTIAL — some files need sudo to update", "WARN")
        log_emit("target", "═══════════════════════════════════════", "WARN")
    else:
        log_emit("target", "═══════════════════════════════════════", "OK")
        log_emit("target", "IMPORT COMPLETE", "OK")
        log_emit("target", "═══════════════════════════════════════", "OK")
    state.target_steps["import"] = "done"


def tgt_import_sudo(password):
    if not state.proj or not is_valid_project(state.proj):
        log_emit("target", "Cannot import: no valid project folder.", "ERROR")
        state.target_steps["import"] = "error"
        return
    dok = Path(state.dok)
    tgt = Path(state.proj)
    sc = dok / "code"
    prefix = get_prefix()
    log_emit("target", "Fixing file ownership with sudo...", "INFO")
    chown_path = str(tgt)
    if shutil.which("wsl"):
        _wsl_path = win2wsl(str(tgt))
        if _wsl_path:
            chown_path = _wsl_path
    if shutil.which("wsl"):
        chown = subprocess.run(
            [
                "wsl",
                "-e",
                "bash",
                "-c",
                f"echo '{password}' | sudo -S chown -R $(whoami): '{chown_path}'",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    else:
        chown = subprocess.run(
            [
                "sudo",
                "-S",
                "chown",
                "-R",
                f"{os.environ.get('USER', 'root')}:",
                chown_path,
            ],
            input=password,
            capture_output=True,
            text=True,
            timeout=30,
        )
    auth_fail = (
        "incorrect password" in chown.stderr.lower()
        or "sorry" in chown.stderr.lower()
        or "a password is required" in chown.stderr.lower()
    )
    if chown.returncode != 0 or auth_fail:
        log_emit(
            "target",
            f"sudo chown failed — wrong password?: {chown.stderr.strip()}",
            "ERROR",
        )
        state.need_sudo_pw = True
        state.target_steps["import"] = "need_sudo"
        return
    log_emit("target", "✓ Ownership fixed", "OK")
    log_emit("target", "Copying updated code...", "INFO")
    _skip_names = {"vision-rag-backups", ".git", "__pycache__"}
    for item in sc.iterdir():
        if item.name in _skip_names:
            continue
        di = tgt / item.name
        try:
            if item.is_dir():
                if di.exists():
                    shutil.rmtree(di)
                shutil.copytree(item, di)
            else:
                if di.exists():
                    di.unlink()
                with open(item, "rb") as fi, open(di, "wb") as fo:
                    fo.write(fi.read())
        except PermissionError as e:
            log_emit(
                "target",
                f"  ⚠ Permission denied: {item.name} ({e}), forcing removal with sudo...",
                "WARN",
            )
            try:
                _r = None
                if shutil.which("wsl"):
                    _wp = win2wsl(str(di)) or str(di)
                    _r = subprocess.run(
                        [
                            "wsl",
                            "-e",
                            "bash",
                            "-c",
                            f"echo '{password}' | sudo -S rm -rf '{_wp}'",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                else:
                    _r = subprocess.run(
                        ["bash", "-c", f"echo '{password}' | sudo -S rm -rf '{di}'"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                if not di.exists():
                    shutil.copytree(item, di)
                    log_emit(
                        "target", f"  ✓ {item.name} (fixed with sudo rm + copy)", "OK"
                    )
                else:
                    _detail = ""
                    if _r:
                        _detail = f" (rc={_r.returncode}"
                        if _r.stderr:
                            _detail += f", stderr={_r.stderr.strip()}"
                        _detail += ")"
                    log_emit(
                        "target",
                        f"  ✗ Could not remove {item.name}{_detail}, trying file-by-file copy...",
                        "WARN",
                    )
                    _fb_ok = 0
                    _fb_fail = 0
                    for src_f in item.rglob("*"):
                        if not src_f.is_file():
                            continue
                        rel = src_f.relative_to(item)
                        tgt_f = di / rel
                        try:
                            tgt_f.parent.mkdir(parents=True, exist_ok=True)
                            with open(src_f, "rb") as fi, open(tgt_f, "wb") as fo:
                                fo.write(fi.read())
                            _fb_ok += 1
                        except Exception as fe:
                            log_emit(
                                "target",
                                f"  ⚠ {rel}: {fe}",
                                "WARN",
                            )
                            _fb_fail += 1
                    if _fb_fail == 0:
                        log_emit(
                            "target",
                            f"  ✓ {item.name} ({_fb_ok} files copied file-by-file)",
                            "OK",
                        )
                    else:
                        log_emit(
                            "target",
                            f"  ⚠ {item.name} ({_fb_ok} ok, {_fb_fail} failed)",
                            "WARN",
                        )
            except Exception as e2:
                log_emit("target", f"  ✗ Failed to copy {item.name}: {e2}", "ERROR")
            continue
        except Exception as e:
            log_emit("target", f"  ⚠ Error copying {item.name}: {e}", "WARN")
            continue
    log_emit("target", "Verifying all files (SHA256)...", "INFO")
    _skip_dirs = {
        "vision-rag-backups",
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "my-pdfs",
    }
    _verify_ok = 0
    _verify_fail = 0
    _verify_missing = 0
    _verify_perm = 0
    for src_f in sorted(sc.rglob("*")):
        if not src_f.is_file():
            continue
        rel = src_f.relative_to(sc)
        if any(p in _skip_dirs for p in rel.parts):
            continue
        tgt_f = tgt / rel
        if not tgt_f.exists():
            log_emit("target", f"  ✗ {rel} — missing on target", "ERROR")
            _verify_missing += 1
            continue
        try:
            src_hash = hashlib.sha256(src_f.read_bytes()).hexdigest()[:12]
            tgt_hash = hashlib.sha256(tgt_f.read_bytes()).hexdigest()[:12]
            if src_hash == tgt_hash:
                _verify_ok += 1
            else:
                log_emit(
                    "target",
                    f"  ✗ {rel} — hash mismatch (src={src_hash} tgt={tgt_hash})",
                    "ERROR",
                )
                _verify_fail += 1
        except PermissionError:
            log_emit("target", f"  ⚠ {rel} — permission denied (root-owned)", "WARN")
            _verify_perm += 1
        except Exception as e:
            log_emit("target", f"  ⚠ {rel} — verify error: {e}", "WARN")
    total = _verify_ok + _verify_fail + _verify_missing + _verify_perm
    if _verify_fail == 0 and _verify_missing == 0 and _verify_perm == 0:
        log_emit("target", f"✓ All {total} files verified (SHA256 match)", "OK")
    else:
        parts = []
        if _verify_fail:
            parts.append(f"{_verify_fail} mismatched")
        if _verify_missing:
            parts.append(f"{_verify_missing} missing")
        if _verify_perm:
            parts.append(f"{_verify_perm} permission denied")
        log_emit("target", f"⚠ {', '.join(parts)} — {_verify_ok}/{total} ok", "WARN")
    log_emit("target", "Code updated", "OK")
    ep = tgt / ".env"
    if ep.exists():
        content = ep.read_text()
        if "COMPOSE_PROJECT_NAME=" in content:
            lines = content.splitlines()
            nl = []
            for line in lines:
                if line.startswith("COMPOSE_PROJECT_NAME="):
                    nl.append(f"COMPOSE_PROJECT_NAME={prefix}")
                else:
                    nl.append(line)
            ep.write_text("\n".join(nl) + "\n")
            log_emit("target", f"Updated COMPOSE_PROJECT_NAME={prefix}", "OK")
        else:
            with open(ep, "a") as f:
                f.write(
                    f"\n# Docker Compose project name\nCOMPOSE_PROJECT_NAME={prefix}\n"
                )
            log_emit("target", f"Added COMPOSE_PROJECT_NAME={prefix}", "OK")
    else:
        ex = tgt / ".env.example"
        if ex.exists():
            shutil.copy2(ex, ep)
            open(ep, "a").write(f"\nCOMPOSE_PROJECT_NAME={prefix}\n")
            log_emit("target", "Created .env from .env.example", "OK")
    if _verify_fail > 0 or _verify_missing > 0 or _verify_perm > 0:
        log_emit("target", "═══════════════════════════════════════", "WARN")
        log_emit("target", "IMPORT PARTIAL — some files could not be updated", "WARN")
        log_emit("target", "═══════════════════════════════════════", "WARN")
    else:
        log_emit("target", "═══════════════════════════════════════", "OK")
        log_emit("target", "IMPORT COMPLETE", "OK")
        log_emit("target", "═══════════════════════════════════════", "OK")
    state.need_sudo_pw = False
    state.sudo_pw = ""
    state.target_steps["import"] = "done"


def tgt_verify():
    proj = Path(state.proj)
    prefix = get_prefix()
    results = {}
    log_emit(
        "target", "╔══════════════════════════════════════════════════════════╗", "INFO"
    )
    log_emit(
        "target", "║  RUNNING AUTOMATED VERIFICATION                          ║", "INFO"
    )
    log_emit(
        "target", "╚══════════════════════════════════════════════════════════╝", "INFO"
    )
    log_emit("target", "Starting containers...", "INFO")
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=str(proj),
            capture_output=True,
            text=True,
            timeout=120,
        )
        log_emit("target", "✓ docker compose up -d", "OK")
        log_emit("target", "Waiting for services to start...", "INFO")
        for i in range(6):
            time.sleep(15)
            log_emit("target", f"  ... waiting ({(i + 1) * 15}s)", "INFO")
    except Exception as e:
        log_emit("target", f"docker compose up failed: {e}", "WARN")
    # Containers
    log_emit("target", "Checking container status...", "INFO")
    try:
        r = subprocess.run(
            [
                "docker",
                "compose",
                "ps",
                "--format",
                "table {{.Name}}\t{{.Image}}\t{{.Status}}",
            ],
            cwd=str(proj),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0 and "Up" in r.stdout:
            results["containers"] = (True, r.stdout[:400])
            log_emit("target", "✓ Containers running", "OK")
        else:
            results["containers"] = (False, "No containers showing 'Up'")
            log_emit("target", "⚠ Containers not up yet — retrying...", "WARN")
            time.sleep(15)
            r = subprocess.run(
                [
                    "docker",
                    "compose",
                    "ps",
                    "--format",
                    "table {{.Name}}\t{{.Image}}\t{{.Status}}",
                ],
                cwd=str(proj),
                capture_output=True,
                text=True,
                timeout=15,
            )
            if r.returncode == 0 and "Up" in r.stdout:
                results["containers"] = (True, r.stdout[:400])
                log_emit("target", "✓ Containers running (retry)", "OK")
    except Exception as e:
        results["containers"] = (False, str(e))
    # Pipeline (retry up to 6x)
    log_emit("target", "Checking pipeline (localhost:8082)...", "INFO")
    pipeline_ok = False
    for attempt in range(6):
        try:
            r = subprocess.run(
                ["curl", "-s", "http://localhost:8082/status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                results["pipeline_status"] = (True, r.stdout[:300])
                log_emit("target", "✓ Pipeline responded", "OK")
                pipeline_ok = True
                break
        except Exception:
            pass
        log_emit(
            "target",
            f"  ... pipeline not ready, waiting 30s (attempt {attempt + 1}/6)",
            "INFO",
        )
        time.sleep(30)
    if not pipeline_ok:
        results["pipeline_status"] = (False, "No response")
    # ColQwen2 (retry up to 6x, check pipeline endpoint + logs)
    log_emit("target", "Checking for 'ready' in pipeline logs...", "INFO")
    colqwen_ok = False
    for attempt in range(6):
        try:
            r = subprocess.run(
                ["curl", "-s", "http://localhost:8082/status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                results["colqwen_ready"] = (True, "Pipeline ready")
                log_emit("target", "✓ ColQwen2 pipeline ready", "OK")
                colqwen_ok = True
                break
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["docker", "logs", "--tail", "100", "open-webui-pipelines"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logs = r.stdout if r.returncode == 0 else ""
            if "ready" in logs.lower():
                results["colqwen_ready"] = (True, "Pipeline ready")
                log_emit("target", "✓ ColQwen2 pipeline ready", "OK")
                colqwen_ok = True
                break
        except Exception:
            pass
        log_emit(
            "target",
            f"  ... ColQwen2 still loading, waiting 30s (attempt {attempt + 1}/6)",
            "INFO",
        )
        time.sleep(30)
    if not colqwen_ok:
        results["colqwen_ready"] = (False, "Still loading")
        log_emit("target", "⚠ ColQwen2 still loading (may need more time)", "WARN")
    # Ollama
    log_emit("target", "Checking Ollama models...", "INFO")
    try:
        r = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            models = [m.get("name", "?") for m in data.get("models", [])]
            results["ollama_models"] = (True, f"{len(models)} models")
            log_emit("target", f"✓ Ollama: {len(models)} models", "OK")
        else:
            results["ollama_models"] = (False, "No response")
    except Exception as e:
        results["ollama_models"] = (False, str(e))
    # hf-cache
    log_emit("target", "Checking hf-cache volume...", "INFO")
    try:
        vol = f"{prefix}_hf-cache"
        r = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{vol}:/vol:ro",
                "busybox",
                "sh",
                "-c",
                "find /vol -type f | wc -l",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        count = r.stdout.strip()
        if r.returncode == 0 and count.isdigit() and int(count) > 0:
            results["hf_cache"] = (True, f"{count} files")
            log_emit("target", f"✓ hf-cache: {count} files", "OK")
        else:
            results["hf_cache"] = (False, "Empty")
    except Exception as e:
        results["hf_cache"] = (False, str(e))
    # .env
    log_emit("target", "Checking .env configuration...", "INFO")
    ep = proj / ".env"
    if ep.exists():
        content = ep.read_text()
        if "COMPOSE_PROJECT_NAME=" in content:
            for line in content.splitlines():
                if line.startswith("COMPOSE_PROJECT_NAME="):
                    val = line.split("=", 1)[-1].strip()
                    results["env_ok"] = (True, f"COMPOSE_PROJECT_NAME={val}")
                    log_emit("target", f"✓ .env configured ({val})", "OK")
                    break
        else:
            results["env_ok"] = (False, "Missing COMPOSE_PROJECT_NAME")
    else:
        results["env_ok"] = (False, ".env not found")
    # Volumes
    log_emit("target", "Checking critical volumes...", "INFO")
    missing = []
    for suffix in ["hf-cache", "qdrant_data", "ollama", "open-webui"]:
        vol = f"{prefix}_{suffix}"
        try:
            r = subprocess.run(
                ["docker", "volume", "inspect", vol], capture_output=True, timeout=5
            )
            if r.returncode != 0:
                missing.append(vol)
        except Exception:
            missing.append(vol)
    if missing:
        results["volumes_ok"] = (False, f"Missing: {', '.join(missing)}")
        log_emit("target", f"✗ Missing: {', '.join(missing)}", "ERROR")
    else:
        results["volumes_ok"] = (True, "All present")
        log_emit("target", "✓ All critical volumes present", "OK")
    state.verify = results
    okc = sum(1 for v in results.values() if v[0])
    log_emit(
        "target",
        f"Verification: {okc}/{len(results)} passed",
        "OK" if okc == len(results) else "WARN",
    )
    state.target_steps["verify"] = "done"


def emergency_restore():
    if not state.bkp_path or not Path(state.bkp_path).exists():
        return False
    script = Path(state.bkp_path) / "restore-from-backup.sh"
    if not script.exists():
        return False
    try:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            log_emit("target", line.rstrip(), "INFO")
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        log_emit("target", f"Restore failed: {e}", "ERROR")
        return False


# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vision RAG Migration Wizard v4.0</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
--bg1:#0a0d12;--bg2:#0f1219;--bg3:#141820;
--card:#161b24;--card2:#1c2230;--hover:#1e2536;
--input:#111620;--input-b:#1a2030;
--accent:#5b9aff;--ac2:#7db4ff;--ac3:#3d7be0;
--ag1:#4a8af4;--ag2:#7c5cf5;
--ok:#34d399;--ok2:#2ab385;--okg1:#22c997;--okg2:#34d399;
--warn:#fbbf24;--warn2:#f59e0b;
--err:#f43f5e;--err2:#e11d48;
--txt:#f0f4f8;--txt2:#94a3b8;--mute:#475569;
--border:#1e293b;--b2:#334155;
}
html,body{height:100%;background:var(--bg1);color:var(--txt);font-family:'Segoe UI','Inter',system-ui,sans-serif;font-size:14px;overflow:hidden}
a{color:var(--ac2);text-decoration:none}

/* Layout */
.app{display:flex;flex-direction:column;height:100vh}
.header{height:68px;background:linear-gradient(180deg,var(--bg3),var(--bg2));border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 28px;gap:14px;flex-shrink:0;position:relative;z-index:10}
.header::after{content:'';position:absolute;bottom:-2px;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--ag1),var(--ag2),var(--okg1),transparent)}
.logo-icon{color:var(--ac2);font-size:30px;font-weight:bold}
.logo-title{font-size:18px;font-weight:800;letter-spacing:-0.5px}
.logo-sub{font-size:18px;font-weight:300;color:var(--txt2)}
.ver{color:var(--mute);font-size:10px;background:var(--card);padding:3px 10px;border-radius:6px;border:1px solid var(--border);letter-spacing:1px;font-weight:600}
.header-spacer{flex:1}
.header-btn{background:var(--card2);color:var(--txt2);border:1px solid var(--border);padding:8px 18px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:500;transition:all .2s}
.header-btn:hover{background:var(--hover);border-color:var(--b2);color:var(--txt)}

.body{display:flex;flex:1;overflow:hidden}

/* Sidebar */
.sidebar{width:270px;background:linear-gradient(180deg,var(--bg2),var(--bg3));border-right:1px solid var(--border);display:flex;flex-direction:column;padding:20px 0;flex-shrink:0}
.sidebar-label{color:var(--mute);font-size:10px;font-weight:700;letter-spacing:1.5px;padding:0 20px 10px}
.sidebar-btn{background:transparent;color:var(--txt2);text-align:left;padding:13px 24px;border:none;border-left:3px solid transparent;font-size:13px;font-weight:500;cursor:pointer;transition:all .2s;width:100%;font-family:inherit}
.sidebar-btn:hover{background:var(--hover);color:var(--txt);border-left-color:var(--b2)}
.sidebar-btn.active{background:rgba(91,154,255,.08);color:var(--ac2);font-weight:600;border-left-color:var(--accent)}
.sidebar-spacer{flex:1}
.safety-notice{background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.15);border-radius:10px;margin:16px;padding:14px 16px}
.safety-title{color:var(--warn);font-weight:700;font-size:12px;letter-spacing:.3px;margin-bottom:6px}
.safety-text{color:var(--warn);font-size:11px;line-height:1.5;opacity:.85}

/* Content */
.content{flex:1;overflow-y:auto;overflow-x:hidden;padding:36px 48px}
.content::-webkit-scrollbar{width:6px}
.content::-webkit-scrollbar-track{background:var(--bg2)}
.content::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
.content::-webkit-scrollbar-thumb:hover{background:var(--mute)}

/* Status bar */
.statusbar{height:38px;background:var(--bg2);border-top:1px solid var(--border);display:flex;align-items:center;padding:0 28px;font-size:12px;color:var(--mute);flex-shrink:0;gap:8px}
.status-dot{width:6px;height:6px;border-radius:50%;background:var(--mute)}
.status-dot.active{background:var(--ok);box-shadow:0 0 6px var(--ok)}
.status-badge{background:rgba(71,85,105,.1);border:1px solid rgba(71,85,105,.2);border-radius:10px;padding:3px 12px;font-size:11px;font-weight:600;color:var(--mute);letter-spacing:.5px}
.status-badge.source{background:rgba(91,154,255,.1);border-color:rgba(91,154,255,.2);color:var(--accent)}
.status-badge.target{background:rgba(52,211,153,.1);border-color:rgba(52,211,153,.2);color:var(--ok)}

/* Section headers */
.sechead{margin-bottom:20px}
.sechead h1{font-size:24px;font-weight:800;letter-spacing:-0.3px;margin-bottom:6px}
.sechead-line{height:2px;width:280px;background:linear-gradient(90deg,var(--ag1),var(--ag2),transparent);border-radius:1px;margin-bottom:8px}
.sechead p{color:var(--txt2);font-size:13px;line-height:1.5}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px;margin-bottom:16px;transition:border-color .3s}
.card:hover{border-color:var(--b2)}
.card-label{color:var(--txt2);font-size:12px;font-weight:600;letter-spacing:.5px;margin-bottom:12px}

/* Step cards */
.step-card{background:var(--card);border:1px solid var(--border);border-radius:14px;margin-bottom:12px;overflow:hidden;transition:border-color .3s}
.step-card:hover{border-color:var(--b2)}
.step-bar{height:3px;background:linear-gradient(90deg,var(--ag1),var(--ag2))}
.step-bar.ok{background:linear-gradient(90deg,var(--okg1),var(--okg2))}
.step-bar.warn{background:linear-gradient(90deg,var(--warn2),var(--warn))}
.step-bar.err{background:linear-gradient(90deg,var(--err2),var(--err))}
.step-inner{display:flex;align-items:center;padding:18px 20px;gap:18px}
.step-badge{width:46px;height:46px;border-radius:23px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:17px;flex-shrink:0;background:linear-gradient(135deg,rgba(74,138,244,.2),rgba(124,92,245,.2));color:var(--ac2);border:2px solid rgba(91,154,255,.27);transition:all .3s}
.step-badge.ok{background:rgba(52,211,153,.1);color:var(--ok);border-color:var(--ok)}
.step-badge.warn{background:rgba(251,191,36,.2);color:var(--warn);border-color:var(--warn)}
.step-badge.err{background:rgba(244,63,94,.1);color:var(--err);border-color:var(--err)}
.step-info{flex:1;min-width:0}
.step-title{font-weight:700;font-size:14px;letter-spacing:.2px;margin-bottom:3px}
.step-title.ok{color:var(--ok)}
.step-title.err{color:var(--err)}
.step-desc{color:var(--txt2);font-size:12px;line-height:1.5}
.step-btn{background:linear-gradient(90deg,var(--ag1),var(--ag2));color:#fff;border:none;padding:10px 28px;border-radius:10px;font-weight:600;font-size:13px;cursor:pointer;transition:all .2s;white-space:nowrap;min-width:110px;font-family:inherit}
.step-btn:hover{filter:brightness(1.15)}
.step-btn:disabled{background:var(--card);color:var(--mute);border:1px solid var(--border);cursor:not-allowed;filter:none}
.step-btn.done{background:rgba(52,211,153,.1);color:var(--ok);border:1px solid rgba(52,211,153,.2)}
.step-btn.err-btn{background:rgba(244,63,94,.1);color:var(--err);border:1px solid rgba(244,63,94,.2)}

/* Form elements */
.input-row{display:flex;gap:10px;align-items:center;margin-bottom:12px}
.text-input{flex:1;background:var(--input);border:1.5px solid var(--border);padding:12px 16px;border-radius:10px;color:var(--txt);font-size:13px;font-family:inherit;transition:border-color .2s}
.text-input:focus{outline:none;border-color:var(--accent);background:var(--input-b)}
.text-input::placeholder{color:var(--mute)}
.btn-secondary{background:var(--card2);color:var(--txt2);border:1px solid var(--border);padding:10px 18px;border-radius:10px;font-size:13px;cursor:pointer;transition:all .2s;white-space:nowrap;font-family:inherit}
.btn-secondary:hover{background:var(--hover);border-color:var(--b2);color:var(--txt)}
.status-row{display:flex;align-items:center;gap:8px;font-size:13px}
.status-icon{font-size:16px;font-weight:bold}
.status-icon.mute{color:var(--mute)}.status-icon.ok{color:var(--ok)}.status-icon.warn{color:var(--warn)}
.status-text{font-weight:500}.status-text.mute{color:var(--mute)}.status-text.ok{color:var(--ok)}.status-text.warn{color:var(--warn)}

/* Buttons */
.btn-primary{background:linear-gradient(90deg,var(--ag1),var(--ag2));color:#fff;border:none;padding:12px 32px;border-radius:10px;font-weight:600;font-size:14px;cursor:pointer;transition:all .2s;letter-spacing:.3px;font-family:inherit}
.btn-primary:hover{filter:brightness(1.15)}
.btn-danger{background:linear-gradient(90deg,var(--err2),var(--err));color:#fff;border:none;padding:12px 28px;border-radius:10px;font-weight:600;cursor:pointer;transition:all .2s;font-family:inherit}
.btn-danger:hover{filter:brightness(1.15)}
.btn-danger:disabled{background:var(--card);color:var(--mute);border:1px solid var(--border);cursor:not-allowed;filter:none}

/* Log area */
.log-card{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-bottom:16px}
.log-header{display:flex;align-items:center;padding:16px 20px 12px;gap:8px}
.log-icon{color:var(--ac2);font-size:14px;font-weight:bold}
.log-title{font-weight:700;font-size:13px;letter-spacing:.3px;flex:1}
.log-body{background:var(--bg2);border-top:1px solid var(--border);padding:14px;font-family:'JetBrains Mono','Fira Code',Consolas,monospace;font-size:11.5px;line-height:1.7;min-height:180px;max-height:400px;overflow-y:auto;color:#94a3b8}
.log-body::-webkit-scrollbar{width:5px}
.log-body::-webkit-scrollbar-thumb{background:var(--b2);border-radius:3px}
.log-entry{margin:1px 0}
.log-ts{color:var(--mute);font-size:10px}
.log-level{font-weight:700;font-size:11px}
.log-level.INFO{color:#8bb4e0}.log-level.OK{color:var(--ok)}.log-level.WARN{color:var(--warn)}.log-level.ERROR{color:var(--err)}.log-level.CMD{color:var(--ac2)}
.log-msg{color:#b8c9db}

/* Welcome page */
.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100%;padding:20px 0}
.welcome-hero{font-size:64px;color:var(--ac2);margin-bottom:8px}
.welcome-title{font-size:36px;font-weight:800;letter-spacing:-1px;margin-bottom:8px}
.welcome-sub{color:var(--txt2);font-size:15px;font-weight:300;margin-bottom:12px}
.welcome-line{height:2px;width:400px;max-width:90%;background:linear-gradient(90deg,transparent,var(--ag1),var(--ag2),var(--okg1),transparent);border-radius:1px;margin-bottom:32px}
.welcome-cards{display:grid;grid-template-columns:1fr 1fr;gap:24px;width:100%;max-width:900px}
.welcome-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:36px;text-align:center;transition:all .3s}
.welcome-card:hover{border-color:rgba(91,154,255,.33);background:var(--card2)}
.welcome-card.target:hover{border-color:rgba(52,211,153,.33)}
.wc-icon-wrap{width:72px;height:72px;border-radius:36px;display:flex;align-items:center;justify-content:center;margin:0 auto 14px;font-size:32px}
.wc-icon-wrap.source{background:linear-gradient(135deg,rgba(74,138,244,.13),rgba(124,92,245,.13));border:2px solid rgba(91,154,255,.2)}
.wc-icon-wrap.target{background:linear-gradient(135deg,rgba(34,201,151,.13),rgba(52,211,153,.13));border:2px solid rgba(52,211,153,.2)}
.wc-title{font-weight:800;font-size:16px;letter-spacing:1.5px;margin-bottom:10px}
.wc-title.source{color:var(--ac2)}.wc-title.target{color:var(--ok)}
.wc-divider{height:1px;background:var(--border);margin-bottom:14px}
.wc-desc{color:var(--txt2);font-size:13px;line-height:1.6;margin-bottom:16px}
.wc-btn{min-width:220px;min-height:50px;font-size:14px}

/* Warning banner */
.warn-banner{background:rgba(244,63,94,.06);border:1px solid rgba(244,63,94,.16);border-radius:14px;padding:18px 24px;display:flex;align-items:center;gap:14px;margin-bottom:16px}
.warn-icon-wrap{width:44px;height:44px;border-radius:22px;background:rgba(244,63,94,.1);border:1px solid rgba(244,63,94,.2);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.warn-icon{color:var(--err);font-size:20px}
.warn-title{color:var(--err);font-weight:800;font-size:14px;letter-spacing:.5px;margin-bottom:4px}
.warn-text{color:var(--err);font-size:12px;line-height:1.5;opacity:.85}

/* Check list */
.check-item{display:flex;align-items:center;gap:8px;padding:4px 0}
.check-icon{color:var(--ok);font-size:13px;font-weight:bold;width:16px}
.check-text{color:var(--txt2);font-size:12px}

/* Checkbox */
.checkbox-row{display:flex;align-items:center;gap:8px;padding:8px 0;color:var(--txt2);font-size:12px}
.checkbox-row input[type=checkbox]{width:18px;height:18px;border-radius:5px;accent-color:var(--accent);cursor:pointer}

/* Verify results */
.verify-card{background:var(--card);border:1px solid rgba(52,211,153,.2);border-radius:14px;padding:24px;margin-bottom:16px}
.verify-title{font-weight:700;color:var(--ok);font-size:14px;letter-spacing:.3px;margin-bottom:10px}
.verify-item{margin:4px 0;font-size:12px;line-height:1.7}

/* Modal */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);display:flex;align-items:center;justify-content:center;z-index:100;opacity:0;pointer-events:none;transition:opacity .3s}
.modal-overlay.active{opacity:1;pointer-events:auto}
.modal{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:28px;max-width:700px;width:90%;max-height:80vh;overflow-y:auto}
.modal h2{font-size:20px;font-weight:700;margin-bottom:16px}
.modal pre{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px;font-family:'JetBrains Mono',Consolas,monospace;font-size:11.5px;overflow-x:auto;max-height:300px;color:var(--txt2)}
.modal-close{float:right;background:var(--card2);color:var(--txt2);border:1px solid var(--border);padding:6px 16px;border-radius:8px;cursor:pointer;font-size:12px}
.modal-close:hover{background:var(--hover);color:var(--txt)}

/* Continue button row */
.continue-row{display:flex;justify-content:flex-end;margin-top:16px}

/* Emergency row */
.emergency-row{display:flex;justify-content:flex-end;margin-bottom:16px}

/* Hidden */
.hidden{display:none!important}

/* Responsive */
@media(max-width:900px){.welcome-cards{grid-template-columns:1fr}.sidebar{width:220px}.content{padding:24px}}
@media(max-width:700px){.sidebar{display:none}.content{padding:16px}}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <span class="logo-icon">◈</span>
    <span class="logo-title">Vision RAG</span>
    <span class="logo-sub">Migration Wizard</span>
    <span class="ver">v4.0</span>
    <div class="header-spacer"></div>
    <button class="header-btn" onclick="showTools()">⚙  Tools</button>
    <button class="header-btn" onclick="showHelp()">?  Help</button>
  </div>

  <div class="body">
    <div class="sidebar">
      <div class="sidebar-label">NAVIGATION</div>
      <button class="sidebar-btn active" data-page="welcome" onclick="goPage('welcome')">🏠   Welcome</button>
      <button class="sidebar-btn" data-page="source" onclick="goPage('source')">📤   Source Export</button>
      <button class="sidebar-btn" data-page="target" onclick="goPage('target')">📥   Target Update</button>
      <div class="sidebar-spacer"></div>
      <div class="safety-notice">
        <div class="safety-title">⚠  Safety Notice</div>
        <div class="safety-text">Always complete the full backup before importing. Your data is irreplaceable.</div>
      </div>
    </div>

    <div class="content" id="content"></div>
  </div>

  <div class="statusbar">
    <div class="status-dot" id="statusDot"></div>
    <span id="statusText">Ready</span>
    <div style="flex:1"></div>
    <span class="status-badge" id="statusBadge">Idle</span>
  </div>
</div>

<!-- Tools Modal -->
<div class="modal-overlay" id="toolsModal">
  <div class="modal">
    <button class="modal-close" onclick="closeModal('toolsModal')">Close</button>
    <h2>⚙  Tools & Cleanup</h2>
    <div id="toolsContent"><p style="color:var(--txt2)">Loading...</p></div>
  </div>
</div>

<!-- Help Modal -->
<div class="modal-overlay" id="helpModal">
  <div class="modal">
    <button class="modal-close" onclick="closeModal('helpModal')">Close</button>
    <h2>Vision RAG Migration Wizard</h2>
    <p style="color:var(--txt2);line-height:1.7;margin-bottom:12px">
    <b>SOURCE MACHINE:</b><br>
    1. Select project folder<br>
    2. Confirm DOK destination<br>
    3. Run all 4 export steps<br>
    4. Eject DOK<br><br>
    <b>TARGET MACHINE:</b><br>
    1. Plug in DOK<br>
    2. Detect package<br>
    3. Run MANDATORY full backup<br>
    4. Import update<br>
    5. Auto-verify<br><br>
    If anything fails, use Emergency Restore (requires typed confirmation).</p>
  </div>
</div>

<!-- Confirm Modal -->
<div class="modal-overlay" id="confirmModal">
  <div class="modal" style="max-width:500px">
    <h2 id="confirmTitle">Confirm</h2>
    <p id="confirmText" style="color:var(--txt2);line-height:1.7;margin-bottom:20px"></p>
    <div style="display:flex;gap:12px;justify-content:flex-end">
      <button class="btn-secondary" onclick="closeModal('confirmModal')">Cancel</button>
      <button class="btn-primary" id="confirmOk" onclick="">Confirm</button>
    </div>
  </div>
</div>

<!-- Restore Confirm Modal -->
<div class="modal-overlay" id="restoreModal">
  <div class="modal" style="max-width:500px">
    <h2 style="color:var(--err)">🚨 Emergency Restore</h2>
    <p style="color:var(--txt2);line-height:1.7;margin-bottom:16px">This will REVERT everything to the backup state. This CANNOT be undone.</p>
    <p style="color:var(--txt2);margin-bottom:12px">Type <b style="color:var(--err)">RESTORE</b> (all caps) to confirm:</p>
    <input type="text" class="text-input" id="restoreInput" placeholder="Type RESTORE here..." style="width:100%;margin-bottom:16px">
    <div style="display:flex;gap:12px;justify-content:flex-end">
      <button class="btn-secondary" onclick="closeModal('restoreModal')">Cancel</button>
      <button class="btn-danger" id="restoreOk" onclick="doRestore()">Restore</button>
    </div>
  </div>
</div>

<!-- Directory Browser Modal -->
<div class="modal-overlay" id="dirBrowserModal">
  <div class="modal" style="max-width:600px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
      <h2 style="flex:1;margin:0">📁 Browse Folders</h2>
      <button class="modal-close" onclick="closeModal('dirBrowserModal')">Cancel</button>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:12px">
      <input type="text" class="text-input" id="dirBrowserPath" value="/" onkeydown="if(event.key==='Enter')dirBrowserNav(this.value)">
      <button class="btn-secondary" onclick="dirBrowserNav(document.getElementById('dirBrowserPath').value)">Go</button>
    </div>
    <div id="dirBrowserContent" style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;max-height:350px;overflow-y:auto;padding:4px"></div>
    <div style="margin-top:16px;display:flex;justify-content:flex-end;gap:8px">
      <button class="btn-secondary" onclick="closeModal('dirBrowserModal')">Cancel</button>
      <button class="btn-primary" id="dirBrowserSelect" onclick="dirBrowserConfirm()" disabled>Select This Folder</button>
    </div>
  </div>
</div>

<script>
let dirBrowserTarget='', dirBrowserSelected='';

function openDirBrowser(target){
  dirBrowserTarget=target;
  dirBrowserSelected='';
  const currentVal=target==='src'?document.getElementById('srcProjPath')?.value||'':target==='tgt'?document.getElementById('tgtProjPath')?.value||'':'';
  document.getElementById('dirBrowserPath').value=currentVal||'/';
  dirBrowserNav(currentVal||'/');
  document.getElementById('dirBrowserModal').classList.add('active');
}

async function dirBrowserNav(path){
  document.getElementById('dirBrowserPath').value=path;
  const r=await api('/api/list-dir?path='+encodeURIComponent(path));
  const c=document.getElementById('dirBrowserContent');
  if(r.error){c.innerHTML='<p style="color:var(--err);padding:12px">Error: '+escHtml(r.error)+'</p>';return}
  let html='';
  const parent=Path_parent(r.path);
  if(parent!==r.path) html+=`<div class="dir-item" onclick="dirBrowserNav('${escHtml(parent)}')" style="cursor:pointer;padding:10px 14px;border-radius:8px;color:var(--txt2);display:flex;align-items:center;gap:8px" onmouseover="this.style.background='var(--hover)'" onmouseout="this.style.background='transparent'"><span>📂</span> <span>..</span></div>`;
  for(const d of (r.dirs||[])){
    const valid=d.valid?'<span style="color:var(--ok);font-size:11px;font-weight:700;margin-left:8px">✓ Valid</span>':'';
    const sel=dirBrowserSelected===d.path?'background:var(--accent)15;':'';
    html+=`<div class="dir-item" onclick="dirBrowserSelectDir('${escHtml(d.path)}')" ondblclick="dirBrowserNav('${escHtml(d.path)}')" style="cursor:pointer;padding:10px 14px;border-radius:8px;color:var(--txt);display:flex;align-items:center;gap:8px;${sel}" onmouseover="this.style.background='var(--hover)'" onmouseout="this.style.background='${sel?'var(--accent)15':'transparent'}'"><span>📁</span> <span>${escHtml(d.name)}</span>${valid}</div>`;
  }
  if(!r.dirs||r.dirs.length===0) html+='<p style="color:var(--mute);padding:12px">No subdirectories found</p>';
  c.innerHTML=html;
}

function dirBrowserSelectDir(path){
  dirBrowserSelected=path;
  document.getElementById('dirBrowserSelect').disabled=false;
  dirBrowserNav(path.replace(/\/[^/]*$/,'')||'/');
  const items=document.querySelectorAll('.dir-item');
  items.forEach(i=>{i.style.background='transparent'});
}

async function dirBrowserConfirm(){
  if(!dirBrowserSelected) return;
  const r=await api('/api/set-project','POST',{path:dirBrowserSelected});
  if(dirBrowserTarget==='src'){
    document.getElementById('srcProjPath').value=dirBrowserSelected;
    updateProjStatusSrc(r.valid,dirBrowserSelected);
  } else if(dirBrowserTarget==='tgt'){
    document.getElementById('tgtProjPath').value=dirBrowserSelected;
    updateProjStatusTgt(r.valid,dirBrowserSelected);
  }
  closeModal('dirBrowserModal');
}

function Path_parent(p){const parts=p.replace(/\/$/,'').split('/');parts.pop();return parts.join('/')||'/'}
let currentPage='welcome', appMode='', projectValid=false, sourceInited=false, targetInited=false;
let sourceSteps={disc:'pending',exp:'pending',code:'pending',bun:'pending'};
let targetSteps={detect:'pending',backup:'pending',import:'pending',verify:'pending'};
let savedSrcProj='', savedTgtProj='', savedDok='';

// ── Navigation ──
function goPage(page){
  currentPage=page;
  document.querySelectorAll('.sidebar-btn').forEach(b=>{
    b.classList.toggle('active',b.dataset.page===page);
  });
  renderPage();
}

// ── API helpers ──
async function api(url,method='GET',body=null){
  const opts=method!=='GET'?{method,body:body?JSON.stringify(body):undefined,headers:body?{'Content-Type':'application/json'}:{}}:{};
  try{const r=await fetch(url,opts);if(!r.ok)console.error('api',url,'status',r.status);return await r.json()}catch(e){console.error('api error',url,e);return{error:e.message}}
}

// ── Status bar ──
function setStatus(text,mode){
  document.getElementById('statusText').textContent=text;
  const dot=document.getElementById('statusDot');
  const badge=document.getElementById('statusBadge');
  dot.className='status-dot'+(mode?' active':'');
  if(mode==='source'){badge.textContent='Source Export';badge.className='status-badge source'}
  else if(mode==='target'){badge.textContent='Target Update';badge.className='status-badge target'}
  else{badge.textContent='Idle';badge.className='status-badge'}
}

// ── Render pages ──
function saveInputs(){
  const s=document.getElementById('srcProjPath'); if(s) savedSrcProj=s.value;
  const t=document.getElementById('tgtProjPath'); if(t) savedTgtProj=t.value;
  const d=document.getElementById('dokPath'); if(d) savedDok=d.value;
}
function restoreInputs(){
  const s=document.getElementById('srcProjPath'); if(s&&savedSrcProj) s.value=savedSrcProj;
  const t=document.getElementById('tgtProjPath'); if(t&&savedTgtProj) t.value=savedTgtProj;
  const d=document.getElementById('dokPath'); if(d&&savedDok) d.value=savedDok;
}
function renderPage(){
  saveInputs();
  const c=document.getElementById('content');
  if(currentPage==='welcome') c.innerHTML=renderWelcome();
  else if(currentPage==='source'){
    c.innerHTML=renderSource();
    restoreInputs();
    if(!sourceInited){sourceInited=true; setTimeout(()=>{autoDetectSrc();rescanDok();setTimeout(()=>startLogStream('source'),500);},100);}
    else setTimeout(()=>startLogStream('source'),50);
  }
  else if(currentPage==='target'){
    c.innerHTML=renderTarget();
    restoreInputs();
    if(!targetInited){targetInited=true; setTimeout(()=>{autoDetectTgt();setTimeout(()=>startLogStream('target'),500);},100);}
    else setTimeout(()=>startLogStream('target'),50);
  }
}

function renderWelcome(){
  return `<div class="welcome">
    <div class="welcome-hero">◈</div>
    <div class="welcome-title">Vision RAG Migration Wizard</div>
    <div class="welcome-sub">Zero-mistake export and update for offline machines</div>
    <div class="welcome-line"></div>
    <div class="welcome-cards">
      <div class="welcome-card">
        <div class="wc-icon-wrap source">🖥️</div>
        <div class="wc-title source">SOURCE MACHINE</div>
        <div class="wc-divider"></div>
        <div class="wc-desc">Export Docker images, project code, and bundle the wizard onto your DOK.<br>The target computer gets everything in one self-contained folder.</div>
        <button class="btn-primary wc-btn" onclick="chooseMode('source')">Start Export  →</button>
      </div>
      <div class="welcome-card target">
        <div class="wc-icon-wrap target">🛡️</div>
        <div class="wc-title target">TARGET MACHINE</div>
        <div class="wc-divider"></div>
        <div class="wc-desc">Apply a safe update from the DOK.<br>Mandatory full backup → Import → Auto verification.<br>One-click restore if anything goes wrong.</div>
        <button class="btn-primary wc-btn" onclick="chooseMode('target')">Start Update  →</button>
      </div>
    </div>
  </div>`;
}

function renderSource(){
  const ss=sourceSteps;
  return `<div class="sechead"><h1>Source Machine — Export to DOK</h1><div class="sechead-line"></div><p>Select the project on <em>this</em> machine, choose a DOK destination, then run the export steps.</p></div>
  <div class="card">
    <div class="card-label">📂  Project Folder Path (on this source machine)</div>
    <div class="input-row">
      <input type="text" class="text-input" id="srcProjPath" value="${escHtml(savedSrcProj)}" placeholder="Auto-detecting..." oninput="onProjInputSrc(this.value)">
      <button class="btn-secondary" onclick="autoDetectSrc()">🔍 Auto-Detect</button>
      <button class="btn-secondary" onclick="openDirBrowser('src')">📁 Browse</button>
      <button class="btn-secondary" onclick="pasteWinPathSrc()">📋 Paste Windows Path</button>
    </div>
    <div class="status-row" id="srcProjStatus"><span class="status-icon mute">○</span><span class="status-text mute">No project selected</span></div>
  </div>
  <div class="card">
    <div class="card-label"> DRIVE  DOK Destination — export goes into a dated subfolder (e.g. VisionRAG_Update_2026-05-30)</div>
    <div class="input-row">
      <input type="text" class="text-input" id="dokPath" value="${escHtml(savedDok)}" placeholder="e.g. /mnt/f/VisionRAG_Update">
      <button class="btn-secondary" onclick="rescanDok()">&#x1F504; Rescan Drives</button>
      <button class="btn-secondary" onclick="mountDrive()">&#x1F4BE; Mount Drive</button>
    </div>
    <div id="mountStatus" style="margin-top:4px;font-size:12px;color:var(--txt2)"></div>
  </div>
  </div>
  ${stepCard(1,'Discover Docker Images','Parse docker-compose.yml to identify all images.','disc',ss.disc,'srcDisc()')}
  ${stepCard(2,'Export Docker Images','Save each image as .tar.gz. May take 10–30 min.','exp',ss.exp,'srcExp()',ss.disc!=='done')}
  ${stepCard(3,'Export Project Code','Copy code, excluding .git, caches, venvs.','code',ss.code,'srcCode()')}
  ${stepCard(4,'Bundle Wizard & Generate Manifest','Copy GUI, scripts, docs, and create manifest.json.','bun',ss.bun,'srcBun()')}
  <label class="checkbox-row"><input type="checkbox" id="autoBundle" checked> Auto-bundle on completion (recommended)</label>
  <div id="debugPanel" style="background:var(--bg2);border:1px solid var(--b2);border-radius:8px;padding:8px 12px;margin:8px 0;font-size:11px;color:var(--txt2);display:none"></div>
  ${logCard('source','Export Log')}
  <div style="height:40px"></div>`;
}

function renderTarget(){
  const ts=targetSteps;
  return `<div class="sechead"><h1>Target Machine — Safe Update</h1><div class="sechead-line"></div><p>Select the project on <em>this</em> machine, then follow the steps in order. Safety backup is enforced.</p></div>
  <div class="card">
    <div class="card-label">📂  Project Folder Path (on this target machine)</div>
    <div class="input-row">
      <input type="text" class="text-input" id="tgtProjPath" value="${escHtml(savedTgtProj)}" placeholder="Auto-detecting..." oninput="onProjInputTgt(this.value)">
      <button class="btn-secondary" onclick="autoDetectTgt()">🔍 Auto-Detect</button>
      <button class="btn-secondary" onclick="openDirBrowser('tgt')">📁 Browse</button>
      <button class="btn-secondary" onclick="pasteWinPathTgt()">📋 Paste Windows Path</button>
    </div>
    <div class="status-row" id="tgtProjStatus"><span class="status-icon mute">○</span><span class="status-text mute">No project selected</span></div>
  </div>
  <div class="warn-banner">
    <div class="warn-icon-wrap"><span class="warn-icon">⚠</span></div>
    <div><div class="warn-title">BACKUP IS MANDATORY</div><div class="warn-text">Step 2 must be completed before importing. This protects your indexed PDFs, model cache, and settings.</div></div>
  </div>
  ${stepCard(1,'Detect Update Package','Find VisionRAG_Update on your DOK.','detect',ts.detect,'tgtDetect()')}
  ${stepCard(2,'FULL SAFETY BACKUP — MANDATORY','Backup images, all volumes, code, and generate restore script.','backup',ts.backup,'tgtBackup()',ts.detect!=='done',true)}
  ${stepCard(3,'Import Update from DOK','Load new images and copy updated code. Volumes preserved.','import',ts.import,'tgtImport()',ts.detect!=='done',false,true)}
  ${stepCard(4,'Auto Verification & Health Check','Check containers, pipeline, Ollama, Qdrant, .env config.','verify',ts.verify,'tgtVerify()',ts.import!=='done')}
  <div id="verifyResults" class="hidden verify-card"><div class="verify-title">✓  Verification Results</div><div id="verifySummary"></div></div>
  <div class="emergency-row"><button class="btn-danger" id="restoreBtn" onclick="confirmRestore()" disabled>🚨  Emergency Restore from Backup</button></div>
  ${logCard('target','Live Log')}
  <div style="height:40px"></div>`;
}

function stepCard(n,title,desc,key,state,onclick,disabled=false,errTitle=false,rerun=false){
  let barCls='',badgeCls='',titleCls='',btnCls='',btnText='Run',btnDisabled=disabled;
  if(state==='done'){barCls='ok';badgeCls='ok';titleCls='ok';btnCls=rerun?'':'done';btnText=rerun?'↻ Re-run':'✓ Complete';btnDisabled=!rerun}
  else if(state==='running'){barCls='warn';badgeCls='warn';btnText='Running...';btnDisabled=true}
  else if(state==='error'){barCls='err';badgeCls='err';titleCls='err';btnCls='err-btn';btnText='⚠ Failed'}
  else if(state==='need_sudo'){barCls='warn';badgeCls='warn';titleCls='warn';btnCls='warn';btnText='🔒 Sudo Required'}
  if(errTitle&&state!=='done') titleCls='err';
  return `<div class="step-card">
    <div class="step-bar ${barCls}"></div>
    <div class="step-inner">
      <div class="step-badge ${badgeCls}">${n}</div>
      <div class="step-info">
        <div class="step-title ${titleCls}">${title}</div>
        <div class="step-desc">${desc}</div>
      </div>
      <button class="step-btn ${btnCls}" ${btnDisabled?'disabled':''} onclick="${onclick}">${btnText}</button>
    </div>
  </div>`;
}

function logCard(channel,title){
  return `<div class="log-card">
    <div class="log-header">
      <span class="log-icon">▸</span>
      <span class="log-title">${title}</span>
      <button class="btn-secondary" style="padding:4px 14px;font-size:11px" onclick="toggleAutoScroll('${channel}')" id="scroll-btn-${channel}">⏸ Pause</button>
      <button class="btn-secondary" style="padding:4px 14px;font-size:11px" onclick="clearLog('${channel}')">Clear</button>
    </div>
    <div class="log-body" id="log-${channel}"></div>
  </div>`;
}

// ── Log polling ──
const logPositions={source:0,target:0};
const logTimers={};
const autoScroll={source:true,target:true};

function toggleAutoScroll(channel){
  autoScroll[channel]=!autoScroll[channel];
  const btn=document.getElementById('scroll-btn-'+channel);
  if(btn) btn.textContent=autoScroll[channel]?'⏸ Pause':'▶ Auto';
}

function startLogStream(channel){
  if(logTimers[channel]) clearInterval(logTimers[channel]);
  pollLogs(channel);
  logTimers[channel]=setInterval(()=>pollLogs(channel),2000);
}

async function pollLogs(channel){
  const el=document.getElementById('log-'+channel);
  if(!el) return;
  try{
    const r=await api('/api/logs/'+channel+'?since='+logPositions[channel]);
    if(r.entries&&r.entries.length>0){
      logPositions[channel]=r.next;
      const colors={INFO:'#8bb4e0',OK:'var(--ok)',WARN:'var(--warn)',ERROR:'var(--err)',CMD:'var(--ac2)'};
      for(const d of r.entries){
        el.innerHTML+=`<div class="log-entry"><span class="log-ts">[${d.ts}]</span> <span class="log-level ${d.level}" style="color:${colors[d.level]||'var(--txt2)'}">${d.level}</span> <span class="log-msg">${escHtml(d.msg)}</span></div>`;
      }
      if(autoScroll[channel]) el.scrollTop=el.scrollHeight;
    }
  }catch(e){}
}
function clearLog(channel){
  const el=document.getElementById('log-'+channel);
  if(el) el.innerHTML='';
}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

// ── Actions ──
function chooseMode(mode){
  appMode=mode;
  setStatus(mode==='source'?'Mode: Source Export':'Mode: Target Update',mode);
  if(mode==='source') goPage('source');
  else goPage('target');
}

// ── Source project path ──
async function autoDetectSrc(){
  const r=await api('/api/auto-detect');
  const el=document.getElementById('srcProjPath');
  if(el&&r.path){
    el.value=r.path;
    savedSrcProj=r.path;
    await api('/api/set-project','POST',{path:r.path});
    updateProjStatusSrc(r.valid,r.path);
  } else if(el){ updateProjStatusSrc(false,null); }
}
async function pasteWinPathSrc(){
  const wp=prompt('Enter Windows path (e.g. C:\\Users\\You\\projects\\Vision_RAG_Git):');
  if(!wp) return;
  const r=await api('/api/win2wsl','POST',{path:wp});
  if(r.path){
    document.getElementById('srcProjPath').value=r.path;
    await api('/api/set-project','POST',{path:r.path});
    updateProjStatusSrc(r.valid,r.path);
  } else alert('Could not convert path.');
}
function onProjInputSrc(val){
  savedSrcProj=val.trim();
  val=val.trim();
  if(val.length>=2&&val[1]===':'&&val[0].match(/[a-zA-Z]/)){
    api('/api/win2wsl','POST',{headers:{'Content-Type':'application/json'},body:JSON.stringify({path:val})}).then(r=>{
      if(r.path){document.getElementById('srcProjPath').value=r.path;updateProjStatusSrc(r.valid,r.path)}
    });
  }
}
function updateProjStatusSrc(valid,path){
  const el=document.getElementById('srcProjStatus');
  if(valid){
    el.innerHTML='<span class="status-icon ok">✓</span><span class="status-text ok">Valid Vision RAG project</span>';
  } else {
    el.innerHTML='<span class="status-icon warn">○</span><span class="status-text warn">'+(path?'Folder may not be a Vision RAG project':'No project found. Use Browse or Paste.')+'</span>';
  }
}

// ── Target project path ──
async function autoDetectTgt(){
  const r=await api('/api/auto-detect');
  const el=document.getElementById('tgtProjPath');
  if(el&&r.path){
    el.value=r.path;
    savedTgtProj=r.path;
    await api('/api/set-project','POST',{path:r.path});
    updateProjStatusTgt(r.valid,r.path);
  } else if(el){ updateProjStatusTgt(false,null); }
}
async function pasteWinPathTgt(){
  const wp=prompt('Enter Windows path (e.g. C:\\Users\\You\\projects\\Vision_RAG_Git):');
  if(!wp) return;
  const r=await api('/api/win2wsl','POST',{path:wp});
  if(r.path){
    document.getElementById('tgtProjPath').value=r.path;
    await api('/api/set-project','POST',{path:r.path});
    updateProjStatusTgt(r.valid,r.path);
  } else alert('Could not convert path.');
}
function onProjInputTgt(val){
  savedTgtProj=val.trim();
  val=val.trim();
  if(val.length>=2&&val[1]===':'&&val[0].match(/[a-zA-Z]/)){
    api('/api/win2wsl','POST',{headers:{'Content-Type':'application/json'},body:JSON.stringify({path:val})}).then(r=>{
      if(r.path){document.getElementById('tgtProjPath').value=r.path;updateProjStatusTgt(r.valid,r.path)}
    });
  }
}
function updateProjStatusTgt(valid,path){
  const el=document.getElementById('tgtProjStatus');
  if(valid){
    el.innerHTML='<span class="status-icon ok">✓</span><span class="status-text ok">Valid Vision RAG project</span>';
  } else {
    el.innerHTML='<span class="status-icon warn">○</span><span class="status-text warn">'+(path?'Folder may not be a Vision RAG project':'No project found. Use Browse or Paste.')+'</span>';
  }
}

async function rescanDok(){
  const r=await api('/api/auto-dok');
  const el=document.getElementById('dokPath');
  if(el&&r.path){ el.value=r.path; savedDok=r.path; }
  else if(el) el.value='';
}

async function mountDrive(){
  const dok=document.getElementById('dokPath')?.value?.trim()||'';
  let drive='';
  if(dok.startsWith('/mnt/')&&dok.length>=6) drive=dok[5].toLowerCase();
  else{ drive=prompt('Enter drive letter to mount (e.g. f):','f'); }
  if(!drive||drive.length!==1||!drive.match(/[a-z]/i)){return;}
  drive=drive.toLowerCase();
  const password=prompt(`Enter your WSL/sudo password to mount ${drive.toUpperCase()}: drive:`);
  if(password===null)return;
  const st=document.getElementById('mountStatus');
  if(st) st.innerHTML='<span style="color:var(--accent)">Mounting '+drive.toUpperCase()+': ...</span>';
  const r=await api('/api/mount-drive','POST',{drive,password});
  if(r.ok){
    if(st) st.innerHTML='<span style="color:var(--ok)">✓ '+drive.toUpperCase()+': mounted at /mnt/'+drive+'</span>';
    await rescanDok();
  }else{
    if(st) st.innerHTML='<span style="color:var(--err)">✗ Mount failed: '+(r.error||'unknown')+'</span>';
  }
}

// ── Source step actions ──
async function srcDisc(){
  window._debug=window._debug||[];
  window._debug.push('srcDisc called at '+new Date().toISOString());
  try{
    const proj=document.getElementById('srcProjPath')?.value;
    window._debug.push('srcProjPath='+proj);
    if(!proj){alert('Project path is empty! Select a project first.');return;}
    await api('/api/set-project','POST',{path:proj.trim()});
    window._debug.push('set-project sent');
    const dok=document.getElementById('dokPath')?.value;
    if(dok) await api('/api/set-dok','POST',{path:dok.trim()});
    window._debug.push('paths saved, calling discover');
    await runStep('source','disc','/api/source/discover');
  }catch(e){window._debug.push('ERROR: '+e.message);alert('Error: '+e.message);}
  const dp=document.getElementById('debugPanel');
  if(dp&&window._debug){dp.style.display='block';dp.innerHTML=window._debug.join('<br>');}
}
async function srcExp(){if(!await saveSrcPaths())return;await runStep('source','exp','/api/source/export')}
async function srcCode(){
  if(!await saveSrcPaths())return;
  await runStep('source','code','/api/source/code');
  if(document.getElementById('autoBundle')?.checked&&sourceSteps.code==='done') srcBun();
}
async function srcBun(){if(!await saveSrcPaths())return;await runStep('source','bun','/api/source/bundle')}

async function saveSrcPaths(){
  const proj=document.getElementById('srcProjPath')?.value?.trim()||savedSrcProj;
  const dok=document.getElementById('dokPath')?.value?.trim()||savedDok;
  if(!proj){alert('Please select a project folder first.');return false}
  savedSrcProj=proj; savedDok=dok;
  await api('/api/set-project','POST',{path:proj});
  if(dok) await api('/api/set-dok','POST',{path:dok});
  return true;
}

// ── Target step actions ──
async function tgtDetect(){if(!await saveTgtPaths())return;await runStep('target','detect','/api/target/detect')}
async function tgtBackup(){if(!await saveTgtPaths())return;await runStep('target','backup','/api/target/backup')}
async function tgtImport(){
  if(!await saveTgtPaths())return;
  targetSteps.import='running'; renderPage();
  const r=await api('/api/target/import','POST');
  if(r.need_sudo){
    targetSteps.import='need_sudo'; renderPage();
    const pw=prompt('Some files are owned by root (created by Docker containers).\nEnter your sudo password to fix ownership and retry:');
    if(pw){
      targetSteps.import='running'; renderPage();
      const r2=await api('/api/target/import-sudo','POST',{password:pw});
      targetSteps.import=r2.ok?'done':'error'; renderPage();
    }
  } else {
    targetSteps.import=r.ok?'done':'error'; renderPage();
  }
}
async function tgtVerify(){if(!await saveTgtPaths())return;await runStep('target','verify','/api/target/verify',true)}

async function saveTgtPaths(){
  const proj=document.getElementById('tgtProjPath')?.value?.trim()||savedTgtProj;
  if(!proj){alert('Please select a project folder first.');return false}
  savedTgtProj=proj;
  await api('/api/set-project','POST',{path:proj});
  return true;
}

async function runStep(channel,key,url,isVerify=false){
  const steps=channel==='source'?sourceSteps:targetSteps;
  steps[key]='running';
  renderPage();
  console.log('runStep',channel,key,url);
  const r=await api(url,'POST');
  steps[key]=r.ok?'done':'error';
  if(channel==='source') sourceSteps={...steps};
  else targetSteps={...steps};
  if(isVerify&&r.results){
    const el=document.getElementById('verifyResults');
    const sum=document.getElementById('verifySummary');
    if(el&&sum){
      el.classList.remove('hidden');
      let html='';
      for(const[k,v]of Object.entries(r.results)){
        const icon=v[0]?'✓':'✗';
        const color=v[0]?'var(--ok)':'var(--err)';
        html+=`<div class="verify-item"><span style="color:${color}">${icon}</span> <b>${k.replace(/_/g,' ').replace(/\b\w/g,l=>l.toUpperCase())}</b>: ${escHtml(String(v[1]).substring(0,80))}</div>`;
      }
      sum.innerHTML=html;
    }
  }
  if(channel==='target'&&key==='backup'&&r.ok){
    const btn=document.getElementById('restoreBtn');
    if(btn) btn.disabled=false;
  }
  renderPage();
}

function confirmRestore(){
  document.getElementById('restoreInput').value='';
  document.getElementById('restoreModal').classList.add('active');
}
async function doRestore(){
  const v=document.getElementById('restoreInput').value.trim();
  if(v!=='RESTORE'){alert('You must type RESTORE in all caps to confirm.');return}
  closeModal('restoreModal');
  const r=await api('/api/emergency-restore','POST');
  if(r.ok) alert('Restore complete!');
  else alert('Restore failed: '+(r.error||'Unknown error'));
}

// ── Tools / Help modals ──
async function showTools(){
  const r=await api('/api/tools/status');
  const c=document.getElementById('toolsContent');
  c.innerHTML=`<p style="color:var(--txt2);margin-bottom:16px">Clean up Docker artifacts or inspect your system.</p>
  <h3 style="font-size:14px;margin-bottom:8px">Docker System Status</h3>
  <pre style="margin-bottom:16px">${escHtml(r.df||'Error')}</pre>
  <p style="color:var(--mute);font-size:11px;margin-bottom:12px">Volumes with 'vision_rag_git_' prefix are NEVER touched.</p>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px">
    <button class="btn-secondary" onclick="toolAction('prune-dangling')">Prune Dangling Images</button>
    <button class="btn-secondary" onclick="toolAction('prune-containers')">Prune Stopped Containers</button>
    <button class="btn-secondary" onclick="toolAction('remove-none')">Remove &lt;none&gt; Tags</button>
    <button class="btn-danger" onclick="toolAction('prune-all')">Prune ALL Unused Images</button>
  </div>
  <h3 style="font-size:14px;margin-bottom:8px">Docker Inventory</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    <div><div style="color:var(--txt2);font-size:11px;margin-bottom:4px">Images</div><pre style="max-height:200px">${escHtml(r.images||'')}</pre></div>
    <div><div style="color:var(--txt2);font-size:11px;margin-bottom:4px">Containers</div><pre style="max-height:200px">${escHtml(r.containers||'')}</pre></div>
  </div>`;
  document.getElementById('toolsModal').classList.add('active');
}

async function toolAction(action){
  const r=await api('/api/tools/action','POST',{action});
  alert(r.ok?r.msg:r.error);
  showTools();
}

function showHelp(){document.getElementById('helpModal').classList.add('active')}
function closeModal(id){document.getElementById(id).classList.remove('active')}

// ── Init ──
renderPage();
</script>
</body>
</html>
"""


# ── Flask Routes ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


_detect_cache = {"project": None, "dok": None, "proj_valid": False}


@app.route("/api/auto-detect")
def api_auto_detect():
    global _detect_cache
    if _detect_cache["project"] is None:
        p = auto_detect_project()
        _detect_cache = {
            "project": p,
            "dok": _detect_cache["dok"],
            "proj_valid": is_valid_project(p) if p else False,
        }
    return jsonify(
        {"path": _detect_cache["project"], "valid": _detect_cache["proj_valid"]}
    )


@app.route("/api/auto-dok")
def api_auto_dok():
    global _detect_cache
    if _detect_cache["dok"] is None:
        _try_mount_all_drives()
        p = auto_detect_dok()
        _detect_cache["dok"] = p
    return jsonify({"path": _detect_cache["dok"]})


@app.route("/api/set-project", methods=["POST"])
def api_set_project():
    d = request.json or {}
    path = d.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path provided", "valid": False})
    if len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        c = win2wsl(path)
        if c:
            path = c
    state.proj = path
    valid = is_valid_project(path)
    _detect_cache["project"] = path
    _detect_cache["proj_valid"] = valid
    return jsonify({"path": path, "valid": valid})


@app.route("/api/mount-drive", methods=["POST"])
def api_mount_drive():
    d = request.json or {}
    drive = d.get("drive", "").strip().lower()
    password = d.get("password", "")
    if not drive or len(drive) != 1 or not drive.isalpha():
        return jsonify({"ok": False, "error": "Invalid drive letter"})
    mnt = Path(f"/mnt/{drive}")
    if _is_drvfs_mounted(mnt):
        return jsonify({"ok": True, "mounted": True, "path": str(mnt)})
    try:
        r = subprocess.run(
            ["sudo", "-S", "mount", "-t", "drvfs", f"{drive.upper()}:", str(mnt)],
            input=password.encode(),
            capture_output=True,
            timeout=15,
        )
        if _is_drvfs_mounted(mnt):
            return jsonify({"ok": True, "mounted": True, "path": str(mnt)})
        err = r.stderr.decode(errors="replace").strip()
        lines = [l for l in err.splitlines() if not l.strip().startswith("[sudo]")]
        err_clean = " ".join(l.strip() for l in lines if l.strip())
        if (
            "Sorry" in err
            or "incorrect" in err.lower()
            or "a password is required" in err.lower()
        ):
            return jsonify({"ok": False, "error": "Wrong password"})
        return jsonify({"ok": False, "error": err_clean[:200] or "mount failed"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:200]})


@app.route("/api/set-dok", methods=["POST"])
def api_set_dok():
    d = request.json or {}
    path = d.get("path", "").strip()
    if path:
        state.dok = path
        _detect_cache["dok"] = path
    return jsonify({"path": state.dok})


@app.route("/api/browse")
def api_browse():
    st = "/mnt"
    for drive in _mnt_drives():
        _ensure_mounted(drive)
        if drive.exists():
            st = str(drive)
            break
    try:
        r = subprocess.run(
            ["zenity", "--file-selection", "--directory", "--filename", st],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode == 0:
            p = r.stdout.strip()
            return jsonify({"path": p, "valid": is_valid_project(p)})
    except Exception:
        pass
    return jsonify({"path": None})


@app.route("/api/win2wsl", methods=["POST"])
def api_win2wsl():
    d = request.json or {}
    wp = d.get("path", "").strip()
    c = win2wsl(wp)
    if c:
        return jsonify({"path": c, "valid": is_valid_project(c)})
    return jsonify({"path": None})


@app.route("/api/logs/<channel>")
def api_logs(channel):
    if channel not in ("source", "target"):
        return jsonify({"entries": []})
    since = int(request.args.get("since", "0"))
    entries = _log_data[channel][since:]
    return jsonify({"entries": entries, "next": len(_log_data[channel])})


@app.route("/api/source/discover", methods=["POST"])
def api_src_disc():
    def run():
        src_disc()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=30)
    return jsonify({"ok": state.source_steps["disc"] == "done"})


@app.route("/api/source/export", methods=["POST"])
def api_src_exp():
    if not state.dok:
        state.dok = auto_detect_dok() or ""

    def run():
        src_exp()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=7200)
    return jsonify({"ok": state.source_steps["exp"] == "done"})


@app.route("/api/source/code", methods=["POST"])
def api_src_code():
    def run():
        src_code()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=300)
    return jsonify({"ok": state.source_steps["code"] == "done"})


@app.route("/api/source/bundle", methods=["POST"])
def api_src_bun():
    def run():
        src_bun()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=60)
    return jsonify({"ok": state.source_steps["bun"] == "done"})


@app.route("/api/target/detect", methods=["POST"])
def api_tgt_detect():
    def run():
        tgt_detect()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=15)
    return jsonify({"ok": state.target_steps["detect"] == "done"})


@app.route("/api/target/backup", methods=["POST"])
def api_tgt_backup():
    def run():
        tgt_backup()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=7200)
    return jsonify({"ok": state.target_steps["backup"] == "done"})


@app.route("/api/target/import", methods=["POST"])
def api_tgt_import():
    def run():
        tgt_import()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=7200)
    return jsonify(
        {"ok": state.target_steps["import"] == "done", "need_sudo": state.need_sudo_pw}
    )


@app.route("/api/target/import-sudo", methods=["POST"])
def api_tgt_import_sudo():
    d = request.json or {}
    pw = d.get("password", "")
    if not pw:
        return jsonify({"ok": False, "error": "No password provided"})
    state.sudo_pw = pw

    def run():
        tgt_import_sudo(pw)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=120)
    ok = state.target_steps["import"] == "done"
    return jsonify({"ok": ok})


@app.route("/api/target/verify", methods=["POST"])
def api_tgt_verify():
    def run():
        tgt_verify()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=600)
    results_serializable = {}
    for k, (ok, detail) in state.verify.items():
        results_serializable[k] = [ok, detail]
    return jsonify(
        {"ok": state.target_steps["verify"] == "done", "results": results_serializable}
    )


@app.route("/api/emergency-restore", methods=["POST"])
def api_emergency_restore():
    ok = emergency_restore()
    return jsonify({"ok": ok})


@app.route("/api/tools/status")
def api_tools_status():
    result = {"df": "", "images": "", "containers": ""}
    try:
        r = subprocess.run(
            ["docker", "system", "df"], capture_output=True, text=True, timeout=10
        )
        result["df"] = r.stdout if r.returncode == 0 else r.stderr
    except Exception as e:
        result["df"] = f"Error: {e}"
    try:
        r = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["images"] = r.stdout if r.returncode == 0 else ""
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["containers"] = r.stdout if r.returncode == 0 else ""
    except Exception:
        pass
    return jsonify(result)


@app.route("/api/tools/action", methods=["POST"])
def api_tools_action():
    d = request.json or {}
    action = d.get("action", "")
    cmds = {
        "prune-dangling": ["docker", "image", "prune", "-f"],
        "prune-containers": ["docker", "container", "prune", "-f"],
        "remove-none": [
            "bash",
            "-c",
            "docker images --filter 'dangling=true' -q | xargs -r docker rmi",
        ],
        "prune-all": ["docker", "image", "prune", "-a", "-f"],
    }
    if action not in cmds:
        return jsonify({"ok": False, "error": "Unknown action"})
    try:
        r = subprocess.run(cmds[action], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return jsonify({"ok": True, "msg": r.stdout.strip()[:500]})
        else:
            return jsonify({"ok": False, "error": r.stderr.strip()[:500]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/list-dir")
def api_list_dir():
    path = request.args.get("path", "/")
    try:
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return jsonify({"path": path, "dirs": [], "error": "Not a directory"})
        dirs = []
        for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if c.is_dir():
                valid = is_valid_project(c)
                dirs.append({"name": c.name, "path": str(c), "valid": valid})
        return jsonify({"path": str(p.resolve()), "dirs": dirs})
    except Exception as e:
        return jsonify({"path": path, "dirs": [], "error": str(e)})


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Vision RAG Migration Wizard v4.0 (Web UI)")
    print("=" * 60)
    print(f" Open your browser: http://localhost:5555")
    print("=" * 60)
    app.run(
        host="0.0.0.0",
        port=5555,
        debug=False,
        threaded=True,
        extra_files=None,
        exclude_patterns=None,
    )


if __name__ == "__main__":
    main()
