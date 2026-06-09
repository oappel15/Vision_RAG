# Vision RAG — Offline Update Guide

This guide covers updating an existing Vision RAG installation on an **offline machine** using an export package from an online source machine.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [On the Source Machine (Online)](#on-the-source-machine-online)
4. [Transfer to Offline Machine](#transfer-to-offline-machine)
5. [On the Target Machine (Offline)](#on-the-target-machine-offline)
6. [Post-Update Verification](#post-update-verification)
7. [OpenCode Integration Setup](#opencode-integration-setup)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The update workflow preserves all your existing data:

| What is Preserved | What is Updated |
|-------------------|-----------------|
| Indexed PDFs (Qdrant embeddings) | Docker images (new versions) |
| ColQwen2 model cache (`hf-cache`) | Project code (`pipelines/`, `pdf-ingest/`, etc.) |
| Ollama models (`ollama` volume) | `docker-compose.yml` configuration |
| Open WebUI settings (`open-webui` volume) | OpenCode integration scripts |
| Your `.env` file (backed up) | |

**Volume data is never touched.** Only container images and host code are replaced.

---

## Prerequisites

### Source Machine (Online)
- Docker Desktop running with WSL2 integration
- Vision RAG project cloned and running
- `rsync` installed (usually pre-installed on WSL2 Ubuntu)

### Target Machine (Offline)
- Docker Desktop running with WSL2 integration
- Existing Vision RAG installation with volumes
- WSL2 Ubuntu (or Git Bash) to run bash scripts

---

## On the Source Machine (Online)

### Step 1: Navigate to the project

```bash
cd /path/to/Vision_RAG_Git
```

### Step 2: Run the export script

```bash
bash opencode/export-update.sh /mnt/d/VisionRAG_Update
```

Replace `/mnt/d/VisionRAG_Update` with your DOK or external drive mount point.

**What this does:**
1. Discovers all Docker images from `docker-compose.yml`
2. Saves each image as a `.tar.gz` file
3. Copies all project code (excluding `.env`, `.git`, caches)
4. Generates a `manifest.json` with checksums
5. Copies the `import-update.sh` script for the target

**Expected output:**
```
========================================
  Vision RAG — Export Update Package
========================================

[INFO] Export directory: /mnt/d/VisionRAG_Update
[INFO] Source repo: /home/user/projects/Vision_RAG_Git

...

========================================
  Export Complete!
========================================

[INFO] Total export size: 12G

Export contents:
  /mnt/d/VisionRAG_Update/
    images/          — Docker images (.tar.gz)
    code/            — Updated project code
    manifest.json    — Inventory with checksums
    import-update.sh — Import script for target machine
```

**Estimated time:** 10–30 minutes depending on image sizes.

---

## Transfer to Offline Machine

1. **Safely eject** the DOK from the source machine
2. **Plug in** the DOK to the offline target machine
3. **Mount** the drive in WSL2:
   ```bash
   # The drive should auto-mount under /mnt/
   ls /mnt/d/VisionRAG_Update
   ```

---

## On the Target Machine (Offline)

### Step 1: Run the import script

```bash
cd /mnt/d/VisionRAG_Update
bash import-update.sh
```

**What this does:**
1. **Detects** your existing Docker volume prefix (e.g., `vision_rag_git`)
2. **Verifies** critical volumes exist (`hf-cache`, `qdrant_data`, `ollama`, `open-webui`)
3. **Asks** for your project directory (auto-detects if found)
4. **Loads** all Docker images from the `.tar.gz` files
5. **Copies** updated code to your project directory
6. **Sets** `COMPOSE_PROJECT_NAME` in `.env` to match existing volumes
7. **Backs up** your existing `.env` before modifying

**Example interaction:**
```
========================================
  Vision RAG — Import Update Package
========================================

[INFO] Export directory: /mnt/d/VisionRAG_Update

[INFO] [1/6] Checking prerequisites...
[OK]   Docker and Docker Compose are available

[INFO] [2/6] Detecting existing Vision RAG volumes...
[OK]   Found existing volume prefix: vision_rag_git
[INFO]   Existing volumes:
    - vision_rag_git_hf-cache (5.2GB)
    - vision_rag_git_qdrant_data (1.8GB)
    - vision_rag_git_ollama (0B)
    - vision_rag_git_open-webui (150MB)

[INFO] [3/6] Pre-flight check: verifying critical volumes...
[OK]   All critical volumes present
[INFO]   Volume contents preview:
    - vision_rag_git_hf-cache: 1247 files
    - vision_rag_git_qdrant_data: 56 files
    - vision_rag_git_ollama: 0 files
    - vision_rag_git_open-webui: 23 files

[INFO] [4/6] Target project directory
[?] Existing project found at: /home/user/projects/Vision_RAG_Git
[?] Use this directory? [Y/n]: y
[INFO]   Target directory: /home/user/projects/Vision_RAG_Git

[INFO] [5/6] Loading Docker images...
[INFO]   Found 7 images to load
[INFO]   Loading ollama-ollama-latest.tar.gz...
[OK]     Loaded successfully
...
[OK]   Loaded 7/7 images

[INFO] [6/6] Updating project code...
[OK]   Backed up existing .env
[INFO]   Copying files to /home/user/projects/Vision_RAG_Git...
[OK]   Code updated

[INFO] [Config] Setting up Docker Compose project name...
[OK]   Updated COMPOSE_PROJECT_NAME=vision_rag_git in .env

========================================
  Import Complete!
========================================

[OK] Updated:
  7 Docker images loaded
  Project code copied to: /home/user/projects/Vision_RAG_Git
  COMPOSE_PROJECT_NAME set to: vision_rag_git

[INFO] Next Steps:

  1. Review your environment variables:
     /home/user/projects/Vision_RAG_Git/.env

  2. Start the updated services:
     cd /home/user/projects/Vision_RAG_Git
     docker compose up -d

  3. Verify everything is running:
     docker compose ps

  4. (Optional) Install/update OpenCode integration:
     cd /home/user/projects/Vision_RAG_Git
     bash opencode/setup-opencode.sh

  5. Check that your indexed PDFs are still available:
     curl http://localhost:8082/status
```

### Step 2: Start the updated services

```bash
cd /home/user/projects/Vision_RAG_Git
docker compose up -d
```

**What happens:**
- Docker detects the new images have different tags
- It **recreates containers** using the new images
- It **reuses the same volumes** (data is preserved)
- Your indexed PDFs, model cache, and settings remain intact

### Step 3: Clean up old images (optional)

After confirming everything works, remove old images to free disk space:

```bash
# Remove dangling (untagged) images
docker image prune -f

# List all images to see what's left
docker images
```

---

## Post-Update Verification

### Check containers are running

```bash
cd /home/user/projects/Vision_RAG_Git
docker compose ps
```

Expected: All 7 services showing `Up` or `healthy`.

### Check indexed PDFs are preserved

```bash
curl http://localhost:8082/status
```

You should see your previously indexed PDFs listed.

### Check the pipeline is ready

```bash
docker logs open-webui-pipelines -f 2>&1 | grep "ColQwen2 ready"
```

Wait for the "ColQwen2 ready" message (may take 1–3 minutes).

### Test a search query

Open your browser to `http://localhost:3000` and ask a question about one of your indexed PDFs.

---

## OpenCode Integration Setup

If you use OpenCode with Vision RAG, update the integration after the import:

### Step 1: Run the setup script

```bash
cd /home/user/projects/Vision_RAG_Git
bash opencode/setup-opencode.sh
```

This installs:
- Vision RAG MCP server (6 tools for search/upload/status)
- Vision RAG skill (auto-triggers on PDF keywords)
- KiCad netlist-to-schematic skill

### Step 2: Restart OpenCode

Close and reopen OpenCode for the MCP server changes to take effect.

### Step 3: Verify in OpenCode

Ask OpenCode:
```
What documents are in my Vision RAG index?
```

It should respond with a list of your indexed PDFs using the `vision_rag_status` tool.

### Custom host (for LAN access)

If OpenCode runs on a different machine than Vision RAG:

```bash
bash opencode/setup-opencode.sh --host 192.168.1.100
```

Replace `192.168.1.100` with the IP of the machine running Vision RAG.

---

## Troubleshooting

### "No existing Vision RAG volumes found"

The import script couldn't find any `_hf-cache` volumes. This means either:
- Docker Desktop isn't running
- The volumes were deleted
- This is a fresh machine (not an update)

**Solution:** If this is intentional (fresh install), the script will use the default prefix `vision_rag_git`. You'll need to set up volumes manually or run the full install from `INSTALL.md`.

### "Missing critical volumes"

One or more of `hf-cache`, `qdrant_data`, `ollama`, or `open-webui` volumes are missing.

**Solution:** Check `docker volume ls` to see what exists. If volumes were accidentally deleted, you may need to restore from a backup or re-index all PDFs.

### "Failed to load image"

A `.tar.gz` file may be corrupted or incomplete.

**Solution:** Re-run the export script on the source machine. Check the `manifest.json` checksums.

### "docker compose up -d tries to rebuild"

This happens if the image tags in `docker-compose.yml` don't match the loaded images.

**Solution:** Ensure `COMPOSE_PROJECT_NAME` is set correctly in `.env`. Check `docker images` to verify the loaded images have the expected tags (e.g., `vision-rag-pipelines:latest`).

### "OpenCode can't connect to Vision RAG"

The MCP server uses `localhost` by default. If Vision RAG runs on a different host:

```bash
bash opencode/setup-opencode.sh --host <vision-rag-ip>
```

Also verify the containers are running: `docker compose ps`

### "Indexed PDFs are gone after update"

This should never happen if volumes are preserved. Check:

```bash
# Verify volumes exist
docker volume ls | grep vision_rag_git

# Check Qdrant data
docker run --rm -v vision_rag_git_qdrant_data:/data busybox ls -la /data

# Check pipeline state
curl http://localhost:8082/status
```

If volumes are truly gone, they may have been pruned accidentally. You'll need to re-index all PDFs.

---

## Quick Reference

| Task | Command |
|------|---------|
| Export from source | `bash opencode/export-update.sh /mnt/d/Export` |
| Import on target | `bash import-update.sh` |
| Start services | `docker compose up -d` |
| Check status | `docker compose ps` |
| View logs | `docker compose logs -f` |
| Setup OpenCode | `bash opencode/setup-opencode.sh` |
| Setup with custom host | `bash opencode/setup-opencode.sh --host 192.168.1.100` |
| Check indexed PDFs | `curl http://localhost:8082/status` |
| Clean old images | `docker image prune -f` |

---

## Files Reference

| File | Purpose |
|------|---------|
| `opencode/export-update.sh` | Export script (run on source) |
| `opencode/import-update.sh` | Import script (run on target) |
| `opencode/setup-opencode.sh` | OpenCode integration installer |
| `UPDATE_GUIDE.md` | This guide |
| `docker-compose.yml` | Service definitions (updated with image tags) |
| `.env.example` | Environment template (includes `COMPOSE_PROJECT_NAME`) |
