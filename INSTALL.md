# Vision RAG — Installation Guide

Complete guide to build and run Vision RAG from a fresh git clone on Windows.

---

## Table of Contents

1. [Decision: Windows vs WSL2](#1-decision-windows-vs-wsl2)
2. [Prerequisites](#2-prerequisites)
3. [Clone and Configure](#3-clone-and-configure)
4. [Create the `.env` File](#4-create-the-env-file)
5. [Build and Start](#5-build-and-start)
6. [First-Run Initialization](#6-first-run-initialization)
7. [Configure the Pipeline in Open WebUI](#7-configure-the-pipeline-in-open-webui)
8. [Add Documents](#8-add-documents)
9. [Verify Everything Works](#9-verify-everything-works)
10. [LAN Access from Other Devices](#10-lan-access-from-other-devices)
11. [Confluence Watcher Setup](#11-confluence-watcher-setup)
12. [GPU Machine Setup (Local Ollama VLM)](#12-gpu-machine-setup-local-ollama-vlm)
13. [Stopping, Restarting, and Updating](#13-stopping-restarting-and-updating)
14. [Troubleshooting](#14-troubleshooting)
15. [Service Reference](#15-service-reference)

---

## 1. Decision: Windows vs WSL2

**Run everything through Docker Desktop on Windows.** You do NOT need to clone or
run anything inside WSL2 directly. Docker Desktop uses WSL2 as its backend
automatically, but you interact with it from a normal Windows terminal
(PowerShell, Windows Terminal, or CMD).

| Approach | Verdict |
|----------|---------|
| **Docker Desktop on Windows (recommended)** | Clone repo on Windows, run `docker compose` from PowerShell. Docker Desktop handles WSL2 integration behind the scenes. Port forwarding to LAN works out of the box. |
| **Clone inside WSL2 Linux filesystem** | Works, but adds complexity — you need to forward ports from WSL2 to Windows for LAN access (the `scripts/` PowerShell helpers exist for this). No real advantage since everything runs in containers anyway. |

**Bottom line:** Keep the repo on your Windows filesystem (e.g.
`C:\Users\you\projects\Vision_RAG`) and run all commands from PowerShell.

---

## 2. Prerequisites

Install these on your Windows machine before starting:

### 2a. Docker Desktop

1. Download from https://www.docker.com/products/docker-desktop/
2. Run the installer — accept defaults
3. During setup, ensure **"Use WSL 2 instead of Hyper-V"** is checked
4. After install, open Docker Desktop and wait for the engine to start
5. Verify in PowerShell:
   ```powershell
   docker --version
   docker compose version
   ```

### 2b. WSL2 (Docker Desktop requires it)

Docker Desktop installs WSL2 automatically if needed. If prompted:

```powershell
wsl --install
```

Restart if asked. You do NOT need to set up an Ubuntu distro unless you want to.

### 2c. Git

1. Download from https://git-scm.com/download/win
2. Install with defaults
3. Verify:
   ```powershell
   git --version
   ```

### 2d. OpenRouter API Key (for cloud VLM — no GPU machines)

1. Go to https://openrouter.ai and create an account
2. Navigate to https://openrouter.ai/keys
3. Click **"Create Key"**
4. Copy the key (starts with `sk-or-v1-...`)
5. Add credit — the default model (`qwen/qwen3-vl-30b-a3b-instruct`) costs
   roughly $0.15-0.30 per 1M tokens. A few dollars covers substantial testing.

> You can skip this if you will ONLY use a local GPU with Ollama. See
> [Section 12](#12-gpu-machine-setup-local-ollama-vlm).

---

## 3. Clone and Configure

Open PowerShell and run:

```powershell
cd C:\Users\$env:USERNAME\projects   # or wherever you keep repos
git clone <your-repo-url> Vision_RAG
cd Vision_RAG
```

Check the structure looks right:

```powershell
dir
```

You should see: `docker-compose.yml`, `Dockerfile`, `pipelines/`, `pdf-ingest/`,
`confluence-watcher/`, `my-pdfs/`, etc.

---

## 4. Create the `.env` File

The repo includes a `.env.example` template. Copy it and edit:

```powershell
copy .env.example .env
notepad .env
```

### Minimal `.env` (no GPU, no Confluence)

```env
OPENROUTER_API_KEY=sk-or-v1-paste-your-actual-key-here
HOST_IP=localhost
```

### Full `.env` (LAN access + Confluence)

```env
# VLM backend — required for cloud mode
OPENROUTER_API_KEY=sk-or-v1-paste-your-actual-key-here

# LAN access — your machine's IPv4 address
# Find it: ipconfig → look for "IPv4 Address" under your active adapter
HOST_IP=192.168.1.100

# Confluence watcher
CONFLUENCE_PAT=your-confluence-personal-access-token
CONFLUENCE_URL=http://confluence.yourcompany.com
CONFLUENCE_SPACES=RAG,ENG
CONFLUENCE_POLL_INTERVAL=60
```

### How to find your LAN IP

```powershell
ipconfig
```

Look for your active network adapter (Wi-Fi or Ethernet) and find the
**IPv4 Address** line (e.g. `192.168.1.100`). That goes into `HOST_IP`.

> **Important:** `.env` is gitignored — it will never be committed. Keep
> secrets here and nowhere else.

---

## 5. Build and Start

From the project root (`Vision_RAG/`), run:

```powershell
docker compose up -d
```

### What happens on first run

This will take **10-30 minutes** depending on your internet speed:

| Step | What's downloading | Size |
|------|--------------------|------|
| Docker base images | `open-webui`, `pipelines`, `qdrant`, `ollama`, `python:3.11` | ~5 GB total |
| Image builds | Installs PyTorch CPU, colpali-engine, poppler, etc. | ~3 GB |
| ColQwen2 model weights | Downloaded at container startup (cached in Docker volume) | ~5 GB |

### Watch the build progress

```powershell
docker compose up -d --build
```

Add `--build` to force a rebuild if you change any Dockerfile or code.

### Verify all containers are running

```powershell
docker compose ps
```

You should see **7 services** all with status `Up`:

```
NAME                    STATUS
ollama                  Up
open-webui              Up
open-webui-pipelines    Up
qdrant                  Up
image-server            Up    (shows as vision_rag_git-image-server-1 or similar)
pdf-ingest              Up
confluence-watcher      Up
```

> If `confluence-watcher` shows `Restarting` and you haven't configured
> Confluence yet, that's expected — it will keep retrying until configured
> or you can ignore it.

---

## 6. First-Run Initialization

The pipeline container needs to download the ColQwen2 model on first start.
This is a one-time ~5 GB download cached in a Docker volume (`hf-cache`).

### Watch for the ready signal

```powershell
docker logs open-webui-pipelines -f
```

Wait until you see a line containing **`ColQwen2 ready`** (typically 1-3 minutes
after containers start, longer on first run due to model download).

Press `Ctrl+C` to stop following logs once you see it.

### If it seems stuck

```powershell
# Check if the container is healthy
docker compose ps

# Check for errors
docker logs open-webui-pipelines --tail 50

# Check Qdrant is responding
docker logs qdrant --tail 20
```

---

## 7. Connect the Pipeline and Configure Open WebUI

1. Open your browser to **http://localhost:3000**
   (or `http://<HOST_IP>:3000` from another device)

2. **Create an admin account** on first visit (this is local-only, pick anything)

### 7a. Register the Pipelines connection

The pipeline server runs as a separate container. Open WebUI needs to be told
where to find it — even though the environment variables are set in
`docker-compose.yml`, you must confirm the connection manually on first setup.

3. Click your **profile icon** (bottom-left) → **Admin Panel**
4. Go to **Settings** → **Connections**
5. In the **OpenAI API** section, click **"+"** to add a new connection:
   - **URL**: `http://pipelines:9099`
   - **API Key**: `0p3n-w3bu!`
6. Click the **verify/refresh** button (circular arrow icon) next to the entry
   — it should show a green checkmark
7. Click **Save**

> **Why is this needed?** The `PIPELINES_BASE_URL` environment variable seeds
> the initial config, but newer versions of Open WebUI require you to confirm
> or add the connection in the admin panel before models from the pipeline
> server appear in the chat model selector.

### 7b. Verify the model appears

8. Go back to the **chat** page (click "New Chat")
9. Click the **model selector** dropdown at the top — you should now see
   **"ColQwen2 Visual RAG"** in the list
10. Select it — this is the model you'll use for all RAG queries

> If the model doesn't appear, go back to **Settings → Connections** and
> confirm the pipelines entry shows a green checkmark. Try clicking the
> refresh button again.

### 7c. Configure pipeline valves

11. Go to **Admin Panel** → **Settings** → **Pipelines**
12. Find **"ColQwen2 Visual RAG"** and click the gear/settings icon
13. Set these valves:

    | Valve | Value | Notes |
    |-------|-------|-------|
    | `SERVER_HOST` | Your LAN IP (e.g. `192.168.1.100`) | Controls where browsers fetch thumbnails and PDFs. Use `localhost` for local-only access. |
    | `VLM_PROVIDER` | `openrouter` | Use `ollama` only if you have a GPU machine (see Section 12) |

14. Click **Save**

> **Changing networks?** If your LAN IP changes (e.g. moving to a different
> Wi-Fi), just update `SERVER_HOST` here. No restart needed.

---

## 8. Add Documents

### Option A: Upload via the Web UI (easiest)

1. Open **http://localhost:8082** (or `http://<HOST_IP>:8082`)
2. Drag and drop PDF files onto the page
3. Click **Index Now**
4. Watch the progress bar — indexing converts each page to an image and
   embeds it with ColQwen2

### Option B: Copy files directly

```powershell
# Copy a PDF into the source directory
copy C:\path\to\your\document.pdf .\my-pdfs\
```

Then trigger indexing — open the Open WebUI chat and type:

```
__index_now__
```

### Option C: Confluence sync (automatic)

If you configured Confluence in `.env`, the watcher automatically polls for
new/changed pages and indexes them. See [Section 11](#11-confluence-watcher-setup).

---

## 9. Verify Everything Works

| Check | How | Expected |
|-------|-----|----------|
| Chat UI loads | http://localhost:3000 | Open WebUI login page |
| Upload UI loads | http://localhost:8082 | PDF upload interface |
| Pipeline is ready | Type `status` in the chat | Shows indexed files and status |
| RAG query works | Ask about an indexed document | Answer with citations |
| Thumbnails load | Should appear below answers | Page thumbnail images |
| PDF links work | Click a citation link | Opens PDF page viewer |
| Image server | http://localhost:8081 | Directory listing of cached images |

---

## 10. LAN Access from Other Devices

Docker Desktop on Windows automatically forwards container ports to `0.0.0.0`,
so other devices on your LAN can reach the services using your Windows IP.

From any device on the same network:

| Service | URL |
|---------|-----|
| Chat UI | `http://<HOST_IP>:3000` |
| Upload UI | `http://<HOST_IP>:8082` |
| Image server | `http://<HOST_IP>:8081` |

Make sure:
1. `HOST_IP` in `.env` is set to your actual LAN IP
2. `SERVER_HOST` valve in Open WebUI matches (see Step 7)
3. Windows Firewall allows inbound on ports 3000, 8081, 8082

> **Note:** On the host machine itself, `http://<HOST_IP>:3000` may not work
> in the browser even though the port is open. This is a known Windows/Docker
> Desktop quirk — the browser on the host sometimes fails to route to its own
> LAN IP. Use `http://localhost:3000` from the host machine. The LAN IP
> (`http://<HOST_IP>:3000`) works correctly from all **other** devices.

### Windows Firewall rule (if devices can't connect)

Run PowerShell **as Administrator**:

```powershell
New-NetFirewallRule -DisplayName "Vision RAG" `
  -Direction Inbound -Protocol TCP `
  -LocalPort 3000,8081,8082 `
  -Action Allow
```

### If using WSL2 directly (not recommended)

If you cloned inside WSL2 instead of Windows, you need port forwarding.
The repo includes helper scripts in `scripts/`:

```powershell
# Run as Administrator
.\scripts\update-wsl-portproxy.ps1
```

This creates `netsh portproxy` rules. To make it persist across reboots:

```powershell
# Run as Administrator
.\scripts\register-portproxy-task.ps1
```

> Note: `register-portproxy-task.ps1` has a hardcoded WSL path. Edit line
> containing `\\wsl.localhost\Ubuntu\home\appel\...` to match your username
> and distro before running.

---

## 11. Confluence Watcher Setup

The watcher polls Confluence Server, renders pages to PDF via headless
Chromium, and triggers indexing automatically.

### Step 1: Generate a Personal Access Token (PAT)

1. Log in to Confluence Server
2. Click your profile avatar (top-right)
3. Go to **Profile** → **Settings** (left sidebar)
4. Click **Personal Access Tokens**
5. Click **Create token**, name it (e.g. `vision-rag`), click **Create**
6. **Copy the token immediately** — it's shown only once

> PATs require Confluence Server 7.9+. For older versions, you'll need to
> modify `watcher.py` to use basic auth.

### Step 2: Add to `.env`

```env
CONFLUENCE_PAT=your-token-here
CONFLUENCE_URL=http://confluence.yourcompany.com
CONFLUENCE_SPACES=RAG,ENG
CONFLUENCE_POLL_INTERVAL=60
```

- **`CONFLUENCE_URL`**: The externally reachable Confluence URL. If Confluence
  runs on the same machine as Docker, use `http://host.docker.internal:8090`.
- **`CONFLUENCE_SPACES`**: Comma-separated Confluence Space Keys (find them
  under Space Settings → Space Key in Confluence).

### Step 3: Restart the watcher

```powershell
docker compose up -d confluence-watcher
```

The watcher indexes all pages on first run, then polls for changes.

### Step 4: Verify

```powershell
docker logs confluence-watcher -f
```

You should see it discovering pages, rendering PDFs, and triggering indexing.

---

## 12. GPU Machine Setup (Local Ollama VLM)

If you have a machine with an NVIDIA GPU (24+ GB VRAM), you can run the
vision-language model locally instead of paying for OpenRouter API calls.

### Requirements

| Item | Minimum |
|------|---------|
| GPU | NVIDIA with ~24 GB VRAM (RTX 3090, 4090, A5000, A100, etc.) |
| NVIDIA drivers | Latest Game Ready or Studio driver |
| Docker Desktop | With GPU support enabled (Settings → Resources → WSL integration) |
| NVIDIA Container Toolkit | Required for Docker GPU passthrough |

### Step 1: Install NVIDIA Container Toolkit

Follow the official guide:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

For WSL2 on Windows, the key steps are:

```bash
# Inside WSL2 Ubuntu:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 2: Enable GPU for the Ollama container

Edit `docker-compose.yml` — add a `deploy` section to the `ollama` service:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    volumes:
      - ollama:/root/.ollama
    ports:
      - "11434:11434"
    deploy:                          # <-- add this block
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

Rebuild:

```powershell
docker compose up -d ollama
```

### Step 3: Pull the VLM model

```powershell
docker exec -it ollama ollama pull qwen3-vl:30b
```

This downloads ~20 GB. Check progress:

```powershell
docker exec -it ollama ollama list
```

### Step 4: Switch the pipeline to Ollama

1. Open WebUI → **Settings → Admin → Pipelines → ColQwen2 Visual RAG**
2. Set **`VLM_PROVIDER`** → `ollama`
3. Confirm **`OLLAMA_VLM_MODEL`** = `qwen3-vl:30b`
4. **`OLLAMA_BASE_URL`** should already be `http://ollama:11434`
5. Save

### Step 5: Verify

Ask a question in the chat. Check Ollama logs to confirm it's processing:

```powershell
docker logs ollama -f
```

### Switching back to OpenRouter

Set **`VLM_PROVIDER`** back to `openrouter` in the pipeline valves. Make sure
`OPENROUTER_API_KEY` is set in `.env`.

### No GPU on this machine?

That's fine. The ColQwen2 **embedding model** runs on CPU (it's used for
indexing and search). Only the **answer-generation VLM** needs a GPU for local
inference. Without a GPU, use OpenRouter (cloud) — it works identically, just
costs a small amount per query.

---

## 13. Stopping, Restarting, and Updating

### Stop all services

```powershell
docker compose down
```

This stops containers but preserves all data (volumes are retained).

### Restart all services

```powershell
docker compose up -d
```

### Full rebuild (after code changes)

```powershell
docker compose up -d --build
```

### Update base images

```powershell
docker compose pull
docker compose up -d --build
```

### Nuclear reset (deletes ALL data including indexed documents and model cache)

```powershell
docker compose down -v
docker compose up -d --build
```

> **Warning:** `-v` removes all named volumes — you'll re-download the
> ColQwen2 model (~5 GB), lose all indexed data, and need to re-upload PDFs.

### View logs

```powershell
# All services
docker compose logs -f

# Specific service
docker logs open-webui-pipelines -f
docker logs qdrant -f
docker logs confluence-watcher -f
docker logs pdf-ingest -f
```

---

## 14. Troubleshooting

### Container won't start / build fails

```powershell
# Check what's happening
docker compose ps
docker compose logs <service-name>

# Force a clean rebuild
docker compose build --no-cache <service-name>
docker compose up -d
```

### "ColQwen2 ready" never appears

- The model download may be in progress. Check logs for download progress:
  ```powershell
  docker logs open-webui-pipelines -f
  ```
- If it's stuck, the HuggingFace CDN might be slow. Wait or restart:
  ```powershell
  docker compose restart pipelines
  ```

### Pipeline times out on search

Qdrant uses HNSW approximate search with a 60s timeout. Check Qdrant health:

```powershell
docker logs qdrant --tail 30
```

### Thumbnails don't load / broken image links

- Verify `SERVER_HOST` valve matches your machine's LAN IP
- Test the image server directly: http://localhost:8081
- Check the image cache has files:
  ```powershell
  dir .\pipelines\cache\images\
  ```

### "Connection refused" from other LAN devices

1. Confirm `HOST_IP` in `.env` is correct
2. Check Windows Firewall (see [Section 10](#10-lan-access-from-other-devices))
3. Try `curl http://<HOST_IP>:3000` from the remote device

### First query is slow

Normal. ColQwen2 loads at startup and needs ~60 seconds. The pipeline returns
"Pipeline initializing" until ready. Subsequent queries are faster.

### Everything re-indexes after restart

`pipeline_state.json` is runtime state and gitignored. If it gets deleted or
corrupted, the pipeline re-indexes all PDFs automatically. This is correct
behavior — not a bug.

### Confluence watcher keeps restarting

Check if `CONFLUENCE_PAT` and `CONFLUENCE_URL` are set correctly in `.env`:

```powershell
docker logs confluence-watcher --tail 30
```

If you don't use Confluence, you can stop just that service:

```powershell
docker compose stop confluence-watcher
```

---

## 15. Service Reference

| Service | Container Name | Port | Purpose |
|---------|---------------|------|---------|
| `open-webui` | `open-webui` | **3000** | Chat UI (browser) |
| `pipelines` | `open-webui-pipelines` | 9099 | ColQwen2 model + RAG logic (internal) |
| `qdrant` | `qdrant` | 6333 | Vector database (internal) |
| `image-server` | (auto-named) | **8081** | Serves page thumbnail images (browser) |
| `pdf-ingest` | `pdf-ingest` | **8082** | Upload/delete UI + PDF viewer (browser) |
| `ollama` | `ollama` | 11434 | Local LLM runtime (internal, optional GPU) |
| `confluence-watcher` | `confluence-watcher` | none | Polls Confluence, renders PDFs (internal) |

**Bold ports** are browser-facing — bookmark these.

### Key Files

| File | Purpose |
|------|---------|
| `.env` | Secrets and machine-specific config (gitignored, never committed) |
| `.env.example` | Template — copy to `.env` and fill in values |
| `docker-compose.yml` | Defines all 7 services and their configuration |
| `my-pdfs/` | Drop PDF files here for indexing |
| `pipelines/colpali-pipeline.py` | Core RAG pipeline logic |
| `pipelines/cache/images/` | Cached page images and thumbnails (auto-generated) |
| `pipelines/pipeline_state.json` | Runtime index state (auto-generated, gitignored) |
| `pdf-ingest/main.py` | Upload/delete sidecar API + embedded UI |
| `confluence-watcher/watcher.py` | Confluence polling and PDF rendering |

### Docker Volumes (persistent data)

| Volume | Purpose |
|--------|---------|
| `hf-cache` | ColQwen2 model weights (~5 GB, cached after first download) |
| `qdrant_data` | Vector embeddings for all indexed pages |
| `ollama` | Pulled Ollama models (only if using local GPU VLM) |
| `open-webui` | Open WebUI user accounts and settings |
