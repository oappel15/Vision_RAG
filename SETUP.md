# Vision RAG — Setup Guide

A self-hosted visual RAG system using ColQwen2 + Qdrant + Open WebUI.
Optionally syncs Confluence pages automatically via a watcher service.

---

## Prerequisites

Install on the new machine before anything else:

| Requirement | Notes |
|---|---|
| **Docker Desktop** | Includes Docker Compose. Enable WSL2 backend in settings. |
| **WSL2** (Windows) | Ubuntu distro recommended. Docker Desktop integrates with it automatically. |
| **Git** | To clone the repo |
| **Python 3.x** | Only needed if you run local scripts — not required for Docker-only setup |

Docker Desktop automatically forwards container ports to the Windows LAN IP — no manual port-forwarding or `netsh` configuration needed.

---

## 1. Clone the Repository

```bash
git clone <repo-url> Vision_RAG
cd Vision_RAG
```

---

## 2. Create `.env`

Copy `.env.example` to `.env` (or create from scratch). The file must live in the project root.

```env
# Required — get from https://openrouter.ai
OPENROUTER_API_KEY=sk-or-v1-...

# Required if using Confluence watcher — generate a PAT in Confluence:
# Profile → Settings → Personal Access Tokens → Create token
CONFLUENCE_PAT=<your-confluence-personal-access-token>

# How often the Confluence watcher polls for changes (seconds)
CONFLUENCE_POLL_INTERVAL=60

# The LAN IP of this machine — used as the startup default for SERVER_HOST valve
# Find it: ip route get 1 | awk '{print $7; exit}' (Linux/WSL2)
#          ipconfig (Windows, look for IPv4 Address)
HOST_IP=<this-machine-lan-ip>
```

> **Note:** `HOST_IP` is only the startup default. You can override it live in the Open WebUI pipeline settings after startup (see Step 5).

---

## 3. Start All Services

```bash
docker compose up -d
```

First run downloads model weights (~5 GB for ColQwen2) and builds images — this takes several minutes.

Check all containers are running:

```bash
docker compose ps
```

Expected services: `open-webui`, `open-webui-pipelines`, `pdf-ingest`, `image-server`, `qdrant`, `ollama`, `confluence-watcher`

---

## 4. Wait for the Pipeline to Initialize

The ColQwen2 model loads at startup. Watch for the ready signal:

```bash
docker logs open-webui-pipelines -f 2>&1 | grep "ColQwen2 ready"
```

This typically takes 1–3 minutes on first run (model is cached in a Docker volume after that).

---

## 5. Configure the Pipeline in Open WebUI

Open **`http://<HOST_IP>:3000`** in a browser.

1. Sign in (create an admin account on first run)
2. Go to **Settings → Admin → Pipelines**
3. Find **ColQwen2 Visual RAG** and click the settings icon
4. Set **`SERVER_HOST`** to this machine's LAN IP (e.g. `10.0.1.5`)
   - This controls where browsers fetch thumbnails (port 8081) and PDFs (port 8082)
   - Use `localhost` if you only need single-machine access
5. Set **`OPENROUTER_API_KEY`** if it wasn't picked up from `.env`
6. Save

> **Moving to a new LAN?** Only `SERVER_HOST` needs updating — change it here and all image/citation links adapt immediately with no restart.

---

## 6. Add PDFs to Index

**Option A — Upload via UI:**
Open `http://<HOST_IP>:8082` → drag and drop PDF files → click **Index Now**

**Option B — Copy files directly:**
```bash
cp your-file.pdf ./my-pdfs/
# Then trigger indexing via Open WebUI chat: type "__index_now__"
```

**Option C — Confluence sync (automatic):**
Configure `CONFLUENCE_URL` and `CONFLUENCE_PAT` in `.env`. The watcher polls every `CONFLUENCE_POLL_INTERVAL` seconds and indexes new/changed pages automatically.

---

## 7. Verify Everything Works

| Check | How |
|---|---|
| Open WebUI loads | `http://<HOST_IP>:3000` |
| Indexing status | Type `status` in the chat |
| RAG query works | Ask a question about an indexed document |
| Thumbnails load | Should appear below the answer |
| PDF links work | Click a citation — should open the source page |
| LAN access | Open `http://<HOST_IP>:3000` from another device |

---

## Service Map

| Service | Port | Purpose |
|---|---|---|
| `open-webui` | 3000 | Chat UI (browser-facing) |
| `pdf-ingest` | 8082 | Upload UI + PDF viewer (browser-facing) |
| `image-server` | 8081 | Serves cached page thumbnails (browser-facing) |
| `pipelines` | 9099 | ColQwen2 model + Qdrant search (internal) |
| `qdrant` | 6333 | Vector database (internal) |
| `ollama` | 11434 | Local LLM host (internal) |
| `confluence-watcher` | — | Polls Confluence, renders PDFs, triggers indexing (internal) |

---

## Key Files

| File | Purpose |
|---|---|
| `.env` | Secrets and machine-specific config |
| `pipelines/colpali-pipeline/valves.json` | Persisted pipeline valve settings |
| `pipelines/pipeline_state.json` | Tracks which PDFs are indexed |
| `my-pdfs/` | PDF source directory (indexed automatically) |
| `pipelines/cache/images/` | Cached page images and thumbnails |
| `confluence-watcher/watcher_state.json` | Tracks Confluence page versions |

---

## Confluence Watcher Setup

The watcher renders Confluence pages to PDF and indexes them automatically.

**Requirements:**
- Confluence Server (tested with v8.x) accessible from Docker
- Personal Access Token with read access to target spaces

**Environment variables** (in `.env`):
```env
CONFLUENCE_PAT=<token>
CONFLUENCE_POLL_INTERVAL=60   # seconds between polls
```

**Docker Compose environment** (already set in `docker-compose.yml`):
```yaml
CONFLUENCE_URL: http://host.docker.internal:8090   # adjust port if needed
CONFLUENCE_SPACES: RAG                              # comma-separated space keys
```

> `host.docker.internal` resolves to the Windows host from inside Docker. If Confluence runs on a different machine, use its IP directly.

---

## Troubleshooting

**Pipeline times out on search:**
Qdrant query uses HNSW approximate search (`exact=False`) with a 60s timeout. If still timing out, check Qdrant is healthy: `docker logs qdrant`

**Thumbnails show stale content after a Confluence update:**
Thumbnails are auto-regenerated when the source image is newer. If stale, check `docker logs open-webui-pipelines` for indexing errors.

**`SERVER_HOST` links not working from other devices:**
Ensure `SERVER_HOST` in pipeline valves matches this machine's current LAN IP. LAN IPs can change on DHCP — consider assigning a static IP to the server.

**Confluence "URL doesn't match" warning in rendered PDFs:**
The watcher uses `--host-resolver-rules` to navigate Chromium to `localhost:<port>` (matching Confluence's configured base URL) while routing traffic to the Docker host. This is expected and handled automatically.

**First query after restart is slow:**
ColQwen2 loads at startup but needs ~60s. The pipeline returns `"Pipeline initializing"` until ready.
