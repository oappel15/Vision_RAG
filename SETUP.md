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

Docker Desktop automatically forwards container ports to the Windows LAN IP — no manual port-forwarding or `netsh` configuration needed.

---

## 1. Clone the Repository

```bash
git clone <repo-url> Vision_RAG
cd Vision_RAG
```

---

## 2. Create `.env`

Create `.env` in the project root. This file holds all secrets and machine-specific config — it is gitignored and never committed.

```env
# ── VLM backend ───────────────────────────────────────────────────────────────
# Required if using OpenRouter (cloud VLM). Get a key at https://openrouter.ai
OPENROUTER_API_KEY=sk-or-v1-...

# ── Confluence watcher (optional) ────────────────────────────────────────────
# Generate a PAT in Confluence: Profile → Settings → Personal Access Tokens
CONFLUENCE_PAT=<your-confluence-personal-access-token>

# Confluence server URL — use the externally reachable address of the server
# If Confluence is on the same machine as Docker: http://host.docker.internal:8090
# If Confluence is on a separate server:          http://confluence.yourcompany.com
CONFLUENCE_URL=http://confluence.yourcompany.com

# Comma-separated Space Keys to index (visible in Confluence: Space Settings → Space Key)
CONFLUENCE_SPACES=RAG,ENG

# How often the watcher polls for changes (seconds)
CONFLUENCE_POLL_INTERVAL=60

# ── LAN access ────────────────────────────────────────────────────────────────
# This machine's LAN IP — used as the startup default for SERVER_HOST valve.
# Find it: ip route get 1 | awk '{print $7; exit}'   (Linux/WSL2)
#          ipconfig                                    (Windows → IPv4 Address)
HOST_IP=<this-machine-lan-ip>
```

> **Note:** `OPENROUTER_API_KEY` is read exclusively from `.env` — it does not appear in the pipeline UI valves.
> `HOST_IP` is only the startup default for `SERVER_HOST`. You can override it live in the pipeline settings (see Step 5).

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
   - Controls where browsers fetch thumbnails (port 8081) and PDFs (port 8082)
   - Use `localhost` if you only need single-machine access
5. Set **`VLM_PROVIDER`** using the dropdown: `openrouter` (cloud) or `ollama` (local GPU)
   - For `openrouter`: ensure `OPENROUTER_API_KEY` is set in `.env` (not in valves)
   - For `ollama`: see [Switching VLM Backend](#switching-vlm-backend-openrouter--local-ollama) below
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
| `.env` | Secrets and machine-specific config (gitignored) |
| `pipelines/colpali-pipeline/valves.json` | Persisted pipeline valve settings (gitignored) |
| `pipelines/pipeline_state.json` | Runtime index state — which PDFs are indexed (gitignored) |
| `my-pdfs/` | PDF source directory (indexed automatically) |
| `pipelines/cache/images/` | Cached page images and thumbnails |
| `confluence-watcher/watcher_state.json` | Tracks Confluence page versions |

> All three gitignored files are created automatically at runtime — do not copy them between machines.

---

## Confluence Watcher Setup

The watcher renders Confluence pages to PDF and indexes them automatically.
Any Confluence user account with read access to the target spaces is sufficient — admin rights are not required.

---

### Step A — Generate a Personal Access Token (PAT) in Confluence

1. Log in to your Confluence Server instance
2. Click your **profile avatar** (top-right corner)
3. Go to **Profile → Settings** (left sidebar)
4. Click **Personal Access Tokens** in the left menu
5. Click **Create token**
6. Give it a name (e.g. `vision-rag-watcher`), set expiry if required, click **Create**
7. **Copy the token immediately** — it is only shown once

> PATs are available on Confluence Server **7.9 and later**. If your instance is older, contact the admin to enable basic auth and adjust the watcher's `_auth_headers()` function to use `Authorization: Basic base64(user:password)`.

---

### Step B — Add the PAT to `.env`

The `.env` file lives in the **project root directory**:

```
Vision_RAG/          ← project root
├── .env             ← put CONFLUENCE_PAT here
├── docker-compose.yml
├── confluence-watcher/
└── ...
```

Open `.env` and set:

```env
CONFLUENCE_PAT=<paste-your-token-here>
```

Full example `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-...
CONFLUENCE_PAT=NDYzNTExNzk2NTU0OgN...
CONFLUENCE_URL=http://confluence.yourcompany.com
CONFLUENCE_SPACES=RAG,ENG
CONFLUENCE_POLL_INTERVAL=60
HOST_IP=10.0.1.5
```

> `.env` is gitignored — it never gets committed. Keep it on the machine only.

---

### Step C — Configure the target Confluence spaces

Set `CONFLUENCE_URL` and `CONFLUENCE_SPACES` in `.env`:

```env
# Confluence hosted by an external team on a separate server:
CONFLUENCE_URL=http://confluence.yourcompany.com

# Comma-separated Space Keys to index
CONFLUENCE_SPACES=RAG,ENG
```

- Space Keys are visible in Confluence under **Space Settings → Space Key**
- If Confluence happens to run on the same machine as Docker, use `http://host.docker.internal:8090`

---

### Step D — Restart the watcher

```bash
docker compose up -d confluence-watcher
```

The watcher polls on startup and then every `CONFLUENCE_POLL_INTERVAL` seconds. On first run with a fresh state it indexes all pages in the configured spaces automatically.

---

## Switching VLM Backend: OpenRouter → Local Ollama

By default the pipeline sends page images to **OpenRouter** (cloud). On a machine with a capable GPU you can switch to a **local Ollama** instance instead — no internet required, no API key needed.

### Requirements

| Item | Detail |
|---|---|
| GPU VRAM | ~24 GB for `qwen3-vl:30b-a3b-instruct` (e.g. RTX 3090 / 4090 / A100) |
| Ollama | Already included as a service in `docker-compose.yml` |
| Model pulled | Must pull the model before switching |

### Step 1 — Pull the model into Ollama

```bash
docker exec -it ollama ollama pull qwen3-vl:30b-a3b-instruct
```

This downloads ~20 GB on first run. Check progress:

```bash
docker exec -it ollama ollama list
```

### Step 2 — Switch the VLM_PROVIDER valve in Open WebUI

1. Open WebUI → **Settings → Admin → Pipelines → ColQwen2 Visual RAG**
2. Set **`VLM_PROVIDER`** → `ollama` (dropdown)
3. Confirm **`OLLAMA_VLM_MODEL`** matches the pulled model name (default: `qwen3-vl:30b-a3b-instruct`)
4. **`OLLAMA_BASE_URL`** should already be `http://ollama:11434` (no change needed)
5. Save — takes effect immediately, no restart needed

### Step 3 — Verify

Ask a question in Open WebUI. The answer should come back without any OpenRouter API calls.
Check Ollama logs to confirm: `docker logs ollama -f`

### Switching back to OpenRouter

Set **`VLM_PROVIDER`** back to `openrouter` in the pipeline valves. Ensure `OPENROUTER_API_KEY` is set in `.env`.

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

**Everything re-indexes from scratch after a restart:**
`pipeline_state.json` is gitignored runtime state — do not copy it between machines or restore it from git. If it gets corrupted or lost, the pipeline will re-index all PDFs automatically (which is correct behavior on a fresh machine).
