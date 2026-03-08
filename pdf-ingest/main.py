"""
pdf-ingest sidecar service
Saves PDFs to shared volume and triggers background indexing in the pipelines container.
No ML dependencies — purely file I/O + HTTP.
"""

import json
import logging
import os
import pathlib

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

log = logging.getLogger("pdf-ingest")

app = FastAPI()

PDF_DIR = pathlib.Path(os.getenv("PDF_DIR", "/app/downloads"))
STATE_FILE = pathlib.Path(os.getenv("STATE_FILE", "/app/pipelines/pipeline_state.json"))
WATCHER_STATE_FILE = pathlib.Path(os.getenv("WATCHER_STATE_FILE", "/app/watcher_state/watcher_state.json"))
PIPELINES_URL = os.getenv("PIPELINES_URL", "http://pipelines:9099")
PIPELINES_API_KEY = os.getenv("PIPELINES_API_KEY", "0p3n-w3bu!")
PIPELINE_MODEL = os.getenv("PIPELINE_MODEL", "colpali-pipeline")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_docs")

PDF_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")


@app.get("/view/{page}/{filename:path}", response_class=HTMLResponse)
def view_pdf_at_page(page: int, filename: str):
    """Render a specific PDF page using PDF.js — no browser fragment support required."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<title>{filename} — Page {page}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#525659;display:flex;flex-direction:column;height:100vh;font-family:sans-serif}}
  #bar{{background:#3d4043;color:#e8eaed;padding:8px 16px;display:flex;align-items:center;gap:10px;font-size:14px;flex-shrink:0}}
  #bar button{{background:#5f6368;color:#e8eaed;border:none;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:13px}}
  #bar button:hover{{background:#8ab4f8;color:#000}}
  #bar button:disabled{{opacity:.4;cursor:default}}
  #bar .info{{color:#9aa0a6;margin-left:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  #main{{flex:1;overflow:auto;display:flex;flex-direction:column;align-items:center;padding:20px;gap:12px}}
  canvas{{background:#fff;box-shadow:0 2px 12px rgba(0,0,0,.5)}}
  #loading{{color:#e8eaed;font-size:15px;margin-top:40px}}
</style>
</head>
<body>
<div id="bar">
  <button id="btnPrev" onclick="go(-1)" disabled>&#9664; Prev</button>
  <button id="btnNext" onclick="go(1)" disabled>Next &#9654;</button>
  <span id="pageLabel">Page {page}</span>
  <span class="info">{filename}</span>
</div>
<div id="main"><div id="loading">Loading…</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<script>
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  let pdf = null, cur = {page};
  const main = document.getElementById('main');

  pdfjsLib.getDocument('/pdfs/{filename}').promise.then(doc => {{
    pdf = doc;
    render(cur);
  }}).catch(e => {{
    document.getElementById('loading').textContent = 'Failed to load PDF: ' + e.message;
  }});

  function render(n) {{
    main.innerHTML = '<div id="loading">Rendering…</div>';
    pdf.getPage(n).then(page => {{
      const vp = page.getViewport({{scale: 1.5}});
      const canvas = document.createElement('canvas');
      canvas.width = vp.width; canvas.height = vp.height;
      main.innerHTML = '';
      main.appendChild(canvas);
      page.render({{canvasContext: canvas.getContext('2d'), viewport: vp}});
      document.getElementById('pageLabel').textContent = `Page ${{n}} of ${{pdf.numPages}}`;
      document.getElementById('btnPrev').disabled = n <= 1;
      document.getElementById('btnNext').disabled = n >= pdf.numPages;
    }});
  }}

  function go(d) {{
    const n = cur + d;
    if (pdf && n >= 1 && n <= pdf.numPages) {{ cur = n; render(cur); }}
  }}
</script>
</body>
</html>"""


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = PDF_DIR / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Clear skip flag so a re-uploaded previously-cancelled file gets indexed
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            if file.filename in state.get("skipped_files", []):
                state["skipped_files"].remove(file.filename)
                tmp = str(STATE_FILE) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(state, f)
                os.replace(tmp, str(STATE_FILE))
        except Exception:
            pass

    # Fire-and-forget: trigger background indexing via existing pipeline API
    try:
        requests.post(
            f"{PIPELINES_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {PIPELINES_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": PIPELINE_MODEL,
                "messages": [{"role": "user", "content": "__index_now__"}],
            },
            timeout=5,
        )
    except Exception:
        pass  # pipeline may still be starting up; indexing will run on next startup too

    return {"status": "uploaded", "filename": file.filename, "size_bytes": len(content)}


@app.get("/status")
def get_status():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return JSONResponse(json.load(f))
    return JSONResponse({})


@app.get("/queue")
def get_queue():
    """Return files waiting to be indexed (queued + paused), ordered as the indexer would pick them."""
    state = {}
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
        except Exception:
            pass

    indexed = set(state.get("indexed_files", []))
    skipped = set(state.get("skipped_files", []))
    file_progress = state.get("file_progress", {})
    job = state.get("index_job", {})
    active_file = job.get("current_file") if job.get("active") else None

    queue = []
    for pdf in sorted(PDF_DIR.rglob("*.pdf")):
        rel = str(pdf.relative_to(PDF_DIR))
        if rel in indexed or rel in skipped:
            continue
        if rel == active_file:
            continue  # already shown in progress bar
        if rel in file_progress:
            queue.append({"filename": rel, "status": "paused", "resume_page": file_progress[rel]})
        else:
            queue.append({"filename": rel, "status": "queued"})
    return queue


def _send_pipeline_command(command: str):
    try:
        requests.post(
            f"{PIPELINES_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {PIPELINES_API_KEY}", "Content-Type": "application/json"},
            json={"model": PIPELINE_MODEL, "messages": [{"role": "user", "content": command}]},
            timeout=5,
        )
    except Exception:
        pass


@app.post("/cancel")
def cancel_indexing():
    """Hard cancel — stops indexing, clears progress, removes partial Qdrant vectors."""
    _send_pipeline_command("__cancel_index__")
    return {"status": "cancel_requested"}


@app.post("/pause")
def pause_indexing():
    """Pause — stops indexing and saves progress so it resumes on next run."""
    _send_pipeline_command("__pause_index__")
    return {"status": "pause_requested"}


@app.delete("/delete/{filename}")
def delete_pdf(filename: str):
    pdf_path = PDF_DIR / filename
    if pdf_path.exists():
        pdf_path.unlink()

    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        changed = False
        if filename in state.get("indexed_files", []):
            state["indexed_files"].remove(filename)
            changed = True
        if filename in state.get("skipped_files", []):
            state["skipped_files"].remove(filename)
            changed = True
        if changed:
            tmp = str(STATE_FILE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, str(STATE_FILE))

    # Remove vectors from Qdrant so stale embeddings don't persist
    qdrant_ok = True
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=filename))]
                )
            ),
        )
        log.info(f"Qdrant vectors deleted for source={filename}")
    except Exception as e:
        qdrant_ok = False
        log.error(f"Qdrant delete FAILED for {filename}: {e}")

    # Remove from watcher_state.json so the Confluence watcher won't re-sync this file
    if WATCHER_STATE_FILE.exists():
        try:
            with open(WATCHER_STATE_FILE) as f:
                wstate = json.load(f)
            changed = False
            for space_key in list(wstate.keys()):
                for page_id in list(wstate[space_key].keys()):
                    if wstate[space_key][page_id].get("pdf_filename") == filename:
                        del wstate[space_key][page_id]
                        changed = True
                        log.info(f"Removed {filename} from watcher_state ({space_key}/{page_id})")
            if changed:
                tmp = str(WATCHER_STATE_FILE) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(wstate, f, indent=2)
                os.replace(tmp, str(WATCHER_STATE_FILE))
        except Exception as e:
            log.warning(f"Failed to clean watcher_state for {filename}: {e}")

    return {"status": "deleted", "filename": filename, "qdrant_cleaned": qdrant_ok}


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Vision RAG — PDF Indexer</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d1117;
      --surface:   #161b22;
      --surface2:  #21262d;
      --border:    #30363d;
      --accent:    #7c3aed;
      --accent-hi: #a78bfa;
      --accent-bg: rgba(124,58,237,.12);
      --green:     #3fb950;
      --green-bg:  rgba(63,185,80,.12);
      --red:       #f85149;
      --red-bg:    rgba(248,81,73,.12);
      --amber:     #d29922;
      --amber-bg:  rgba(210,153,34,.12);
      --text:      #e6edf3;
      --text-muted:#8b949e;
      --radius:    12px;
      --radius-sm: 8px;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 0;
    }

    /* ── Layout ── */
    .shell {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }

    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 32px;
      height: 60px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 10;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 15px;
      font-weight: 600;
      letter-spacing: -.2px;
    }

    .logo-icon {
      width: 28px; height: 28px;
      background: linear-gradient(135deg, var(--accent), #4f46e5);
      border-radius: 7px;
      display: flex; align-items: center; justify-content: center;
      font-size: 15px;
    }

    #statusDot {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-muted);
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 5px 12px;
    }

    .dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--green);
      flex-shrink: 0;
    }
    .dot.busy { background: var(--amber); animation: pulse 1.4s ease-in-out infinite; }
    .dot.err  { background: var(--red); }

    @keyframes pulse {
      0%,100% { opacity: 1; }
      50%      { opacity: .35; }
    }

    main {
      max-width: 860px;
      margin: 0 auto;
      padding: 36px 24px 60px;
      width: 100%;
    }

    /* ── Cards ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 20px;
    }

    .card-title {
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--text-muted);
      margin-bottom: 18px;
    }

    /* ── Drop zone ── */
    #dropzone {
      border: 2px dashed var(--border);
      border-radius: var(--radius-sm);
      padding: 44px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
      position: relative;
    }
    #dropzone:hover, #dropzone.over {
      border-color: var(--accent);
      background: var(--accent-bg);
    }
    #dropzone input[type=file] {
      position: absolute; inset: 0; opacity: 0; cursor: pointer;
    }
    .drop-icon { font-size: 36px; margin-bottom: 12px; }
    .drop-label {
      font-size: 15px; font-weight: 500; color: var(--text);
      margin-bottom: 4px;
    }
    .drop-sub { font-size: 13px; color: var(--text-muted); }
    #selectedFile {
      margin-top: 12px; font-size: 13px; color: var(--accent-hi);
      font-weight: 500; min-height: 18px;
    }

    /* ── Upload button ── */
    #uploadBtn {
      margin-top: 16px;
      width: 100%;
      background: linear-gradient(135deg, var(--accent), #4f46e5);
      color: #fff;
      border: none;
      border-radius: var(--radius-sm);
      padding: 13px 20px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity .15s, transform .1s;
      display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    #uploadBtn:hover  { opacity: .88; }
    #uploadBtn:active { transform: scale(.98); }
    #uploadBtn:disabled { opacity: .4; cursor: default; }

    /* ── Toast ── */
    #toast {
      position: fixed;
      bottom: 28px; left: 50%;
      transform: translateX(-50%) translateY(80px);
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 20px;
      font-size: 14px;
      font-weight: 500;
      opacity: 0;
      transition: transform .3s cubic-bezier(.34,1.56,.64,1), opacity .3s;
      z-index: 100;
      white-space: nowrap;
      max-width: 90vw;
    }
    #toast.show {
      transform: translateX(-50%) translateY(0);
      opacity: 1;
    }
    #toast.ok  { border-color: var(--green);  color: var(--green);  }
    #toast.err { border-color: var(--red);    color: var(--red);    }

    /* ── Progress ── */
    #progressWrap { display: none; }
    #progressWrap.visible { display: block; }

    .progress-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 10px;
    }
    .progress-file {
      font-size: 14px; font-weight: 500;
      color: var(--accent-hi);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      max-width: 70%;
    }
    .progress-pct {
      font-size: 13px; font-weight: 600; color: var(--text-muted);
    }
    .progress-track {
      height: 6px;
      background: var(--surface2);
      border-radius: 99px;
      overflow: hidden;
      margin-bottom: 8px;
    }
    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-hi));
      border-radius: 99px;
      transition: width .5s ease;
      width: 0%;
    }
    .progress-footer {
      display: flex; align-items: center; justify-content: space-between; margin-top: 6px;
    }
    .progress-pages {
      font-size: 12px; color: var(--text-muted);
    }
    .ctrl-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: var(--radius-sm);
      padding: 4px 12px;
      font-size: 12px;
      cursor: pointer;
      transition: background .15s, color .15s, border-color .15s;
    }
    #pauseBtn:hover {
      background: var(--amber-bg);
      color: var(--amber);
      border-color: rgba(210,153,34,.3);
    }
    #cancelBtn:hover {
      background: var(--red-bg);
      color: var(--red);
      border-color: rgba(248,81,73,.3);
    }
    .ctrl-btn:disabled { opacity: .4; cursor: default; }

    .idle-badge {
      display: inline-flex; align-items: center; gap: 6px;
      background: var(--green-bg);
      color: var(--green);
      font-size: 13px; font-weight: 500;
      padding: 5px 12px;
      border-radius: 20px;
      border: 1px solid rgba(63,185,80,.25);
    }

    /* ── File library ── */
    #fileList { list-style: none; }
    #fileList li {
      display: flex; align-items: center; gap: 12px;
      padding: 11px 0;
      border-bottom: 1px solid var(--border);
    }
    #fileList li:last-child { border-bottom: none; }

    .file-icon {
      width: 34px; height: 34px; flex-shrink: 0;
      background: var(--accent-bg);
      border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
    }
    .file-name {
      flex: 1; font-size: 14px; font-weight: 500;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .file-badge {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: .05em;
      background: var(--green-bg); color: var(--green);
      border: 1px solid rgba(63,185,80,.2);
      border-radius: 20px; padding: 2px 9px;
      flex-shrink: 0;
    }

    .btn-del {
      flex-shrink: 0;
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: var(--radius-sm);
      padding: 5px 10px;
      font-size: 12px;
      cursor: pointer;
      transition: background .15s, color .15s, border-color .15s;
    }
    .btn-del:hover {
      background: var(--red-bg);
      color: var(--red);
      border-color: rgba(248,81,73,.3);
    }

    .empty-state {
      text-align: center; padding: 32px 0;
      color: var(--text-muted); font-size: 14px;
    }
    .empty-state .empty-icon { font-size: 32px; margin-bottom: 8px; }

    /* ── Queue list ── */
    #queueList li {
      display: flex; align-items: center; gap: 8px;
      font-size: 13px; color: var(--text-muted);
      padding: 3px 0;
    }
    #queueList .q-icon { flex-shrink: 0; font-size: 13px; }
    #queueList .q-name {
      flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      color: var(--text);
    }
    #queueList .q-badge {
      font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em;
      padding: 1px 7px; border-radius: 20px; flex-shrink: 0;
    }
    .q-badge.queued  { background: var(--surface2); color: var(--text-muted); border: 1px solid var(--border); }
    .q-badge.paused  { background: var(--accent-bg); color: var(--accent-hi); border: 1px solid rgba(124,58,237,.25); }
  </style>
</head>
<body>
<div class="shell">

  <header>
    <div class="logo">
      <div class="logo-icon">📄</div>
      Vision RAG — PDF Indexer
    </div>
    <div id="statusDot">
      <span class="dot" id="dot"></span>
      <span id="dotLabel">Checking…</span>
    </div>
  </header>

  <main>

    <!-- Upload card -->
    <div class="card">
      <div class="card-title">Upload Document</div>

      <div id="dropzone">
        <input type="file" id="fileInput" accept=".pdf" />
        <div class="drop-icon">📂</div>
        <div class="drop-label">Drop a PDF here or click to browse</div>
        <div class="drop-sub">Only PDF files · any size</div>
        <div id="selectedFile"></div>
      </div>

      <button id="uploadBtn" disabled onclick="uploadFile()">
        <span id="uploadBtnIcon">⬆</span>
        <span id="uploadBtnLabel">Select a file first</span>
      </button>
    </div>

    <!-- Indexing status card -->
    <div class="card">
      <div class="card-title">Indexing Status</div>
      <div id="progressWrap">
        <div class="progress-header">
          <div class="progress-file" id="progFile">—</div>
          <div class="progress-pct" id="progPct">0%</div>
        </div>
        <div class="progress-track">
          <div class="progress-bar" id="progBar"></div>
        </div>
        <div class="progress-footer">
          <div class="progress-pages" id="progPages"></div>
          <button id="pauseBtn"  class="ctrl-btn" onclick="pauseIndexing()">⏸ Pause</button>
          <button id="cancelBtn" class="ctrl-btn" onclick="cancelIndexing()">✕ Cancel</button>
        </div>
      </div>
      <div id="idleWrap"><div class="idle-badge">● Idle — all documents indexed</div></div>
      <div id="queueWrap" style="display:none; margin-top:14px; border-top:1px solid var(--border); padding-top:12px;">
        <div style="font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); margin-bottom:8px;">Up next</div>
        <ul id="queueList" style="list-style:none; display:flex; flex-direction:column; gap:5px;"></ul>
      </div>
    </div>

    <!-- File library card -->
    <div class="card">
      <div class="card-title">Document Library</div>
      <ul id="fileList"></ul>
      <div class="empty-state" id="emptyState" style="display:none">
        <div class="empty-icon">🗂</div>
        No documents indexed yet.<br>Upload a PDF above to get started.
      </div>
    </div>

  </main>
</div>

<!-- Toast -->
<div id="toast"></div>

<script>
  // ── Drag-and-drop ──────────────────────────────────────────────────
  const dz = document.getElementById('dropzone');
  const fi = document.getElementById('fileInput');

  dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    if (e.dataTransfer.files.length) {
      fi.files = e.dataTransfer.files;
      onFileSelected();
    }
  });
  fi.addEventListener('change', onFileSelected);

  function onFileSelected() {
    const f = fi.files[0];
    if (!f) return;
    document.getElementById('selectedFile').textContent = `${f.name}  (${(f.size/1024).toFixed(1)} KB)`;
    const btn = document.getElementById('uploadBtn');
    btn.disabled = false;
    document.getElementById('uploadBtnLabel').textContent = `Upload & Index  "${f.name}"`;
  }

  // ── Upload ─────────────────────────────────────────────────────────
  async function uploadFile() {
    const f = fi.files[0];
    if (!f) return;
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true;
    document.getElementById('uploadBtnLabel').textContent = 'Uploading…';
    document.getElementById('uploadBtnIcon').textContent = '⏳';

    const form = new FormData();
    form.append('file', f);
    try {
      const r = await fetch('/upload', { method: 'POST', body: form });
      const d = await r.json();
      if (r.ok) {
        toast(`✓ ${d.filename} uploaded — indexing started`, 'ok');
        fi.value = '';
        document.getElementById('selectedFile').textContent = '';
        document.getElementById('uploadBtnLabel').textContent = 'Select a file first';
        document.getElementById('uploadBtnIcon').textContent = '⬆';
        // Show queued state immediately — don't wait for next poll
        showQueued(d.filename);
      } else {
        toast(`Error: ${d.detail}`, 'err');
        btn.disabled = false;
        document.getElementById('uploadBtnLabel').textContent = 'Retry Upload';
        document.getElementById('uploadBtnIcon').textContent = '⬆';
      }
    } catch (e) {
      toast(`Upload failed: ${e}`, 'err');
      btn.disabled = false;
      document.getElementById('uploadBtnLabel').textContent = 'Retry Upload';
      document.getElementById('uploadBtnIcon').textContent = '⬆';
    }
    refresh();
  }

  // ── Delete ─────────────────────────────────────────────────────────
  async function deletePdf(filename) {
    if (!confirm(`Remove "${filename}" from the index and delete from disk?`)) return;
    // Optimistic: remove from DOM immediately without waiting for API or next poll
    const li = document.querySelector(`li[data-file="${CSS.escape(filename)}"]`);
    if (li) li.remove();
    checkEmptyState();
    const r = await fetch('/delete/' + encodeURIComponent(filename), { method: 'DELETE' });
    if (r.ok) {
      toast(`Deleted: ${filename}`, 'ok');
    } else {
      toast('Delete failed — refreshing', 'err');
    }
    refresh();
  }

  // ── Show empty state if file list is now empty ────────────────────
  function checkEmptyState() {
    const ul = document.getElementById('fileList');
    const empty = document.getElementById('emptyState');
    empty.style.display = ul.children.length === 0 ? '' : 'none';
  }

  // ── Pause indexing (saves progress, resumes next run) ──────────────
  async function pauseIndexing() {
    const btn = document.getElementById('pauseBtn');
    btn.disabled = true;
    btn.textContent = 'Pausing…';
    await fetch('/pause', { method: 'POST' });
    toast('Paused — will resume from this point next run', 'ok');
  }

  // ── Cancel indexing (clears progress, no resume) ───────────────────
  async function cancelIndexing() {
    const btn = document.getElementById('cancelBtn');
    btn.disabled = true;
    btn.textContent = 'Cancelling…';
    await fetch('/cancel', { method: 'POST' });
    toast('Cancelled — progress cleared, partial vectors removed', 'ok');
  }

  // ── Immediate queued feedback (before first poll returns active state) ──
  function showQueued(filename) {
    document.getElementById('dot').className = 'dot busy';
    document.getElementById('dotLabel').textContent = 'Indexing…';
    const pw = document.getElementById('progressWrap');
    document.getElementById('idleWrap').style.display = 'none';
    pw.classList.add('visible');
    document.getElementById('progFile').textContent = filename;
    document.getElementById('progPct').textContent = '0%';
    document.getElementById('progBar').style.width = '0%';
    document.getElementById('progPages').textContent = 'Starting…';
  }

  // ── Queue ──────────────────────────────────────────────────────────
  async function refreshQueue() {
    try {
      const r = await fetch('/queue');
      const queue = await r.json();
      const wrap = document.getElementById('queueWrap');
      const ul   = document.getElementById('queueList');
      if (!queue.length) { wrap.style.display = 'none'; return; }
      wrap.style.display = '';
      ul.innerHTML = '';
      queue.forEach(f => {
        const li = document.createElement('li');
        const sub = f.status === 'paused' ? ` · resumed from p.${f.resume_page}` : '';
        const badgeClass = f.status === 'paused' ? 'paused' : 'queued';
        li.innerHTML = `
          <span class="q-icon">📄</span>
          <span class="q-name" title="${f.filename}">${f.filename}${sub}</span>
          <span class="q-badge ${badgeClass}">${f.status}</span>
        `;
        ul.appendChild(li);
      });
    } catch (_) {}
  }

  // ── Status poll ────────────────────────────────────────────────────
  async function refreshStatus() {
    try {
      const r = await fetch('/status');
      const data = await r.json();
      const job = data.index_job || {};
      const indexed = data.indexed_files || [];

      // Header dot
      const dot = document.getElementById('dot');
      const dotLabel = document.getElementById('dotLabel');
      if (job.active) {
        dot.className = 'dot busy';
        dotLabel.textContent = 'Indexing…';
      } else {
        dot.className = 'dot';
        dotLabel.textContent = `${indexed.length} doc${indexed.length !== 1 ? 's' : ''} indexed`;
      }

      // Progress section
      const pw = document.getElementById('progressWrap');
      const iw = document.getElementById('idleWrap');
      if (job.active) {
        pw.classList.add('visible');
        iw.style.display = 'none';
        const pct = job.total_pages > 0 ? Math.round(job.current_page / job.total_pages * 100) : 0;
        document.getElementById('progFile').textContent = job.current_file;
        document.getElementById('progPct').textContent = pct + '%';
        document.getElementById('progBar').style.width = pct + '%';
        document.getElementById('progPages').textContent =
          job.current_page === 0 ? 'Starting…' : `Page ${job.current_page} of ${job.total_pages}`;
        const pb = document.getElementById('pauseBtn');
        pb.disabled = false; pb.textContent = '⏸ Pause';
        const cb = document.getElementById('cancelBtn');
        cb.disabled = false; cb.textContent = '✕ Cancel';
      } else {
        pw.classList.remove('visible');
        iw.style.display = '';
      }

      // File list
      const ul = document.getElementById('fileList');
      const empty = document.getElementById('emptyState');
      ul.innerHTML = '';
      if (indexed.length === 0) {
        empty.style.display = '';
      } else {
        empty.style.display = 'none';
        indexed.forEach(fn => {
          const esc = fn.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
          const li = document.createElement('li');
          li.dataset.file = fn;
          li.innerHTML = `
            <div class="file-icon">📄</div>
            <div class="file-name" title="${fn}">${fn}</div>
            <span class="file-badge">indexed</span>
            <button class="btn-del" onclick="deletePdf('${esc}')">Remove</button>
          `;
          ul.appendChild(li);
        });
      }
    } catch (e) {
      document.getElementById('dotLabel').textContent = 'Offline';
      document.getElementById('dot').className = 'dot err';
    }
  }

  // ── Toast helper ───────────────────────────────────────────────────
  let toastTimer;
  function toast(msg, type = 'ok') {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = `show ${type}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = ''; }, 3500);
  }

  function refresh() { refreshStatus(); refreshQueue(); }
  refresh();
  setInterval(refresh, 3000);
</script>
</body>
</html>
"""
