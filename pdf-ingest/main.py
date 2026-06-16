"""
pdf-ingest sidecar service
Saves PDFs to shared volume and triggers background indexing in the pipelines container.
No ML dependencies — purely file I/O + HTTP.
"""

import json
import logging
import os
import pathlib
import urllib.request
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

log = logging.getLogger("pdf-ingest")

app = FastAPI()

PDF_DIR = pathlib.Path(os.getenv("PDF_DIR", "/app/downloads"))
STATE_FILE = pathlib.Path(os.getenv("STATE_FILE", "/app/pipelines/pipeline_state.json"))
LABELS_FILE = pathlib.Path(os.getenv("LABELS_FILE", "/app/pipelines/labels.json"))
WATCHER_STATE_FILE = pathlib.Path(
    os.getenv("WATCHER_STATE_FILE", "/app/watcher_state/watcher_state.json")
)
IMAGE_CACHE_DIR = pathlib.Path(
    os.getenv("IMAGE_CACHE_DIR", "/app/pipelines/cache/images")
)
PIPELINES_URL = os.getenv("PIPELINES_URL", "http://pipelines:9099")
PIPELINES_API_KEY = os.getenv("PIPELINES_API_KEY", "0p3n-w3bu!")
PIPELINE_MODEL = os.getenv("PIPELINE_MODEL", "colpali-pipeline")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_docs")

PDF_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_cached_images(filename: str) -> int:
    """Remove cached page images and thumbnails for a given PDF filename.
    Returns the number of files removed."""
    if not IMAGE_CACHE_DIR.exists():
        return 0
    stem = pathlib.Path(filename).stem
    removed = 0
    for pattern in (f"{stem}_p*.jpg", f"{stem}_p*_thumb.jpg"):
        for f in IMAGE_CACHE_DIR.glob(pattern):
            f.unlink()
            removed += 1
    if removed:
        log.info(f"Cleaned up {removed} cached image(s) for {filename}")
    return removed


# ── Labels storage ────────────────────────────────────────────────────
# labels.json: { "filename.pdf": ["label1", "label2", ...], ... }
# The filename itself is always an implicit label (added at search time
# by the pipeline), so it is NOT stored redundantly here.


def _load_labels() -> dict:
    if LABELS_FILE.exists():
        try:
            with open(LABELS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_labels(labels: dict):
    tmp = str(LABELS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=2)
    os.replace(tmp, str(LABELS_FILE))


def _build_full_labels(filename: str, user_labels: list = None) -> list:
    """Build the full labels list (auto + user), deduped, matching pipeline logic."""
    stem = pathlib.Path(filename).stem
    combined, seen = [], set()
    for label in [filename, stem] + (user_labels or []):
        lower = label.lower()
        if lower not in seen:
            seen.add(lower)
            combined.append(label)
    return combined


@app.on_event("startup")
def _patch_existing_labels():
    """On startup, patch default labels into Qdrant for any indexed docs missing them."""
    if not STATE_FILE.exists():
        return
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        return

    indexed = state.get("indexed_files", [])
    if not indexed:
        return

    all_labels = _load_labels()
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        for filename in indexed:
            user_labels = all_labels.get(filename, [])
            combined = _build_full_labels(filename, user_labels)
            qdrant.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "labels": combined,
                    "labels_lower": [l.lower() for l in combined],
                },
                points=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="source", match=MatchValue(value=filename)
                            )
                        ]
                    )
                ),
            )
            log.info(f"Startup: patched labels for {filename}: {combined}")
    except Exception as e:
        log.warning(f"Startup label patch failed (Qdrant may not be ready): {e}")


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
async def upload_pdf(
    file: UploadFile = File(...),
    labels: Optional[str] = Form(None),  # JSON array of label strings
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = PDF_DIR / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Parse and store user-supplied labels
    user_labels = []
    if labels:
        try:
            parsed = json.loads(labels)
            if isinstance(parsed, list):
                user_labels = [str(l).strip() for l in parsed if str(l).strip()]
        except (json.JSONDecodeError, TypeError):
            pass

    if user_labels:
        all_labels = _load_labels()
        all_labels[file.filename] = user_labels
        _save_labels(all_labels)

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

    return {
        "status": "uploaded",
        "filename": file.filename,
        "size_bytes": len(content),
        "labels": user_labels,
    }


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
            queue.append(
                {"filename": rel, "status": "paused", "resume_page": file_progress[rel]}
            )
        else:
            queue.append({"filename": rel, "status": "queued"})
    return queue


def _send_pipeline_command(command: str):
    try:
        requests.post(
            f"{PIPELINES_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {PIPELINES_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": PIPELINE_MODEL,
                "messages": [{"role": "user", "content": command}],
            },
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

    _cleanup_cached_images(filename)

    # Remove labels for this file
    all_labels = _load_labels()
    if filename in all_labels:
        del all_labels[filename]
        _save_labels(all_labels)

    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        changed = False
        if filename in state.get("indexed_files", []):
            state["indexed_files"].remove(filename)
            changed = True
        if filename in state.get("file_progress", {}):
            del state["file_progress"][filename]
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
                    must=[
                        FieldCondition(key="source", match=MatchValue(value=filename))
                    ]
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
                        log.info(
                            f"Removed {filename} from watcher_state ({space_key}/{page_id})"
                        )
            if changed:
                tmp = str(WATCHER_STATE_FILE) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(wstate, f, indent=2)
                os.replace(tmp, str(WATCHER_STATE_FILE))
        except Exception as e:
            log.warning(f"Failed to clean watcher_state for {filename}: {e}")

    return {"status": "deleted", "filename": filename, "qdrant_cleaned": qdrant_ok}


# ── Labels API ────────────────────────────────────────────────────────


@app.get("/labels")
def get_all_labels():
    """Return all labels for all files: { filename: [labels], ... }"""
    return JSONResponse(_load_labels())


@app.get("/labels/{filename}")
def get_file_labels(filename: str):
    """Return labels for a specific file."""
    all_labels = _load_labels()
    return JSONResponse(all_labels.get(filename, []))


@app.put("/labels/{filename}")
async def update_file_labels(filename: str, request_body: dict):
    """Update labels for a specific file. Body: {"labels": ["label1", ...]}
    Also patches labels directly into Qdrant so existing vectors are immediately filterable.
    """
    all_labels = _load_labels()
    new_labels = request_body.get("labels", []) if request_body else []
    new_labels = [str(l).strip() for l in new_labels if str(l).strip()]
    if new_labels:
        all_labels[filename] = new_labels
    else:
        all_labels.pop(filename, None)
    _save_labels(all_labels)

    # Build the full labels list (auto + user) matching pipeline logic
    combined = _build_full_labels(filename, new_labels)

    # Patch labels on existing Qdrant points for this file
    qdrant_ok = True
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"labels": combined, "labels_lower": [l.lower() for l in combined]},
            points=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="source", match=MatchValue(value=filename))
                    ]
                )
            ),
        )
        log.info(f"Qdrant labels patched for source={filename}: {combined}")
    except Exception as e:
        qdrant_ok = False
        log.warning(f"Qdrant label patch failed for {filename}: {e}")

    return {"filename": filename, "labels": new_labels, "qdrant_patched": qdrant_ok}


@app.get("/all-labels")
def get_unique_labels():
    """Return a flat deduplicated list of all labels used across all files."""
    all_labels = _load_labels()
    unique = set()
    for labels_list in all_labels.values():
        for label in labels_list:
            unique.add(label)
    return JSONResponse(sorted(unique))


# ── Direct search endpoint ───────────────────────────────────────────
@app.post("/search")
async def do_search(payload: dict):
    """Forward a search query directly to the pipelines service and stream the response."""
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Query is empty"}, status_code=400)

    body = json.dumps(
        {
            "model": PIPELINE_MODEL,
            "messages": [{"role": "user", "content": query}],
        }
    ).encode()

    req = urllib.request.Request(
        f"{PIPELINES_URL}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {PIPELINES_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    def stream():
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    yield line
        except Exception as e:
            yield json.dumps({"error": str(e)}).encode()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(
        content=r"""<!DOCTYPE html>
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

    /* ── Labels ── */
    .labels-section {
      margin-top: 16px;
      display: none;
    }
    .labels-section.visible { display: block; }
    .labels-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 10px;
    }
    .labels-title {
      font-size: 13px; font-weight: 600; color: var(--text-muted);
      text-transform: uppercase; letter-spacing: .06em;
    }
    .btn-add-label {
      background: var(--accent-bg);
      border: 1px solid rgba(124,58,237,.3);
      color: var(--accent-hi);
      border-radius: var(--radius-sm);
      padding: 4px 12px;
      font-size: 12px; font-weight: 600;
      cursor: pointer;
      transition: background .15s, border-color .15s;
    }
    .btn-add-label:hover {
      background: rgba(124,58,237,.22);
      border-color: var(--accent-hi);
    }
    .label-row {
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 8px;
    }
    .label-row input {
      flex: 1;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      font-size: 13px;
      outline: none;
      transition: border-color .15s;
    }
    .label-row input:focus { border-color: var(--accent); }
    .label-row input::placeholder { color: var(--text-muted); }
    .btn-remove-label {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: var(--radius-sm);
      width: 30px; height: 30px;
      font-size: 14px;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
      transition: background .15s, color .15s, border-color .15s;
    }
    .btn-remove-label:hover {
      background: var(--red-bg);
      color: var(--red);
      border-color: rgba(248,81,73,.3);
    }
    .auto-label {
      display: inline-flex; align-items: center; gap: 6px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 20px; padding: 4px 12px;
      font-size: 12px; color: var(--text-muted);
      margin-bottom: 10px;
    }
    .auto-label .al-icon { color: var(--accent-hi); }

    /* ── Label pills (in library) ── */
    .label-pills {
      display: flex; flex-wrap: wrap; gap: 4px;
      margin-top: 4px;
    }
    .label-pill {
      display: inline-block;
      background: var(--accent-bg);
      color: var(--accent-hi);
      border: 1px solid rgba(124,58,237,.2);
      border-radius: 12px; padding: 1px 8px;
      font-size: 11px; font-weight: 500;
    }
    .label-pill.auto {
      background: var(--surface2);
      color: var(--text-muted);
      border-color: var(--border);
    }
    .file-info { flex: 1; min-width: 0; }
    .file-info .file-name {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .btn-edit-labels {
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
    .btn-edit-labels:hover {
      background: var(--accent-bg);
      color: var(--accent-hi);
      border-color: rgba(124,58,237,.3);
    }

    /* ── Modal ── */
    .modal-overlay {
      display: none;
      position: fixed; inset: 0;
      background: rgba(0,0,0,.6);
      z-index: 50;
      align-items: center; justify-content: center;
    }
    .modal-overlay.visible { display: flex; }
    .modal {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 24px;
      width: 90%; max-width: 500px;
      max-height: 80vh; overflow-y: auto;
    }
    .modal-title {
      font-size: 16px; font-weight: 600;
      margin-bottom: 16px;
    }
    .modal-footer {
      display: flex; gap: 10px; justify-content: flex-end;
      margin-top: 16px;
    }
    .modal-btn {
      padding: 8px 18px;
      border-radius: var(--radius-sm);
      font-size: 13px; font-weight: 600;
      cursor: pointer; border: none;
    }
    .modal-btn.primary {
      background: linear-gradient(135deg, var(--accent), #4f46e5);
      color: #fff;
    }
    .modal-btn.secondary {
      background: var(--surface2);
      color: var(--text-muted);
      border: 1px solid var(--border);
    }
    .modal-btn:hover { opacity: .85; }

    /* ── Search card ── */
    .search-row {
      display: flex; gap: 10px; align-items: flex-start;
    }
    .search-input-wrap {
      flex: 1; position: relative;
    }
    #searchInput {
      width: 100%;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: var(--radius-sm);
      padding: 11px 14px;
      font-size: 14px;
      outline: none;
      transition: border-color .15s;
    }
    #searchInput:focus { border-color: var(--accent); }
    #searchInput::placeholder { color: var(--text-muted); }

    .label-chips {
      display: flex; flex-wrap: wrap; gap: 6px;
      margin-bottom: 12px; min-height: 0;
    }
    .label-chip {
      display: inline-flex; align-items: center; gap: 5px;
      background: var(--accent-bg);
      color: var(--accent-hi);
      border: 1px solid rgba(124,58,237,.25);
      border-radius: 16px; padding: 4px 10px 4px 12px;
      font-size: 12px; font-weight: 500;
      animation: chipIn .2s ease;
    }
    @keyframes chipIn {
      from { transform: scale(.85); opacity: 0; }
      to   { transform: scale(1);   opacity: 1; }
    }
    .label-chip .chip-x {
      background: none; border: none;
      color: var(--accent-hi); opacity: .6;
      cursor: pointer; font-size: 14px; line-height: 1;
      padding: 0 2px;
    }
    .label-chip .chip-x:hover { opacity: 1; }

    .autocomplete-dropdown {
      display: none;
      position: absolute; left: 0; right: 0; top: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      border-top: none;
      border-radius: 0 0 var(--radius-sm) var(--radius-sm);
      max-height: 200px; overflow-y: auto;
      z-index: 20;
    }
    .autocomplete-dropdown.open { display: block; }
    .ac-item {
      padding: 8px 14px;
      font-size: 13px;
      cursor: pointer;
      display: flex; justify-content: space-between; align-items: center;
      color: var(--text);
    }
    .ac-item:hover, .ac-item.active {
      background: var(--accent-bg);
      color: var(--accent-hi);
    }
    .ac-item .ac-count {
      font-size: 11px; color: var(--text-muted);
    }
    .ac-hint {
      padding: 6px 14px;
      font-size: 11px; color: var(--text-muted);
      border-bottom: 1px solid var(--border);
    }

    #searchBtn {
      background: linear-gradient(135deg, var(--accent), #4f46e5);
      color: #fff; border: none;
      border-radius: var(--radius-sm);
      padding: 11px 20px;
      font-size: 14px; font-weight: 600;
      cursor: pointer; white-space: nowrap;
      transition: opacity .15s;
      flex-shrink: 0;
    }
    #searchBtn:hover { opacity: .88; }
    #searchBtn:disabled { opacity: .4; cursor: default; }

    .search-mode-toggle {
      display: flex; gap: 0; margin-bottom: 14px;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    .search-mode-toggle button {
      flex: 1; padding: 8px 14px;
      background: var(--surface2);
      color: var(--text-muted);
      border: none; font-size: 13px; font-weight: 500;
      cursor: pointer;
      transition: background .15s, color .15s;
    }
    .search-mode-toggle button.active {
      background: var(--accent-bg);
      color: var(--accent-hi);
    }
    .search-mode-toggle button:hover:not(.active) {
      background: var(--surface);
    }

    .search-copied {
      font-size: 12px; color: var(--green);
      margin-top: 8px; min-height: 18px;
    }
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

      <!-- Labels section (appears after file selected) -->
      <div id="labelsSection" class="labels-section">
        <div class="labels-header">
          <div class="labels-title">Labels</div>
          <button class="btn-add-label" onclick="addLabelRow()">+ Add Label</button>
        </div>
        <div class="auto-label" id="autoLabel" style="display:none">
          <span class="al-icon">🏷</span>
          <span id="autoLabelText">filename.pdf</span>
          <span style="color:var(--text-muted); font-style:italic">(auto)</span>
        </div>
        <div id="labelRows"></div>
      </div>

      <button id="uploadBtn" disabled onclick="uploadFile()">
        <span id="uploadBtnIcon">⬆</span>
        <span id="uploadBtnLabel">Select a file first</span>
      </button>
    </div>

    <!-- Search card -->
    <div class="card">
      <div class="card-title">Search Documents</div>

      <div class="search-mode-toggle">
        <button id="modeLabels" class="active" onclick="setSearchMode('labels')">🏷 Filter by Labels</button>
      </div>

      <div id="labelFilterArea">
        <div class="label-chips" id="selectedLabels"></div>
        <div style="position:relative">
          <input type="text" id="labelInput" placeholder="Type to find labels…"
                 autocomplete="off"
                 style="width:100%; background:var(--surface2); border:1px solid var(--border);
                        color:var(--text); border-radius:var(--radius-sm);
                        padding:8px 12px; font-size:13px; outline:none; margin-bottom:12px;" />
          <div class="autocomplete-dropdown" id="acDropdown"></div>
        </div>
      </div>

      <div class="search-row">
        <div class="search-input-wrap">
          <input type="text" id="searchInput" placeholder="What are you looking for?" onkeydown="if(event.key==='Enter')doSearch()" />
        </div>
        <button id="searchBtn" onclick="doSearch()">Search</button>
      </div>
      <div class="search-copied" id="searchCopied"></div>
      <div style="margin-top:8px; padding:8px 10px; background:var(--surface2); border:1px solid var(--border); border-radius:var(--radius-sm); font-size:12px; line-height:1.5; color:var(--accent-hi);">
        <b>Search</b> opens Open WebUI with your query auto-submitted. Press <b>Enter</b> or click the button.
      </div>
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

<!-- Edit Labels Modal -->
<div class="modal-overlay" id="editLabelsModal">
  <div class="modal">
    <div class="modal-title" id="editLabelsTitle">Edit Labels</div>
    <div class="auto-label">
      <span class="al-icon">🏷</span>
      <span id="modalAutoLabel">filename.pdf</span>
      <span style="color:var(--text-muted); font-style:italic">(auto)</span>
    </div>
    <div id="modalLabelRows"></div>
    <button class="btn-add-label" onclick="addModalLabelRow()" style="margin-top:8px">+ Add Label</button>
    <div class="modal-footer">
      <button class="modal-btn secondary" onclick="closeEditLabels()">Cancel</button>
      <button class="modal-btn primary" onclick="saveEditLabels()">Save Labels</button>
    </div>
  </div>
</div>

<script>
  // ── Search card ─────────────────────────────────────────────────────
  let searchMode = 'labels';
  let selectedSearchLabels = [];
  let allKnownLabels = [];
  let acIndex = -1;

  function setSearchMode(mode) {
    searchMode = mode;
    document.getElementById('labelFilterArea').style.display = mode === 'labels' ? '' : 'none';
    if (mode === 'labels') {
      loadAllKnownLabels();
      document.getElementById('labelInput').focus();
    }
  }

  async function loadAllKnownLabels() {
    try {
      // Fetch labels per file to build label -> doc count
      const r = await fetch('/labels');
      const data = await r.json(); // { filename: [labels], ... }
      const counts = {};
      // Also include auto-labels (filename + stem)
      for (const [fn, labels] of Object.entries(data)) {
        const stem = fn.replace(/\.pdf$/i, '');
        const all = [fn, stem, ...labels];
        const seen = new Set();
        all.forEach(l => {
          const lower = l.toLowerCase();
          if (!seen.has(lower)) {
            seen.add(lower);
            counts[l] = (counts[l] || 0) + 1;
          }
        });
      }
      // Also count files that only have auto-labels (not in labels.json)
      const sr = await fetch('/status');
      const sdata = await sr.json();
      (sdata.indexed_files || []).forEach(fn => {
        const stem = fn.replace(/\.pdf$/i, '');
        if (!counts[fn]) counts[fn] = 1;
        if (!counts[stem]) counts[stem] = 1;
      });
      allKnownLabels = Object.entries(counts)
        .map(([label, count]) => ({ label, count }))
        .sort((a, b) => a.label.toLowerCase().localeCompare(b.label.toLowerCase()));
    } catch (_) {}
  }

  // Autocomplete for label input
  const labelInput = document.getElementById('labelInput');
  const acDropdown = document.getElementById('acDropdown');

  labelInput.addEventListener('input', () => {
    const q = labelInput.value.trim().toLowerCase();
    renderAcDropdown(q);
  });

  labelInput.addEventListener('focus', () => {
    renderAcDropdown(labelInput.value.trim().toLowerCase());
  });

  labelInput.addEventListener('keydown', e => {
    const items = acDropdown.querySelectorAll('.ac-item');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      acIndex = Math.min(acIndex + 1, items.length - 1);
      updateAcActive(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      acIndex = Math.max(acIndex - 1, 0);
      updateAcActive(items);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (acIndex >= 0 && items[acIndex]) {
        selectLabel(items[acIndex].dataset.label);
      } else if (labelInput.value.trim()) {
        // Allow typing a custom label not in the list
        selectLabel(labelInput.value.trim());
      }
    } else if (e.key === 'Escape') {
      acDropdown.classList.remove('open');
    }
  });

  // Close dropdown when clicking outside
  document.addEventListener('click', e => {
    if (!e.target.closest('#labelFilterArea')) {
      acDropdown.classList.remove('open');
    }
  });

  function renderAcDropdown(q) {
    const already = new Set(selectedSearchLabels.map(l => l.toLowerCase()));
    const filtered = allKnownLabels.filter(
      item => !already.has(item.label.toLowerCase()) &&
              (q === '' || item.label.toLowerCase().includes(q))
    );
    acDropdown.innerHTML = '';
    acIndex = -1;
    if (filtered.length === 0 && !q) {
      acDropdown.innerHTML = '<div class="ac-hint">No labels available</div>';
      acDropdown.classList.add('open');
      return;
    }
    if (filtered.length === 0) {
      acDropdown.innerHTML = '<div class="ac-hint">No matches — press Enter to use as custom label</div>';
      acDropdown.classList.add('open');
      return;
    }
    if (q === '') {
      acDropdown.innerHTML = '<div class="ac-hint">All labels — type to filter</div>';
    }
    filtered.slice(0, 15).forEach(item => {
      const div = document.createElement('div');
      div.className = 'ac-item';
      div.dataset.label = item.label;
      // Highlight matching part
      const idx = item.label.toLowerCase().indexOf(q);
      let display = item.label;
      if (q && idx >= 0) {
        display = item.label.slice(0, idx) +
                  '<strong>' + item.label.slice(idx, idx + q.length) + '</strong>' +
                  item.label.slice(idx + q.length);
      }
      div.innerHTML = `<span>${display}</span><span class="ac-count">${item.count} doc${item.count !== 1 ? 's' : ''}</span>`;
      div.addEventListener('mousedown', e => {
        e.preventDefault();
        selectLabel(item.label);
      });
      acDropdown.appendChild(div);
    });
    if (filtered.length > 15) {
      acDropdown.innerHTML += `<div class="ac-hint">${filtered.length - 15} more…</div>`;
    }
    acDropdown.classList.add('open');
  }

  function updateAcActive(items) {
    items.forEach((el, i) => el.classList.toggle('active', i === acIndex));
    if (items[acIndex]) items[acIndex].scrollIntoView({ block: 'nearest' });
  }

  function selectLabel(label) {
    label = label.trim();
    if (!label) return;
    if (selectedSearchLabels.some(l => l.toLowerCase() === label.toLowerCase())) return;
    selectedSearchLabels.push(label);
    renderSelectedLabels();
    labelInput.value = '';
    acDropdown.classList.remove('open');
    labelInput.focus();
  }

  function removeSearchLabel(label) {
    selectedSearchLabels = selectedSearchLabels.filter(l => l !== label);
    renderSelectedLabels();
  }

  function renderSelectedLabels() {
    const container = document.getElementById('selectedLabels');
    container.innerHTML = '';
    selectedSearchLabels.forEach(label => {
      const chip = document.createElement('span');
      chip.className = 'label-chip';
      chip.innerHTML = `🏷 ${label.replace(/</g,'&lt;')} <button class="chip-x" title="Remove">&times;</button>`;
      chip.querySelector('.chip-x').addEventListener('click', () => removeSearchLabel(label));
      container.appendChild(chip);
    });
  }

  function doSearch() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) { document.getElementById('searchInput').focus(); return; }

    // Build the full query with label prefixes
    let fullQuery = '';
    if (searchMode === 'labels' && selectedSearchLabels.length) {
      const prefixes = selectedSearchLabels.map(l => {
        return l.includes(' ') ? `label:"${l}"` : `label:${l}`;
      });
      fullQuery = prefixes.join(' ') + ' ' + query;
    } else {
      fullQuery = query;
    }

    // Copy to clipboard as fallback and show feedback
    navigator.clipboard.writeText(fullQuery).then(() => {
      document.getElementById('searchCopied').textContent =
        '✓ Search sent to Open WebUI — query also copied to clipboard';
      setTimeout(() => {
        document.getElementById('searchCopied').textContent = '';
      }, 4000);
    }).catch(() => {});

    // Open Open WebUI with query pre-filled and auto-submitted
    const owUrl = `${window.location.protocol}//${window.location.hostname}:3000/?q=${encodeURIComponent(fullQuery)}`;
    window.open(owUrl, '_blank');

    toast(`Query ready: ${fullQuery}`, 'ok');
  }

  // ── Label rows (upload form) ──────────────────────────────────────
  function addLabelRow(value = '') {
    const container = document.getElementById('labelRows');
    const row = document.createElement('div');
    row.className = 'label-row';
    row.innerHTML = `
      <input type="text" placeholder="e.g. project name, category, topic…" value="${value.replace(/"/g, '&quot;')}" />
      <button class="btn-remove-label" onclick="this.parentElement.remove()" title="Remove label">✕</button>
    `;
    container.appendChild(row);
    row.querySelector('input').focus();
  }

  function getUploadLabels() {
    const inputs = document.querySelectorAll('#labelRows .label-row input');
    const labels = [];
    inputs.forEach(inp => {
      const v = inp.value.trim();
      if (v) labels.push(v);
    });
    return labels;
  }

  // ── Modal label rows (edit existing labels) ───────────────────────
  let editingFilename = null;

  function addModalLabelRow(value = '') {
    const container = document.getElementById('modalLabelRows');
    const row = document.createElement('div');
    row.className = 'label-row';
    row.innerHTML = `
      <input type="text" placeholder="e.g. project name, category, topic…" value="${value.replace(/"/g, '&quot;')}" />
      <button class="btn-remove-label" onclick="this.parentElement.remove()" title="Remove label">✕</button>
    `;
    container.appendChild(row);
    row.querySelector('input').focus();
  }

  async function openEditLabels(filename) {
    editingFilename = filename;
    document.getElementById('editLabelsTitle').textContent = `Edit Labels — ${filename}`;
    document.getElementById('modalAutoLabel').textContent = filename;
    document.getElementById('modalLabelRows').innerHTML = '';
    // Load existing labels
    try {
      const r = await fetch('/labels/' + encodeURIComponent(filename));
      const labels = await r.json();
      if (labels.length) {
        labels.forEach(l => addModalLabelRow(l));
      } else {
        addModalLabelRow();
      }
    } catch (_) {
      addModalLabelRow();
    }
    document.getElementById('editLabelsModal').classList.add('visible');
  }

  function closeEditLabels() {
    document.getElementById('editLabelsModal').classList.remove('visible');
    editingFilename = null;
  }

  async function saveEditLabels() {
    if (!editingFilename) return;
    const inputs = document.querySelectorAll('#modalLabelRows .label-row input');
    const labels = [];
    inputs.forEach(inp => {
      const v = inp.value.trim();
      if (v) labels.push(v);
    });
    try {
      await fetch('/labels/' + encodeURIComponent(editingFilename), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ labels }),
      });
      toast(`Labels updated for ${editingFilename}`, 'ok');
    } catch (e) {
      toast(`Failed to save labels: ${e}`, 'err');
    }
    closeEditLabels();
    refresh();
  }

  // Close modal on overlay click
  document.getElementById('editLabelsModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeEditLabels();
  });

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
    // Show labels section + auto-label
    document.getElementById('labelsSection').classList.add('visible');
    document.getElementById('autoLabel').style.display = '';
    document.getElementById('autoLabelText').textContent = f.name;
    // Add one empty label row if none exist
    if (document.getElementById('labelRows').children.length === 0) {
      addLabelRow();
    }
  }

  // ── Upload ─────────────────────────────────────────────────────────
  async function uploadFile() {
    const f = fi.files[0];
    if (!f) return;
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true;
    document.getElementById('uploadBtnLabel').textContent = 'Uploading…';
    document.getElementById('uploadBtnIcon').textContent = '⏳';

    const labels = getUploadLabels();
    const form = new FormData();
    form.append('file', f);
    if (labels.length) {
      form.append('labels', JSON.stringify(labels));
    }
    try {
      const r = await fetch('/upload', { method: 'POST', body: form });
      const d = await r.json();
      if (r.ok) {
        const lblCount = labels.length;
        const lblMsg = lblCount ? ` with ${lblCount} label${lblCount > 1 ? 's' : ''}` : '';
        toast(`✓ ${d.filename} uploaded${lblMsg} — indexing started`, 'ok');
        fi.value = '';
        document.getElementById('selectedFile').textContent = '';
        document.getElementById('uploadBtnLabel').textContent = 'Select a file first';
        document.getElementById('uploadBtnIcon').textContent = '⬆';
        // Reset labels section
        document.getElementById('labelsSection').classList.remove('visible');
        document.getElementById('labelRows').innerHTML = '';
        document.getElementById('autoLabel').style.display = 'none';
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

  // Tracks a pending pause/cancel while waiting for the current page embed to finish
  let pendingAction = null; // { type: 'pausing'|'cancelling', file: string }

  // ── Pause indexing (saves progress, resumes next run) ──────────────
  async function pauseIndexing() {
    const pb = document.getElementById('pauseBtn');
    const cb = document.getElementById('cancelBtn');
    const currentFile = document.getElementById('progFile').textContent;
    pendingAction = { type: 'pausing', file: currentFile };
    pb.disabled = true; pb.textContent = '⏸ Finishing page…';
    cb.disabled = true;
    document.getElementById('progPages').textContent = '⏸ Pausing — finishing current page…';
    await fetch('/pause', { method: 'POST' });
  }

  // ── Cancel indexing (clears progress, no resume) ───────────────────
  async function cancelIndexing() {
    const pb = document.getElementById('pauseBtn');
    const cb = document.getElementById('cancelBtn');
    const currentFile = document.getElementById('progFile').textContent;
    pendingAction = { type: 'cancelling', file: currentFile };
    cb.disabled = true; cb.textContent = '✕ Finishing page…';
    pb.disabled = true;
    document.getElementById('progPages').textContent = '✕ Cancelling — finishing current page…';
    await fetch('/cancel', { method: 'POST' });
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
  let cachedLabels = {};

  async function refreshLabels() {
    try {
      const r = await fetch('/labels');
      cachedLabels = await r.json();
    } catch (_) {}
  }

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
        const pb = document.getElementById('pauseBtn');
        const cb = document.getElementById('cancelBtn');
        if (pendingAction && job.current_file === pendingAction.file) {
          // Still on the same file — keep buttons greyed, show feedback in progPages
          pb.disabled = true;
          cb.disabled = true;
          const pageLabel = job.current_page > 0 ? `page ${job.current_page} of ${job.total_pages}` : 'starting…';
          if (pendingAction.type === 'pausing') {
            pb.textContent = '⏸ Finishing page…';
            cb.textContent = '✕ Cancel';
            document.getElementById('progPages').textContent = `⏸ Pausing — finishing ${pageLabel}`;
          } else {
            cb.textContent = '✕ Finishing page…';
            pb.textContent = '⏸ Pause';
            document.getElementById('progPages').textContent = `✕ Cancelling — finishing ${pageLabel}`;
          }
        } else {
          // No pending action (or file changed) — restore buttons and show page count
          if (pendingAction) {
            pendingAction = null;
            toast('Done — moving to next file in queue', 'ok');
          }
          pb.disabled = false; pb.textContent = '⏸ Pause';
          cb.disabled = false; cb.textContent = '✕ Cancel';
          document.getElementById('progPages').textContent =
            job.current_page === 0 ? 'Starting…' : `Page ${job.current_page} of ${job.total_pages}`;
        }
      } else {
        if (pendingAction) pendingAction = null;
        pw.classList.remove('visible');
        iw.style.display = '';
      }

      // File list with labels
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
          // Build label pills
          const fileLabels = cachedLabels[fn] || [];
          let pillsHtml = `<span class="label-pill auto">📄 ${fn}</span>`;
          fileLabels.forEach(l => {
            pillsHtml += `<span class="label-pill">${l.replace(/</g,'&lt;')}</span>`;
          });
          li.innerHTML = `
            <div class="file-icon">📄</div>
            <div class="file-info">
              <div class="file-name" title="${fn}">${fn}</div>
              <div class="label-pills">${pillsHtml}</div>
            </div>
            <span class="file-badge">indexed</span>
            <button class="btn-edit-labels" onclick="openEditLabels('${esc}')" title="Edit labels">🏷 Labels</button>
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

  function refresh() { refreshLabels().then(() => refreshStatus()); refreshQueue(); }
  refresh();
  setInterval(refresh, 3000);
  loadAllKnownLabels();
</script>
</body>
</html>
""",
        headers={"Cache-Control": "no-store"},
    )
