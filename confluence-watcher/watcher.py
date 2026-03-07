"""
Confluence Watchdog — Incremental Qdrant Indexing
Polls Confluence REST API, detects new/changed/deleted pages,
renders them to PDF via Playwright, and triggers the Vision RAG
pipeline to re-index only the changed content.
"""

import json
import logging
import os
import pathlib
import re
import socket
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "http://localhost:8090")
CONFLUENCE_PAT = os.getenv("CONFLUENCE_PAT", "")
CONFLUENCE_SPACES = [s.strip() for s in os.getenv("CONFLUENCE_SPACES", "RAG").split(",") if s.strip()]
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_docs")
PIPELINES_URL = os.getenv("PIPELINES_URL", "http://pipelines:9099")
PIPELINES_API_KEY = os.getenv("PIPELINES_API_KEY", "0p3n-w3bu!")
PIPELINE_MODEL = os.getenv("PIPELINE_MODEL", "colpali-pipeline")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "300"))

# Derive a localhost-based render URL so Playwright navigates to localhost:PORT,
# matching Confluence's configured base URL and avoiding the "URL doesn't match" warning.
# Chromium's --host-resolver-rules is used to route localhost → actual Docker host IP.
_parsed_url = urlparse(CONFLUENCE_URL)
CONFLUENCE_PORT = _parsed_url.port or 8090
CONFLUENCE_RENDER_URL = f"http://localhost:{CONFLUENCE_PORT}"

PDF_DIR = pathlib.Path(os.getenv("PDF_DIR", "/app/downloads"))
STATE_DIR = pathlib.Path(os.getenv("STATE_DIR", "/app/state"))
PIPELINE_STATE_FILE = pathlib.Path(os.getenv("PIPELINE_STATE_FILE", "/app/pipelines/pipeline_state.json"))
WATCHER_STATE_FILE = STATE_DIR / "watcher_state.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# ── Auth headers ─────────────────────────────────────────────────────────────
def _auth_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if CONFLUENCE_PAT:
        headers["Authorization"] = f"Bearer {CONFLUENCE_PAT}"
    return headers


# ── Slug / filename helpers ───────────────────────────────────────────────────
def slugify(text: str, max_len: int = 35) -> str:
    """Convert a page title to a lowercase, filename-safe slug."""
    s = re.sub(r"[—–\-]", " ", text)       # dashes → space
    s = re.sub(r"[^\w\s]", "", s)           # drop punctuation
    s = re.sub(r"\s+", "_", s.strip())      # spaces → underscore
    s = s.lower()
    s = re.sub(r"_+", "_", s).strip("_")   # collapse underscores
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


_homepage_cache: dict[str, str] = {}

def get_space_homepage_id(space_key: str) -> str:
    """Fetch and cache the space homepage ID (used to exclude it from paths)."""
    if space_key in _homepage_cache:
        return _homepage_cache[space_key]
    url = f"{CONFLUENCE_URL}/rest/api/space/{space_key}?expand=homepage"
    resp = requests.get(url, headers=_auth_headers(), timeout=15)
    resp.raise_for_status()
    homepage_id = str(resp.json()["homepage"]["id"])
    _homepage_cache[space_key] = homepage_id
    log.info(f"Space {space_key} homepage id={homepage_id}")
    return homepage_id


def build_pdf_filename(space_key: str, page_id: str, title: str,
                       ancestors: list, homepage_id: str) -> str:
    """
    Build a hierarchical, human-readable PDF path.

    Ancestors (excluding the space homepage) become subdirectories.
    The leaf filename is the page title slug + page_id.

    Examples:
      top-level SpaceX page  → confluence_RAG_spacex_company_overview_131179.pdf
      child xAI Merger        → spacex_company_overview/confluence_RAG_xai_merger_131181.pdf
      grandchild Grok Model   → spacex_company_overview/xai_merger/confluence_RAG_grok_model_131183.pdf
    """
    relevant = [a for a in ancestors if str(a["id"]) != homepage_id]
    leaf = f"confluence_{space_key}_{slugify(title)}_{page_id}.pdf"
    if not relevant:
        return leaf
    subdir = "/".join(slugify(a["title"]) for a in relevant)
    return f"{subdir}/{leaf}"


# ── Watcher state helpers ─────────────────────────────────────────────────────
def _load_watcher_state() -> dict:
    if WATCHER_STATE_FILE.exists():
        try:
            with open(WATCHER_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            log.warning("Corrupt watcher_state.json — starting fresh")
    return {}


def _save_watcher_state(state: dict) -> None:
    tmp = str(WATCHER_STATE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, str(WATCHER_STATE_FILE))


def update_watcher_state(state: dict, space_key: str, page_id: str,
                         version: int, title: str, pdf_filename: str) -> None:
    if space_key not in state:
        state[space_key] = {}
    state[space_key][page_id] = {
        "version": version,
        "title": title,
        "pdf_filename": pdf_filename,
        "last_indexed": datetime.now(timezone.utc).isoformat(),
    }
    _save_watcher_state(state)
    log.info(f"State updated: {space_key}/{page_id} v{version} → {pdf_filename}")


# ── Confluence API ────────────────────────────────────────────────────────────
def poll_confluence(space_key: str) -> list[dict]:
    """Return all pages in a space with version numbers and ancestor chain."""
    pages = []
    start = 0
    limit = 100
    while True:
        url = (
            f"{CONFLUENCE_URL}/rest/api/content"
            f"?spaceKey={space_key}&type=page"
            f"&expand=version,ancestors&limit={limit}&start={start}"
        )
        try:
            resp = requests.get(url, headers=_auth_headers(), timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Confluence API error for space {space_key}: {e}")
            break

        data = resp.json()
        results = data.get("results", [])
        for page in results:
            pages.append({
                "id": str(page["id"]),
                "title": page["title"],
                "version": page["version"]["number"],
                "space_key": space_key,
                "ancestors": page.get("ancestors", []),
            })

        size = data.get("size", 0)
        start += size
        if start >= data.get("totalSize", 0) or size == 0:
            break

    log.info(f"Polled {len(pages)} pages from space {space_key}")
    return pages


def has_changed(watcher_state: dict, space_key: str, page_id: str, version: int) -> bool:
    stored = watcher_state.get(space_key, {}).get(page_id)
    if stored is None:
        return True
    return version > stored["version"]


# ── PDF rendering ─────────────────────────────────────────────────────────────
def _resolve_docker_host_ip() -> str:
    """Resolve the Docker host IP so Chromium can reach it via localhost."""
    for hostname in ("host.docker.internal", "gateway.docker.internal"):
        try:
            ip = socket.gethostbyname(hostname)
            log.info(f"Docker host IP resolved via {hostname}: {ip}")
            return ip
        except socket.gaierror:
            pass
    log.warning("Could not resolve Docker host IP — falling back to 172.17.0.1")
    return "172.17.0.1"


def render_page_to_pdf(page_id: str, pdf_path: pathlib.Path) -> None:
    """Use Playwright headless Chromium to render a Confluence page as PDF.

    Navigates via localhost:{port} (matching Confluence's configured base URL)
    while routing localhost → Docker host IP via Chromium's --host-resolver-rules.
    This avoids Confluence's 'URL doesn't match' base URL warning.
    """
    page_url = f"{CONFLUENCE_RENDER_URL}/pages/viewpage.action?pageId={page_id}"
    log.info(f"Rendering {page_url} → {pdf_path}")

    host_ip = _resolve_docker_host_ip()

    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--host-resolver-rules=MAP localhost {host_ip}",
        ])
        ctx = browser.new_context(
            extra_http_headers={"Authorization": f"Bearer {CONFLUENCE_PAT}"} if CONFLUENCE_PAT else {}
        )
        pg = ctx.new_page()
        try:
            pg.goto(page_url, wait_until="networkidle", timeout=60_000)
            try:
                pg.wait_for_selector("#main-content", timeout=10_000)
            except Exception:
                pass
            pg.pdf(path=str(pdf_path), format="A4", print_background=True)
        finally:
            browser.close()

    log.info(f"PDF saved: {pdf_path} ({pdf_path.stat().st_size} bytes)")


# ── Qdrant deletion ───────────────────────────────────────────────────────────
def delete_from_qdrant(pdf_filename: str) -> None:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=pdf_filename))]
                )
            ),
        )
        log.info(f"Deleted Qdrant vectors for source={pdf_filename}")
    except Exception as e:
        log.warning(f"Qdrant delete failed for {pdf_filename}: {e}")


# ── Pipeline state patch ──────────────────────────────────────────────────────
def patch_pipeline_state(filename: str) -> None:
    if not PIPELINE_STATE_FILE.exists():
        return
    try:
        with open(PIPELINE_STATE_FILE) as f:
            state = json.load(f)
        if filename in state.get("indexed_files", []):
            state["indexed_files"] = [f for f in state["indexed_files"] if f != filename]
            tmp = str(PIPELINE_STATE_FILE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, str(PIPELINE_STATE_FILE))
            log.info(f"Removed {filename} from pipeline_state indexed_files")
    except Exception as e:
        log.warning(f"Failed to patch pipeline_state.json: {e}")


# ── Trigger reindex ───────────────────────────────────────────────────────────
def trigger_reindex() -> None:
    try:
        resp = requests.post(
            f"{PIPELINES_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {PIPELINES_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": PIPELINE_MODEL,
                "messages": [{"role": "user", "content": "__index_now__"}],
            },
            timeout=10,
        )
        log.info(f"Reindex triggered — pipeline responded {resp.status_code}")
    except Exception as e:
        log.warning(f"Failed to trigger reindex: {e}")


# ── Deletion detection ────────────────────────────────────────────────────────
def handle_deleted_pages(watcher_state: dict, space_key: str, live_page_ids: set[str]) -> None:
    stored_ids = set(watcher_state.get(space_key, {}).keys())
    deleted_ids = stored_ids - live_page_ids

    for page_id in deleted_ids:
        entry = watcher_state[space_key][page_id]
        pdf_filename = entry["pdf_filename"]
        log.info(f"Page {page_id} ({entry['title']!r}) deleted from Confluence — cleaning up")

        pdf_path = PDF_DIR / pdf_filename
        if pdf_path.exists():
            pdf_path.unlink()
            log.info(f"Deleted PDF: {pdf_path}")
            # Remove parent dir if now empty
            try:
                pdf_path.parent.rmdir()
            except OSError:
                pass

        delete_from_qdrant(pdf_filename)
        patch_pipeline_state(pdf_filename)
        del watcher_state[space_key][page_id]

    if deleted_ids:
        _save_watcher_state(watcher_state)


# ── Main poll loop ────────────────────────────────────────────────────────────
def process_space(space_key: str, watcher_state: dict) -> None:
    homepage_id = get_space_homepage_id(space_key)
    pages = poll_confluence(space_key)
    live_page_ids = {p["id"] for p in pages}

    handle_deleted_pages(watcher_state, space_key, live_page_ids)

    needs_reindex = False
    for page in pages:
        page_id = page["id"]
        version = page["version"]
        title = page["title"]
        ancestors = page["ancestors"]

        if not has_changed(watcher_state, space_key, page_id, version):
            continue

        action = "new" if page_id not in watcher_state.get(space_key, {}) else "updated"
        new_filename = build_pdf_filename(space_key, page_id, title, ancestors, homepage_id)
        pdf_path = PDF_DIR / new_filename

        # Build a readable path label for logging  (e.g. SpaceX / Falcon 9)
        path_label = " / ".join(
            [a["title"] for a in ancestors if str(a["id"]) != homepage_id] + [title]
        )
        log.info(f"[{action}] {path_label!r} (v{version}) → {new_filename}")

        if action == "updated":
            old_entry = watcher_state[space_key][page_id]
            old_filename = old_entry["pdf_filename"]
            delete_from_qdrant(old_filename)
            patch_pipeline_state(old_filename)
            # If the filename changed (title/move), delete the old PDF from disk
            if old_filename != new_filename:
                old_path = PDF_DIR / old_filename
                if old_path.exists():
                    old_path.unlink()
                log.info(f"Filename changed: {old_filename} → {new_filename}")

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            render_page_to_pdf(page_id, pdf_path)
        except Exception as e:
            log.error(f"Failed to render page {page_id}: {e}")
            continue

        update_watcher_state(watcher_state, space_key, page_id, version, title, new_filename)
        needs_reindex = True

    if needs_reindex:
        trigger_reindex()


def main() -> None:
    log.info(f"Confluence watcher starting — spaces={CONFLUENCE_SPACES}, interval={POLL_INTERVAL}s")
    log.info(f"Confluence: {CONFLUENCE_URL}, Qdrant: {QDRANT_HOST}:{QDRANT_PORT}/{COLLECTION_NAME}")
    log.info(f"PDF dir: {PDF_DIR}, Pipeline: {PIPELINES_URL}")

    while True:
        watcher_state = _load_watcher_state()
        for space_key in CONFLUENCE_SPACES:
            try:
                process_space(space_key, watcher_state)
            except Exception as e:
                log.error(f"Error processing space {space_key}: {e}", exc_info=True)

        log.info(f"Poll complete — sleeping {POLL_INTERVAL}s")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
