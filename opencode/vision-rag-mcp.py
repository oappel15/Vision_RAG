#!/usr/bin/env python3
"""
Vision RAG MCP Server — exposes the Vision RAG pipeline to OpenCode.

Tools:
  - vision_rag_status:      Check indexing status and list indexed files with labels
  - vision_rag_labels:      List all available labels for filtering
  - vision_rag_is_indexed:  Check if a specific PDF file is already indexed
  - vision_rag_upload:      Upload a PDF file to the index with labels
  - vision_rag_search:      Search indexed documents with optional label filters
  - vision_rag_set_labels:  Add/update labels on an already-indexed document

Communicates with:
  - pdf-ingest sidecar (port 8082): file upload, labels, status
  - pipelines service (port 9099):  search queries via chat completions API
"""

import json
import os
import sys
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger("vision-rag-mcp")

# ── Configuration ────────────────────────────────────────────────────
INGEST_URL = os.getenv("VISION_RAG_INGEST_URL", "http://localhost:8082")
PIPELINES_URL = os.getenv("VISION_RAG_PIPELINES_URL", "http://localhost:9099")
PIPELINES_API_KEY = os.getenv("VISION_RAG_PIPELINES_API_KEY", "0p3n-w3bu!")
PIPELINE_MODEL = os.getenv("VISION_RAG_PIPELINE_MODEL", "colpali-pipeline")

mcp = FastMCP("vision-rag", instructions=(
    "Vision RAG document search system. Use these tools to check if PDFs "
    "are indexed, upload new ones with labels, and search document contents "
    "including images, tables, diagrams, and schematics."
))


def _ingest_get(path: str) -> dict | list | None:
    """GET request to pdf-ingest sidecar."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{INGEST_URL}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.error(f"Ingest GET {path} failed: {e}")
        return None


def _ingest_put(path: str, body: dict) -> dict | None:
    """PUT request to pdf-ingest sidecar."""
    import urllib.request
    try:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{INGEST_URL}{path}", data=data, method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.error(f"Ingest PUT {path} failed: {e}")
        return None


def _pipeline_query(query: str) -> str:
    """Send a query to the pipeline and collect the full streamed response."""
    import urllib.request
    body = json.dumps({
        "model": PIPELINE_MODEL,
        "messages": [{"role": "user", "content": query}],
    }).encode()
    req = urllib.request.Request(
        f"{PIPELINES_URL}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {PIPELINES_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = []
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    parsed = json.loads(data)
                    choices = parsed.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            result.append(content)
                except json.JSONDecodeError:
                    pass
            return "".join(result)
    except Exception as e:
        return f"Error querying pipeline: {e}"


# ── Tools ────────────────────────────────────────────────────────────

@mcp.tool()
def vision_rag_status() -> str:
    """Check the Vision RAG indexing status. Returns which files are indexed,
    current indexing progress, and labels for each document."""
    status = _ingest_get("/status")
    labels = _ingest_get("/labels")
    if status is None:
        return "Error: Could not connect to Vision RAG service (port 8082). Is it running?"

    indexed = status.get("indexed_files", [])
    job = status.get("index_job", {})

    lines = []
    if job.get("active"):
        f = job.get("current_file", "?")
        p = job.get("current_page", 0)
        t = job.get("total_pages", 0)
        lines.append(f"Indexing in progress: {f} (page {p}/{t})")
    else:
        lines.append("Indexing idle.")

    lines.append(f"\nIndexed files ({len(indexed)}):")
    labels = labels or {}
    for fn in indexed:
        file_labels = labels.get(fn, [])
        lbl_str = f"  labels: [{', '.join(file_labels)}]" if file_labels else ""
        lines.append(f"  - {fn}{lbl_str}")

    if not indexed:
        lines.append("  (none)")

    return "\n".join(lines)


@mcp.tool()
def vision_rag_labels() -> str:
    """List all available labels in the Vision RAG system. Labels can be used
    to filter searches to specific documents or categories."""
    labels = _ingest_get("/labels")
    if labels is None:
        return "Error: Could not connect to Vision RAG service."

    status = _ingest_get("/status")
    indexed = (status or {}).get("indexed_files", [])

    # Build label -> files map (including auto-labels)
    label_map: dict[str, list[str]] = {}
    for fn in indexed:
        stem = Path(fn).stem
        user_labels = labels.get(fn, [])
        all_labels = [fn, stem] + user_labels
        seen = set()
        for lbl in all_labels:
            lower = lbl.lower()
            if lower not in seen:
                seen.add(lower)
                label_map.setdefault(lbl, []).append(fn)

    lines = ["Available labels:"]
    for lbl in sorted(label_map.keys(), key=str.lower):
        files = label_map[lbl]
        lines.append(f"  - {lbl} ({len(files)} doc{'s' if len(files)!=1 else ''})")

    return "\n".join(lines)


@mcp.tool()
def vision_rag_is_indexed(filename: str) -> str:
    """Check if a specific PDF file is indexed in Vision RAG.

    Args:
        filename: The PDF filename to check (e.g. 'Toyota.pdf'). Can also be
                  just a stem like 'Toyota' — will match case-insensitively.
    """
    status = _ingest_get("/status")
    if status is None:
        return "Error: Could not connect to Vision RAG service."

    indexed = status.get("indexed_files", [])
    labels_data = _ingest_get("/labels") or {}

    # Normalize for matching
    target = filename.lower()
    if not target.endswith(".pdf"):
        target_pdf = target + ".pdf"
    else:
        target_pdf = target

    for fn in indexed:
        if fn.lower() == target_pdf or fn.lower() == target:
            file_labels = labels_data.get(fn, [])
            lbl_str = f" with labels: [{', '.join(file_labels)}]" if file_labels else ""
            return f"YES — '{fn}' is indexed{lbl_str}"

    return f"NO — '{filename}' is not indexed. Use vision_rag_upload to add it."


@mcp.tool()
def vision_rag_upload(filepath: str, labels: list[str] | None = None) -> str:
    """Upload a PDF file to Vision RAG for indexing.

    Args:
        filepath: Absolute path to the PDF file on disk.
        labels: Optional list of labels/tags for this document.
                The filename is always added as an automatic label.
                Examples: ['project-x', 'schematics', 'Toyota']
    """
    import urllib.request
    import uuid

    path = Path(filepath)
    if not path.exists():
        return f"Error: File not found: {filepath}"
    if not path.suffix.lower() == ".pdf":
        return f"Error: Only PDF files are accepted, got: {path.suffix}"

    # Build multipart form data manually (no requests library)
    boundary = uuid.uuid4().hex
    body_parts = []

    # File part
    body_parts.append(f"--{boundary}".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"'.encode()
    )
    body_parts.append(b"Content-Type: application/pdf")
    body_parts.append(b"")
    body_parts.append(path.read_bytes())

    # Labels part
    if labels:
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="labels"')
        body_parts.append(b"")
        body_parts.append(json.dumps(labels).encode())

    body_parts.append(f"--{boundary}--".encode())
    body_data = b"\r\n".join(body_parts)

    try:
        req = urllib.request.Request(
            f"{INGEST_URL}/upload",
            data=body_data,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            lbl_str = f" with labels {labels}" if labels else ""
            return (
                f"Uploaded: {result.get('filename', path.name)}{lbl_str}\n"
                f"Size: {result.get('size_bytes', 0)} bytes\n"
                f"Indexing has been triggered automatically. "
                f"Use vision_rag_status to check progress."
            )
    except Exception as e:
        return f"Error uploading: {e}"


@mcp.tool()
def vision_rag_search(query: str, labels: list[str] | None = None) -> str:
    """Search indexed documents using Vision RAG. The query is matched against
    document page images using ColQwen2 multi-vector search, then a Vision LLM
    reads the matched pages and generates an answer with citations.

    This can extract information from images, diagrams, tables, schematics,
    and any visual content in the indexed PDFs.

    Args:
        query: The search question (e.g. 'list all sensors on the wiring diagram')
        labels: Optional list of labels to filter by. Only documents matching
                ALL specified labels will be searched. Case-insensitive.
                Examples: ['Toyota'], ['Confluence', 'project-x']
    """
    # Build query with label prefixes
    parts = []
    if labels:
        for lbl in labels:
            if " " in lbl or "/" in lbl:
                parts.append(f'label:"{lbl}"')
            else:
                parts.append(f"label:{lbl}")
    parts.append(query)
    full_query = " ".join(parts)

    log.info(f"Searching: {full_query}")
    result = _pipeline_query(full_query)

    if not result.strip():
        return "No response from Vision RAG pipeline. The service may be busy indexing."

    return result


@mcp.tool()
def vision_rag_set_labels(filename: str, labels: list[str]) -> str:
    """Add or update labels on an already-indexed document. Labels are also
    patched directly into the vector database for immediate search filtering.

    Args:
        filename: The indexed PDF filename (e.g. 'Toyota.pdf')
        labels: List of labels to set. These replace any existing user labels.
                The filename itself is always an automatic label.
    """
    result = _ingest_put(
        f"/labels/{filename}",
        {"labels": labels},
    )
    if result is None:
        return f"Error: Could not update labels for {filename}"

    patched = result.get("qdrant_patched", False)
    return (
        f"Labels updated for {filename}: {labels}\n"
        f"Qdrant patched: {'yes' if patched else 'no (may need re-index)'}"
    )


if __name__ == "__main__":
    mcp.run()
