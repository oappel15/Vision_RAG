"""
title: ColQwen2 Visual RAG (CPU) – Multi-Vector MaxSim
author: adapted
version: 10.1
license: MIT
description: Visual RAG pipeline using ColQwen2 multi-vector MaxSim + Qdrant + OpenRouter VLM.
             Background indexing thread — queries are never blocked by indexing.
"""

import asyncio
import os, json, base64, io, logging, pathlib, hashlib, re, threading, time, gc, uuid
from typing import List, Literal, Optional

import torch
from PIL import Image
from pydantic import BaseModel
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SearchParams,
    MultiVectorConfig,
    MultiVectorComparator,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from colpali_engine.models import ColQwen2, ColQwen2Processor

log = logging.getLogger("colpali-pipeline")
log.setLevel(logging.DEBUG)

STATE_FILE = "/app/pipelines/pipeline_state.json"
SCHEMA_VERSION = (
    3  # v1 = mean-pooled (broken), v2 = multi-vector MaxSim, v3 = UUID point IDs
)


class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str = "qdrant"
        QDRANT_PORT: int = 6333
        PDF_DIR: str = "/app/downloads"
        COLLECTION_NAME: str = "target_knowledge"
        TOP_K: int = 8
        MAX_PAGES_PER_DOC: int = (
            2  # max pages per document in TOP_K results (0 = no limit)
        )
        SCORE_THRESHOLD: float = 0.0
        # ── VLM backend ── set VLM_PROVIDER to "ollama" to use local Ollama instead
        VLM_PROVIDER: Literal["openrouter", "ollama"] = "openrouter"
        OPENROUTER_MODEL: str = "qwen/qwen3-vl-30b-a3b-instruct"
        THUMBNAIL_SCORE_THRESHOLD: float = (
            0.0  # min score for cited page to show thumbnail; 0 = show all
        )
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        OLLAMA_VLM_MODEL: str = "qwen3-vl:30b"  # model must be pulled in Ollama first
        SHOW_SOURCE_PAGES: bool = True
        SERVER_HOST: str = os.getenv("SERVER_HOST", "localhost")
        IMAGE_CACHE_DIR: str = "/app/pipelines/cache/images"

    def __init__(self):
        self.name = "ColQwen2 Visual RAG"
        self.valves = self.Valves()
        self.model = None
        self.processor = None
        self.qdrant = None
        self._initialized = False
        self._index_lock = threading.Lock()
        self._index_thread: threading.Thread = None
        self._cancel_flag = threading.Event()
        self._cancel_hard = (
            threading.Event()
        )  # hard cancel: clears progress + Qdrant vectors
        self._current_indexing_file: str = None
        self._pending_trigger = False  # queued __index_now__ while thread was busy
        self._query_active = (
            threading.Event()
        )  # set while a query is running; indexing pauses between pages
        self._skipped_file: str = (
            None  # file that was paused/cancelled — defer to end of queue
        )

    async def on_startup(self):
        log.info("on_startup: loading model eagerly in executor …")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self._start_background_index()

    async def on_shutdown(self):
        log.info("Pipeline shutdown")

    # ── helpers ──────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
        return {}

    def _save_state(self, state: dict):
        """Atomic write — prevents JSON corruption if process dies mid-write."""
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)

    def _save_page_image(
        self, page_img: Image.Image, filename: str, page_num: int
    ) -> str:
        """Save page image to cache dir, return filename."""
        os.makedirs(self.valves.IMAGE_CACHE_DIR, exist_ok=True)
        safe_name = pathlib.Path(filename).stem
        img_filename = f"{safe_name}_p{page_num}.jpg"
        img_path = os.path.join(self.valves.IMAGE_CACHE_DIR, img_filename)
        page_img.save(img_path, format="JPEG", quality=85)
        # Invalidate stale thumbnail so it gets regenerated on next query
        thumb_path = img_path.replace(".jpg", "_thumb.jpg")
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
        return img_filename

    def _load_page_image_b64(self, img_filename: str) -> str:
        """Load cached image as base64 for VLM."""
        img_path = os.path.join(self.valves.IMAGE_CACHE_DIR, img_filename)
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _make_thumbnail_file(self, img_filename: str, max_width: int = 150) -> str:
        """Create/cache thumbnail, return thumbnail filename."""
        thumb_filename = img_filename.replace(".jpg", "_thumb.jpg")
        thumb_path = os.path.join(self.valves.IMAGE_CACHE_DIR, thumb_filename)
        src_path = os.path.join(self.valves.IMAGE_CACHE_DIR, img_filename)
        stale = not os.path.exists(thumb_path) or os.path.getmtime(
            src_path
        ) > os.path.getmtime(thumb_path)
        if stale:
            img = Image.open(src_path)
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            img.save(thumb_path, format="JPEG", quality=70)
        return thumb_filename

    # ── model loading ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_snapshot(hf_hub_dir: pathlib.Path, repo_id: str):
        """Return the local snapshot directory for *repo_id* if it exists.

        The HuggingFace cache layout is:
            <hf_hub_dir>/models--<org>--<name>/snapshots/<commit_hash>/

        We read ``refs/main`` to find the current commit hash, then return
        the snapshot path.  Returns ``None`` when the model hasn't been
        downloaded yet.
        """
        repo_dir = hf_hub_dir / f"models--{repo_id.replace('/', '--')}"
        ref_file = repo_dir / "refs" / "main"
        if not ref_file.is_file():
            return None
        commit_hash = ref_file.read_text().strip()
        snap = repo_dir / "snapshots" / commit_hash
        return snap if snap.is_dir() else None

    def _load_model(self):
        """Load ColQwen2 + connect Qdrant. Called once at startup via executor."""
        if self._initialized:
            return

        log.info("=== Model loading starting ===")

        self.valves.OPENROUTER_MODEL = os.getenv(
            "OPENROUTER_MODEL", self.valves.OPENROUTER_MODEL
        )
        self.valves.COLLECTION_NAME = os.getenv(
            "TARGET_KNOWLEDGE", self.valves.COLLECTION_NAME
        )

        try:
            model_name = os.getenv("COLQWEN2_MODEL", "vidore/colqwen2-v1.0")

            # Resolve local snapshot paths so that from_pretrained() loads
            # directly from disk and never contacts HuggingFace Hub.
            # ColQwen2 is a LoRA adapter on top of "vidore/colqwen2-base",
            # so we must resolve both the adapter and the base model.
            hf_hub_dir = (
                pathlib.Path(
                    os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                )
                / "hub"
            )
            adapter_snap = self._resolve_snapshot(hf_hub_dir, model_name)
            base_snap = self._resolve_snapshot(hf_hub_dir, "vidore/colqwen2-base")

            original_base = None
            adapter_cfg_path = None

            if adapter_snap and base_snap:
                log.info(
                    f"Loading ColQwen2 (CPU) from local cache:\n"
                    f"  adapter: {adapter_snap}\n"
                    f"  base:    {base_snap}"
                )
                # Temporarily patch the adapter_config so PEFT loads the
                # base model from the local snapshot instead of the Hub.
                adapter_cfg_path = adapter_snap / "adapter_config.json"
                adapter_cfg = json.loads(adapter_cfg_path.read_text())
                original_base = adapter_cfg.get("base_model_name_or_path")
                if original_base and not pathlib.Path(original_base).is_dir():
                    adapter_cfg["base_model_name_or_path"] = str(base_snap)
                    adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2))
                    log.info(f"  Patched adapter_config base_model → {base_snap}")

                load_path = str(adapter_snap)
            else:
                log.info(f"Loading ColQwen2 (CPU) — downloading '{model_name}' …")
                load_path = model_name

            self.model = ColQwen2.from_pretrained(
                load_path,
                torch_dtype=torch.float32,
                device_map="cpu",
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained(load_path)

            # Restore original adapter_config so future online loads still work
            if original_base and adapter_cfg_path:
                adapter_cfg["base_model_name_or_path"] = original_base
                adapter_cfg_path.write_text(json.dumps(adapter_cfg, indent=2))

            log.info("✓ ColQwen2 ready")

            self.qdrant = QdrantClient(
                host=self.valves.QDRANT_HOST,
                port=self.valves.QDRANT_PORT,
                timeout=60,
            )
            log.info(
                f"✓ Qdrant connected at {self.valves.QDRANT_HOST}:{self.valves.QDRANT_PORT}"
            )

            self._initialized = True
            log.info("=== Model loading complete ===")

        except Exception as e:
            log.error(f"!!! Model loading failed: {e}", exc_info=True)
            raise

    # ── background indexing ───────────────────────────────────────────

    def _start_background_index(self):
        """Spawn a daemon thread to run indexing, unless one is already running."""
        if self._index_thread and self._index_thread.is_alive():
            self._pending_trigger = True  # re-run after current pass finishes
            log.info(
                "Indexing already in progress — trigger queued for after current run"
            )
            return
        self._pending_trigger = False
        self._index_thread = threading.Thread(
            target=self._background_index_worker,
            daemon=True,
            name="pdf-indexer",
        )
        self._index_thread.start()
        log.info("Background indexing thread started")

    def _background_index_worker(self):
        self._cancel_flag.clear()
        self._cancel_hard.clear()
        self._current_indexing_file = None
        with self._index_lock:
            try:
                self._index_local_pdfs()
            except Exception as e:
                log.error(f"Background indexing failed: {e}", exc_info=True)
        # Decide whether to spawn another pass.
        # Must spawn directly — calling _start_background_index() from inside the
        # worker thread would see self._index_thread.is_alive()==True and deadlock.
        # Always restart to process remaining queue, unless normal idle completion.
        # Must spawn directly — calling _start_background_index() from inside the
        # worker thread would see self._index_thread.is_alive()==True and deadlock.
        if self._cancel_flag.is_set() or self._pending_trigger:
            self._pending_trigger = False
            log.info("Restarting indexer to continue remaining queue")
            t = threading.Thread(
                target=self._background_index_worker, daemon=True, name="pdf-indexer"
            )
            self._index_thread = t
            t.start()

    def _format_index_status(self) -> str:
        """Return a human-readable markdown string describing current index state."""
        state = self._load_state()
        indexed = state.get("indexed_files", [])
        job = state.get("index_job", {})

        lines = ["**Indexing Status**\n"]
        if job.get("active"):
            lines.append(f"🔄 **In progress:** `{job['current_file']}`")
            lines.append(f"   Page {job['current_page']} / {job['total_pages']}")
        else:
            lines.append("✅ **Idle** (no active indexing job)")

        if indexed:
            lines.append(f"\n**Indexed files ({len(indexed)}):**")
            for fn in indexed:
                lines.append(f"  - {fn}")
        else:
            lines.append("\nNo files indexed yet.")

        return "\n".join(lines)

    # ── indexing ─────────────────────────────────────────────────────

    def _index_local_pdfs(self):
        from pdf2image import pdfinfo_from_path

        collection = self.valves.COLLECTION_NAME
        pdf_dir = pathlib.Path(self.valves.PDF_DIR)
        pdfs = sorted(pdf_dir.rglob("*.pdf"))
        if not pdfs:
            log.warning(f"No PDFs found in {pdf_dir}")
            return

        state = self._load_state()

        # ── Schema migration: wipe old mean-pooled index ────────────
        if state.get("schema_version", 1) != SCHEMA_VERSION:
            log.info(
                f"Schema v{state.get('schema_version', 1)} → v{SCHEMA_VERSION}: re-indexing all PDFs"
            )
            try:
                self.qdrant.delete_collection(collection)
            except Exception:
                pass
            state = {
                "schema_version": SCHEMA_VERSION,
                "indexed_files": [],
                "file_progress": {},
            }
            self._save_state(state)

        indexed = set(state.get("indexed_files", []))
        file_progress = state.get("file_progress", {})

        # Fresh files first, then partially-indexed, skipped file (last paused/cancelled) last.
        skipped = self._skipped_file
        self._skipped_file = None  # consume it

        def _sort_key(p):
            rel = str(p.relative_to(pdf_dir))
            if rel == skipped:
                return (2, rel)  # deferred — goes last
            if rel in file_progress:
                return (1, rel)  # partial progress
            return (0, rel)  # fresh — goes first

        to_index = sorted(
            [p for p in pdfs if str(p.relative_to(pdf_dir)) not in indexed],
            key=_sort_key,
        )

        if not to_index:
            log.info("All PDFs already indexed – skipping")
            return

        log.info(f"Found {len(to_index)} PDFs to index or resume")

        # ── Create collection with MULTI-VECTOR MaxSim ──────────────
        try:
            self.qdrant.get_collection(collection)
        except Exception:
            dummy = self.processor.process_images([Image.new("RGB", (32, 32))])
            with torch.no_grad():
                dim = self.model(**dummy).shape[-1]
            self.qdrant.create_collection(
                collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    ),
                ),
            )
            log.info(f"Created collection '{collection}' dim={dim} with MaxSim")

        # ── Reconcile indexed_files against Qdrant ──────────────────
        # Done here (after collection is confirmed reachable) so Qdrant scroll is safe.
        # Removes any file that claims to be indexed but has no vectors (e.g. after OOM restart).
        if indexed:
            ghost_files = []
            for filename in list(indexed):
                pts, _ = self.qdrant.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="source", match=MatchValue(value=filename)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                if not pts:
                    ghost_files.append(filename)
            if ghost_files:
                log.info(
                    f"Reconciliation: removing {len(ghost_files)} phantom entries → {ghost_files}"
                )
                state = self._load_state()
                state["indexed_files"] = [
                    f for f in state["indexed_files"] if f not in ghost_files
                ]
                self._save_state(state)
                indexed -= set(ghost_files)
                # Rebuild to_index to include the newly discovered ghosts
                to_index = sorted(
                    [p for p in pdfs if str(p.relative_to(pdf_dir)) not in indexed],
                    key=_sort_key,
                )
                log.info(f"After reconciliation: {len(to_index)} PDFs to index")

        for pdf_file in to_index:
            if self._cancel_flag.is_set():
                log.info("Indexing cancelled — stopping before next file")
                break
            rel = str(pdf_file.relative_to(pdf_dir))
            self._current_indexing_file = rel
            # Resume from the last completed page, or start from 1
            start_page = file_progress.get(rel, 0) + 1
            log.info(f"  Indexing {rel} (from page {start_page}) …")
            self._ingest(rel, collection, start_page)
            if self._cancel_flag.is_set():
                log.info(f"Indexing cancelled after {rel}")
                self._skipped_file = rel  # defer this file to end of queue on next pass
                break
            # Reload state before writing so sidecar deletions made during indexing are preserved.
            # Only add to indexed_files if the PDF still exists — the sidecar may have deleted it
            # while we were indexing other files, and we must not re-add a deleted file.
            state = self._load_state()
            if (
                pathlib.Path(self.valves.PDF_DIR) / rel
            ).exists() and rel not in state.get("indexed_files", []):
                state.setdefault("indexed_files", []).append(rel)
            state["schema_version"] = SCHEMA_VERSION
            state.get("file_progress", {}).pop(rel, None)
            self._save_state(state)
            self._current_indexing_file = (
                None  # clear so cancel-between-files doesn't blame this file
            )

        # Clear index_job on completion or cancellation
        state = self._load_state()
        if self._cancel_flag.is_set():
            if self._cancel_hard.is_set():
                # Hard cancel: delete PDF from disk + Qdrant vectors + clear state
                current_file = self._current_indexing_file
                if current_file:
                    state.get("file_progress", {}).pop(current_file, None)
                    try:
                        self.qdrant.delete(
                            collection_name=collection,
                            points_selector=FilterSelector(
                                filter=Filter(
                                    must=[
                                        FieldCondition(
                                            key="source",
                                            match=MatchValue(value=current_file),
                                        )
                                    ]
                                )
                            ),
                        )
                        log.info(
                            f"Hard cancel: deleted Qdrant vectors for {current_file}"
                        )
                    except Exception as e:
                        log.warning(
                            f"Hard cancel: Qdrant cleanup failed for {current_file}: {e}"
                        )
                    try:
                        pdf_path = pathlib.Path(self.valves.PDF_DIR) / current_file
                        if pdf_path.exists():
                            pdf_path.unlink()
                            log.info(f"Hard cancel: deleted file {pdf_path}")
                    except Exception as e:
                        log.warning(
                            f"Hard cancel: file deletion failed for {current_file}: {e}"
                        )
                    log.info(f"Hard cancel complete — {current_file} deleted")
            else:
                log.info(
                    "Pause — partial progress checkpointed, will resume on next run"
                )
        else:
            log.info("✓ All PDFs indexed")
        state["index_job"] = {}
        self._save_state(state)

    def _ingest(self, filename: str, collection: str, start_page: int = 1) -> None:
        from pdf2image import pdfinfo_from_path

        pdf_path = pathlib.Path(self.valves.PDF_DIR) / filename
        total_pages = pdfinfo_from_path(str(pdf_path))["Pages"]
        log.info(f"    {filename}: {total_pages} total pages")

        # Write index_job immediately so UI shows progress before first page completes
        state = self._load_state()
        state.setdefault("file_progress", {})
        state["index_job"] = {
            "active": True,
            "current_file": filename,
            "current_page": 0,
            "total_pages": total_pages,
        }
        self._save_state(state)

        for page_num in range(start_page, total_pages + 1):
            if self._cancel_flag.is_set():
                log.info(
                    f"    Cancelled at page {page_num}/{total_pages} — progress saved"
                )
                break
            # Convert one page at a time to avoid loading the full PDF into memory
            page_img = convert_from_path(
                str(pdf_path), dpi=200, first_page=page_num, last_page=page_num
            )[0].convert("RGB")

            img_filename = self._save_page_image(page_img, filename, page_num)

            # Pause between pages if a query is actively using the model
            if self._query_active.is_set():
                log.info(f"    Pausing indexing — query in progress")
                while self._query_active.is_set():
                    time.sleep(0.2)
                time.sleep(0.5)  # let OS reclaim query tensors before next page
                log.info(f"    Resuming indexing")

            batch = self.processor.process_images([page_img])
            with torch.no_grad():
                emb = self.model(**batch)
            multi_vec = emb[0].tolist()
            del page_img, batch, emb  # free memory immediately
            gc.collect()

            # Check again after the slow embed step — skip upsert if cancelled
            if self._cancel_flag.is_set():
                log.info(
                    f"    Cancelled after embed page {page_num}/{total_pages} — skipping upsert"
                )
                break

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{page_num}"))
            self.qdrant.upsert(
                collection,
                [
                    PointStruct(
                        id=point_id,
                        vector=multi_vec,
                        payload={
                            "source": filename,
                            "page_number": page_num,
                            "image_filename": img_filename,
                        },
                    )
                ],
            )

            log.info(
                f"    page {page_num}/{total_pages}  id={point_id}  patches={len(multi_vec)}"
            )

            # Checkpoint: reload fresh state so any sidecar deletions made during this
            # long-running ingest are not overwritten by a stale state object.
            state = self._load_state()
            state.setdefault("file_progress", {})[filename] = page_num
            state["index_job"] = {
                "active": True,
                "current_file": filename,
                "current_page": page_num,
                "total_pages": total_pages,
            }
            self._save_state(state)

    # ── search with MULTI-VECTOR MaxSim ──────────────────────────────

    def _search(self, query: str, top_k: int = 5):
        self._query_active.set()
        try:
            batch = self.processor.process_queries([query])
            with torch.no_grad():
                emb = self.model(**batch)
            q_vecs = emb[0].tolist()  # shape: (num_query_tokens, dim)
            del batch, emb
        finally:
            self._query_active.clear()
            gc.collect()  # reclaim query tensors before indexing resumes

        # ── Full multi-vector query (all query tokens) ──────────────
        log.info(f"Query encoded → {len(q_vecs)} tokens")

        return self.qdrant.query_points(
            collection_name=self.valves.COLLECTION_NAME,
            query=q_vecs,  # ← MULTI-VECTOR query
            limit=top_k,
            search_params=SearchParams(exact=False),
            with_payload=True,
        ).points

    # ── VLM helpers ───────────────────────────────────────────────────

    def _expand_refs(self, text: str, hits, cited: set) -> str:
        """Replace [REF:N ...] variants with bold page links.
        Handles: [REF:1], [REF:1, REF:2], [REF:1, 2], [REF:1, Page 26],
                 [REF:1, REF:2, Page 10], and similar VLM output variations.
        """

        def _sub(m):
            # Extract only REF indices — numbers preceded by "REF:" or "REF :"
            # This avoids capturing page numbers like "Page 26"
            nums = re.findall(r"REF\s*:\s*(\d+)", m.group(0), re.IGNORECASE)
            if not nums:
                # Fallback: if the VLM wrote something like [REF:1, 2] (shorthand)
                # extract all bare numbers that aren't preceded by "Page"
                nums = re.findall(r"(?<!Page\s)(?<!Page)(?<!p)(\d+)", m.group(0))
            parts = []
            seen = set()
            for n in nums:
                if n in seen:
                    continue
                seen.add(n)
                idx = int(n) - 1
                if 0 <= idx < len(hits):
                    cited.add(idx)
                    hit = hits[idx]
                    pg = hit.payload.get("page_number", "?")
                    source = hit.payload.get("source", "")
                    base = f"http://{self.valves.SERVER_HOST}:8082/view"
                    url = f"{base}/{pg}/{source}" if source else ""
                    parts.append(
                        f"**[\\[Page {pg}\\]]({url})**" if url else f"Page {pg}"
                    )
            return " ".join(parts) if parts else m.group(0)

        # Broad pattern: matches [REF:N ...] with any content up to closing ]
        return re.sub(
            r"\[REF\s*:\s*[^\]]+\]",
            _sub,
            text,
            flags=re.IGNORECASE,
        )

    def _linkify_plain_pages(self, text: str, hits, cited: set) -> str:
        """Fallback: convert plain 'Page X' or '[Page X]' mentions into clickable
        links if that page number exists in the retrieved hits.
        Skips pages that were already linked by _expand_refs (contain markdown links)."""
        # Build a lookup: page_number → (hit_index, source)
        page_map = {}
        for i, h in enumerate(hits):
            pg = h.payload.get("page_number")
            src = h.payload.get("source", "")
            if pg is not None and pg not in page_map:
                page_map[pg] = (i, src)

        def _sub_plain(m):
            pg = int(m.group(1))
            if pg not in page_map:
                return m.group(0)  # not in retrieved pages, leave as-is
            idx, source = page_map[pg]
            cited.add(idx)
            base = f"http://{self.valves.SERVER_HOST}:8082/view"
            url = f"{base}/{pg}/{source}" if source else ""
            if url:
                return f"**[\\[Page {pg}\\]]({url})**"
            return m.group(0)

        # Match "Page 29", "page 29", "[Page 29]" but NOT already-linked
        # "[\[Page 29\]](http://...)" (those contain backslash-escaped brackets)
        return re.sub(
            r"(?<!\\\[)\b[Pp]age\s+(\d+)\b(?!\\\])",
            _sub_plain,
            text,
        )

    def _build_source_table(self, cited_hits) -> str:
        """Build a markdown thumbnail table from a list of hits. Returns empty string if no images."""
        headers, divider, images = [], [], []
        for hit in cited_hits:
            img_filename = hit.payload.get("image_filename", "")
            source = hit.payload.get("source", "unknown")
            page = hit.payload.get("page_number", "?")
            score = hit.score
            if img_filename:
                thumb_filename = self._make_thumbnail_file(img_filename)
                img_base = f"http://{self.valves.SERVER_HOST}:8081"
                full_url = f"{img_base}/{img_filename}"
                thumb_path = os.path.join(self.valves.IMAGE_CACHE_DIR, thumb_filename)
                mtime = (
                    int(os.path.getmtime(thumb_path))
                    if os.path.exists(thumb_path)
                    else 0
                )
                thumb_url = f"{img_base}/{thumb_filename}?v={mtime}"
                headers.append(f"p{page} · {source} ({score:.2f})")
                divider.append(":---:")
                images.append(f"[![p{page}]({thumb_url})]({full_url})")
        if not images:
            return ""
        return (
            "| " + " | ".join(headers) + " |\n"
            "| " + " | ".join(divider) + " |\n"
            "| " + " | ".join(images) + " |\n\n"
        )

    def _stream_vlm(self, query: str, hits, cited: set):
        """Generator: stream VLM SSE response, replacing [REF:N] inline.
        Supports two backends selected by the VLM_PROVIDER valve:
          - "openrouter"  : cloud API via openrouter.ai (requires OPENROUTER_API_KEY)
          - "ollama"      : local Ollama instance (requires model pulled, no API key)
        """
        import requests as _req

        provider = self.valves.VLM_PROVIDER.lower()

        if provider == "ollama":
            url = f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            model = self.valves.OLLAMA_VLM_MODEL
        else:  # openrouter (default)
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                yield "Error: OPENROUTER_API_KEY not set in .env."
                return
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            model = self.valves.OPENROUTER_MODEL

        content_parts = [{"type": "text", "text": query}]
        for i, hit in enumerate(hits, 1):
            img_filename = hit.payload.get("image_filename", "")
            src = hit.payload.get("source", "?")
            pg = hit.payload.get("page_number", "?")
            if img_filename:
                img_b64 = self._load_page_image_b64(img_filename)
                content_parts.append(
                    {"type": "text", "text": f"[REF:{i}] Page {pg} from {src}"}
                )
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )

        with _req.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert document analyst and technical writer. "
                            "You are given page images from the user's document collection, each labeled [REF:N].\n\n"
                            "RESPONSE STYLE:\n"
                            "- Write thorough, well-structured answers. Never give a one-liner when a complete explanation is warranted.\n"
                            "- Use markdown formatting to make answers visually clear: headers (##/###) to organize sections, "
                            "bullet points or numbered lists for sequences and enumerations, "
                            "**bold** for key terms and figure/table references, `code` for values/commands/identifiers, "
                            "and tables where comparing multiple items. Do NOT use italic (*text*).\n"
                            "- Open with a short direct answer, then elaborate with details, context, and examples from the pages.\n"
                            "- Close with a brief summary or takeaway when the answer is long.\n\n"
                            "CITATION RULES — follow exactly:\n"
                            "1. Answer using ONLY information visible in the provided pages.\n"
                            "2. Every factual claim MUST be followed immediately by its inline citation [REF:N] — no exceptions "
                            "(example: 'The voltage range is 3.3 V to 5 V [REF:2].').\n"
                            "3. Only cite a page if it DIRECTLY provides the information used. "
                            "Do NOT cite pages that are only tangentially related or marginally mentioned.\n"
                            "4. Chain citations when information spans multiple pages "
                            "(example: 'Connect A to B [REF:1], then B to C [REF:3].').\n"
                            "5. If none of the pages contain relevant information, say so plainly."
                        ),
                    },
                    {"role": "user", "content": content_parts},
                ],
                "max_tokens": 4096,
                "stream": True,
            },
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            carry = ""
            try:
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    data = line[6:]
                    if data == b"[DONE]":
                        break
                    parsed = json.loads(data)
                    choices = parsed.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0]["delta"].get("content", "")
                    if not delta:
                        continue

                    text = carry + delta
                    carry = ""

                    # Hold back from the first unclosed [ (multi-ref safe)
                    open_pos = text.find("[")
                    while open_pos != -1:
                        close_pos = text.find("]", open_pos)
                        if close_pos == -1:
                            carry = text[open_pos:]
                            text = text[:open_pos]
                            break
                        open_pos = text.find("[", close_pos + 1)

                    text = self._expand_refs(text, hits, cited)
                    text = self._linkify_plain_pages(text, hits, cited)
                    if text:
                        yield text
            except Exception as stream_err:
                log.warning(f"Stream cut short: {stream_err}")

            # Flush carry buffer (also handles graceful stream cut)
            if carry:
                carry = self._expand_refs(carry, hits, cited)
                carry = self._linkify_plain_pages(carry, hits, cited)
                if carry:
                    yield carry

    # ── streaming entry point ─────────────────────────────────────────

    def _pipe_stream(self, query: str):
        """Generator yielding streaming response chunks for a normal query."""
        try:
            # Step 1 — show retrieval status immediately
            yield "> 🔍 Searching indexed documents…\n\n"

            hits = self._search(query, top_k=self.valves.TOP_K)
            if self.valves.SCORE_THRESHOLD > 0:
                hits = [h for h in hits if h.score >= self.valves.SCORE_THRESHOLD]

            if not hits:
                yield "No relevant pages found in the indexed documents."
                return

            log.info(f"Retrieved {len(hits)} pages")
            for h in hits:
                log.info(
                    f"  → {h.payload.get('source')} p{h.payload.get('page_number')} "
                    f"score={h.score:.4f}"
                )

            # Step 2 — pages found summary
            lines = [f"> 📄 **Found {len(hits)} pages — sending to Vision model:**"]
            for h in hits:
                src = h.payload.get("source", "?")
                pg = h.payload.get("page_number", "?")
                lines.append(f">   - `{src}` p{pg} (score: {h.score:.2f})")
            yield "\n".join(lines) + "\n\n---\n\n"

            # Step 3 — stream VLM answer
            cited: set = set()
            yield from self._stream_vlm(query, hits, cited)

            # Step 4 — source thumbnails (only cited pages above score threshold)
            if self.valves.SHOW_SOURCE_PAGES:
                cited_hits = [hits[i] for i in sorted(cited) if i < len(hits)]
                if self.valves.THUMBNAIL_SCORE_THRESHOLD > 0:
                    cited_hits = [
                        h
                        for h in cited_hits
                        if h.score >= self.valves.THUMBNAIL_SCORE_THRESHOLD
                    ]
                table = self._build_source_table(cited_hits)
                if table:
                    yield "\n\n---\n\n**📄 Source Pages:**\n\n" + table

        except Exception as e:
            log.error(f"pipe stream error: {e}", exc_info=True)
            yield f"\n\nError: {e}"

    # ── main entry ───────────────────────────────────────────────────

    def pipe(self, body: dict, **kwargs):
        if not self._initialized:
            return "Pipeline initializing — model loading in progress. Please retry in ~60 seconds."

        messages = body.get("messages", [])
        query = messages[-1]["content"] if messages else ""
        if not query:
            return "Please ask a question about your documents."

        log.info(f"Query: {query}")

        # Internal trigger from pdf-ingest sidecar
        if query.strip() == "__index_now__":
            self._start_background_index()
            return "__ok__"

        if query.strip() == "__cancel_index__":
            self._cancel_hard.set()
            self._cancel_flag.set()
            log.info(
                "Hard cancel — indexing will stop, progress cleared, partial vectors removed"
            )
            return "__ok__"

        if query.strip() == "__pause_index__":
            self._cancel_flag.set()
            log.info("Pause — indexing will stop, progress saved for resume")
            return "__ok__"

        # User status query
        if query.strip().lower() in ("status", "indexing status", "/status"):
            return self._format_index_status()

        # Normal query → streaming generator
        return self._pipe_stream(query)
