"""
title: ColQwen2 Visual RAG (CPU) – Multi-Vector MaxSim
author: adapted
version: 10.1
license: MIT
description: Visual RAG pipeline using ColQwen2 multi-vector MaxSim + Qdrant + OpenRouter VLM.
             Background indexing thread — queries are never blocked by indexing.
"""

import asyncio
import os, json, base64, io, logging, pathlib, hashlib, re, threading, time, gc, typing, uuid
from typing import List, Literal, Optional

import torch
from PIL import Image
from pydantic import BaseModel, model_validator
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
LABELS_FILE = "/app/pipelines/labels.json"
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
        # Note: only include models whose production OpenRouter deployment
        # actually accepts image inputs. OpenRouter's model card "image"
        # modality flag is sometimes ahead of upstream provider support
        # (e.g. moonshotai/kimi-k2.6 advertises vision but the default-
        # routed AtlasCloud deployment rejects image payloads with HTTP 400).
        # For such models we pin the upstream provider via PROVIDER_PINS
        # in _stream_vlm (see provider routing logic there).
        OPENROUTER_MODEL: Literal[
            "qwen/qwen3-vl-30b-a3b-instruct",
            "qwen/qwen3-vl-235b-a22b-instruct",
            "qwen/qwen3-vl-235b-a22b-thinking",
            "qwen/qwen3.5-122b-a10b",
            "qwen/qwen3.6-35b-a3b",
            "qwen/qwen3.6-plus",
            "google/gemini-3.1-pro-preview",
            "google/gemma-4-31b-it",
            "openai/gpt-5.2",
            "anthropic/claude-opus-4.7",
            "z-ai/glm-4.6v",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "moonshotai/kimi-k2.6",  # pinned to Moonshot AI provider (only one that accepts images)
        ] = "qwen/qwen3-vl-30b-a3b-instruct"
        THUMBNAIL_SCORE_THRESHOLD: float = (
            0.0  # min score for cited page to show thumbnail; 0 = show all
        )
        OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        OLLAMA_VLM_MODEL: str = "qwen3-vl:30b"  # model must be pulled in Ollama first
        SHOW_SOURCE_PAGES: bool = True
        SERVER_HOST: str = os.getenv("SERVER_HOST", "localhost")
        IMAGE_CACHE_DIR: str = "/app/pipelines/cache/images"
        # ── Page rendering quality (affects images sent to the VLM) ──
        # Higher DPI improves small-text reading (schematics, datasheets)
        # at the cost of larger payloads and slower indexing. Only applied
        # to newly-indexed pages — delete IMAGE_CACHE_DIR and re-index to
        # regenerate existing pages at the new DPI.
        INDEXING_DPI: int = 300
        INDEXING_JPEG_QUALITY: int = 90

        @model_validator(mode="before")
        @classmethod
        def _sanitize_stale_openrouter_model(cls, data):
            """Drop a persisted OPENROUTER_MODEL value that is no longer in the
            Literal[...] choices (e.g. a model ID that was removed from the
            dropdown or from OpenRouter's catalog). Without this, Open WebUI
            replaying a stale valves.json would crash pipeline startup with a
            Pydantic literal_error. We silently fall back to the field default.
            """
            if not isinstance(data, dict):
                return data
            model_value = data.get("OPENROUTER_MODEL")
            if model_value is None:
                return data
            allowed = typing.get_args(
                cls.model_fields["OPENROUTER_MODEL"].annotation
            )
            if model_value not in allowed:
                log.warning(
                    "Discarding stale OPENROUTER_MODEL %r from persisted valves "
                    "(not in current choices %s); falling back to default.",
                    model_value,
                    list(allowed),
                )
                data.pop("OPENROUTER_MODEL", None)
            return data

    def __init__(self):
        self.name = "Search"
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

    def _load_labels(self) -> dict:
        """Load labels.json: { "filename.pdf": ["label1", ...], ... }"""
        if os.path.exists(LABELS_FILE):
            try:
                with open(LABELS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _get_file_labels(self, filename: str) -> list:
        """Return all labels for a file: always includes the filename itself."""
        all_labels = self._load_labels()
        user_labels = all_labels.get(filename, [])
        # Always include the filename (without .pdf extension) as an implicit label
        stem = pathlib.Path(filename).stem
        auto_labels = [filename, stem]
        # Combine: auto labels + user labels, deduped, preserving order
        combined = []
        seen = set()
        for label in auto_labels + user_labels:
            lower = label.lower()
            if lower not in seen:
                seen.add(lower)
                combined.append(label)
        return combined

    def _save_page_image(
        self, page_img: Image.Image, filename: str, page_num: int
    ) -> str:
        """Save page image to cache dir, return filename."""
        os.makedirs(self.valves.IMAGE_CACHE_DIR, exist_ok=True)
        safe_name = pathlib.Path(filename).stem
        img_filename = f"{safe_name}_p{page_num}.jpg"
        img_path = os.path.join(self.valves.IMAGE_CACHE_DIR, img_filename)
        page_img.save(
            img_path,
            format="JPEG",
            quality=self.valves.INDEXING_JPEG_QUALITY,
        )
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

        # Allow env var override, but only if the value is one of the allowed
        # Literal choices declared on the Valves model — otherwise Pydantic
        # would raise on assignment. Fall back to the default silently.
        _env_model = os.getenv("OPENROUTER_MODEL")
        if _env_model:
            _allowed = typing.get_args(
                self.Valves.model_fields["OPENROUTER_MODEL"].annotation
            )
            if _env_model in _allowed:
                self.valves.OPENROUTER_MODEL = _env_model
            else:
                log.warning(
                    "OPENROUTER_MODEL env var %r is not in allowed choices %s; "
                    "using default %r",
                    _env_model,
                    list(_allowed),
                    self.valves.OPENROUTER_MODEL,
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
            all_labels = self._load_labels()
            for fn in indexed:
                file_labels = all_labels.get(fn, [])
                label_str = f" [labels: {', '.join(file_labels)}]" if file_labels else ""
                lines.append(f"  - {fn}{label_str}")
        else:
            lines.append("\nNo files indexed yet.")

        lines.append(
            "\n**Tip:** Use `label:name` prefix in your query to filter by label "
            "(e.g. `label:arduino pinout diagram`)."
        )

        return "\n".join(lines)

    def _format_labels_list(self) -> str:
        """Return a formatted list of all available labels for filtering."""
        all_labels = self._load_labels()
        state = self._load_state()
        indexed = state.get("indexed_files", [])

        # Build a map: label → list of filenames
        label_to_files: dict = {}
        for fn in indexed:
            file_labels = self._get_file_labels(fn)
            for lbl in file_labels:
                label_to_files.setdefault(lbl, []).append(fn)

        if not label_to_files:
            return (
                "No labels found. Upload documents with labels via the "
                "PDF Indexer dashboard (port 8082)."
            )

        # Separate user-defined labels from auto-generated ones
        auto_labels = {}  # labels that are just filenames/stems
        user_labels = {}  # labels explicitly added by users
        all_filenames = set(indexed)
        all_stems = {pathlib.Path(fn).stem for fn in indexed}
        for lbl, files in label_to_files.items():
            if lbl in all_filenames or lbl in all_stems:
                auto_labels[lbl] = files
            else:
                user_labels[lbl] = files

        lines = ["**Available Labels** — case-insensitive\n"]
        lines.append("Copy a snippet below and paste it before your question:\n")

        if user_labels:
            lines.append("### Custom Labels")
            for lbl in sorted(user_labels.keys(), key=str.lower):
                files = user_labels[lbl]
                snippet = f'/label:"{lbl}"' if " " in lbl or "/" in lbl else f"/label:{lbl}"
                count = len(files)
                lines.append(
                    f"  - `{snippet}` — {count} doc{'s' if count != 1 else ''}"
                )
            lines.append("")

        if auto_labels:
            lines.append("### Document Names")
            for lbl in sorted(auto_labels.keys(), key=str.lower):
                # Only show stems (shorter), skip full filenames with .pdf
                if lbl.lower().endswith(".pdf"):
                    continue
                files = auto_labels[lbl]
                snippet = f'/label:"{lbl}"' if " " in lbl or "/" in lbl else f"/label:{lbl}"
                lines.append(f"  - `{snippet}`")
            lines.append("")

        lines.append("---")
        lines.append(
            "**Examples:**\n"
            "  - `/label:toyota engine specs` — search Toyota docs only\n"
            "  - `/label:Confluence project overview` — Confluence pages only\n"
            "  - `/label:toyota /label:Confluence meeting notes` — both labels must match"
        )

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

        # Load labels for this file (includes auto-generated filename label)
        file_labels = self._get_file_labels(filename)
        log.info(f"    Labels for {filename}: {file_labels}")

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
                str(pdf_path),
                dpi=self.valves.INDEXING_DPI,
                first_page=page_num,
                last_page=page_num,
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
                            "labels": file_labels,
                            "labels_lower": [l.lower() for l in file_labels],
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

    @staticmethod
    def _parse_label_filters(query: str) -> tuple:
        """Parse label filter prefixes from query.

        Supports (with or without leading /):
          - label:value          /label:value
          - label:"my value"     /label:"my value"
          - labels:value         /labels:value
          - Multiple: can repeat prefix

        Returns (clean_query, label_list) where label_list may be empty.
        """
        label_filters = []
        clean = query

        # Match /label:"quoted" or /label:unquoted (leading / is optional)
        pattern = r'/?labels?:"([^"]+)"|/?labels?:(\S+)'
        matches = re.findall(pattern, clean, re.IGNORECASE)
        for quoted, unquoted in matches:
            val = (quoted or unquoted).strip()
            if val:
                label_filters.append(val)

        # Remove matched patterns from query
        clean = re.sub(r'\s*/?labels?:"[^"]+"\s*', ' ', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s*/?labels?:\S+\s*', ' ', clean, flags=re.IGNORECASE)
        clean = clean.strip()

        return clean, label_filters

    def _search(self, query: str, top_k: int = 5, label_filters: list = None):
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

        # Build optional Qdrant filter for labels
        query_filter = None
        if label_filters:
            # Filter against labels_lower for case-insensitive matching.
            # All conditions in "must" are ANDed = all labels must be present.
            conditions = []
            for lbl in label_filters:
                conditions.append(
                    FieldCondition(
                        key="labels_lower",
                        match=MatchValue(value=lbl.lower()),
                    )
                )
            query_filter = Filter(must=conditions)
            log.info(f"Label filter applied (case-insensitive): {label_filters}")

        return self.qdrant.query_points(
            collection_name=self.valves.COLLECTION_NAME,
            query=q_vecs,  # ← MULTI-VECTOR query
            query_filter=query_filter,
            limit=top_k,
            search_params=SearchParams(exact=False),
            with_payload=True,
        ).points

    # ── VLM helpers ───────────────────────────────────────────────────

    def _expand_refs(self, text: str, hits, cited: set) -> str:
        """Replace [REF:N ...] and (REF:N ...) variants with bold page links.
        Handles: [REF:1], (REF:1), [REF:1, REF:2], [REF:1, 2], [REF:1, Page 26],
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

        # Matches both [REF:N ...] and (REF:N ...) with any content up to closing bracket
        return re.sub(
            r"[\[(]REF\s*:\s*[^\])]+[\])]",
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

        # ── Reasoning policy per model family ──
        # Some models burn an enormous reasoning budget that consumes most
        # of max_tokens before producing any visible content, causing the
        # answer to be truncated (finish_reason='length'). For those, fully
        # disable reasoning. For others, keep reasoning but exclude it from
        # the visible stream (preserves accuracy on hard visual tasks
        # without leaking <think> blocks into the user-facing answer).
        m = model.lower()
        REASONING_RUNAWAY_MODELS = (
            "qwen3.5",  # Qwen 3.5 122B/35B/9B reasoning generates 10K+ tokens
            "qwen3.6",  # Same family behavior
            "nemotron-3",  # Nemotron 3 reasoning variants
            "kimi-k2",  # Kimi K2.x reasons silently for minutes via Moonshot;
            #            reasoning tokens never reach the stream so we'd
            #            otherwise wait forever. Disabling reasoning lets
            #            content stream immediately.
        )
        if any(tag in m for tag in REASONING_RUNAWAY_MODELS):
            reasoning_param = {"enabled": False}
        else:
            reasoning_param = {"exclude": True}

        # ── Provider routing pins ──
        # Some models on OpenRouter route by default to providers whose
        # production deployment doesn't honor the model's full capabilities.
        # Most notable: moonshotai/kimi-k2.6 is served by multiple providers
        # but only Moonshot AI's own endpoint accepts image inputs — the
        # default load-balanced route (AtlasCloud) returns HTTP 400 on any
        # image payload. For such models we pin the upstream provider via
        # OpenRouter's `provider.only` + `allow_fallbacks: false` mechanism.
        PROVIDER_PINS = {
            # model_id (substring match) -> required provider slug
            "moonshotai/kimi-k2.6": "moonshotai",
        }
        provider_param = None
        for model_tag, provider_slug in PROVIDER_PINS.items():
            if model_tag in m:
                provider_param = {
                    "only": [provider_slug],
                    "allow_fallbacks": False,
                }
                break

        # ── Schematic-mode detection ──
        # Schematics, datasheets with pinouts, and circuit diagrams require
        # a different reading discipline than prose documents. Without
        # special instructions, mid-tier VLMs (Qwen3-VL-instruct, GLM-4.6v,
        # smaller models) tend to "autocomplete" the answer from training
        # data instead of literally transcribing what's printed in the
        # image — e.g. GLM-4.6v outputting a 40-pin pinout for the 28-pin
        # ATmega328 because it pattern-matched on "ATmega + Arduino" and
        # recalled a different chip in the family.
        #
        # We detect schematic-style queries by either:
        #   (a) keywords in the user query (pinout, schematic, circuit, etc.)
        #   (b) any retrieved filename containing "schematic"
        # When triggered, we append a schematic-specific addendum to the
        # system prompt. Frontier models (Gemini/GPT/Claude) ignore the
        # extra instructions when not relevant; mid-tier models get a much
        # stronger anchor to "transcribe, don't recall".
        SCHEMATIC_KEYWORDS = (
            "schematic", "pinout", "pin out", "pin-out", "pin number",
            "circuit diagram", "net name", "netname", "ic pin",
        )
        query_lower = query.lower()
        retrieved_filenames = " ".join(
            (h.payload.get("source", "") or "") for h in hits
        ).lower()
        schematic_mode = (
            any(kw in query_lower for kw in SCHEMATIC_KEYWORDS)
            or "schematic" in retrieved_filenames
        )
        log.info(
            "VLM POST → %s (reasoning=%s, schematic_mode=%s, provider=%s)",
            model,
            reasoning_param,
            schematic_mode,
            provider_param,
        )

        SCHEMATIC_ADDENDUM = (
            "\n\nSCHEMATIC-MODE INSTRUCTIONS — apply when reading electrical "
            "schematics, datasheet pinout diagrams, or circuit drawings:\n"
            "1. TRANSCRIBE, do not RECALL. Even if you recognize the part "
            "number, read every pin label DIRECTLY off the image. Do NOT "
            "supply pin functions, alternate-function names, or pin counts "
            "from your training data unless they are visibly printed on this "
            "specific page. A part number like 'ATmega328' may appear in "
            "many packages (DIP-28, TQFP-32, MLF-32); only use the package "
            "actually shown.\n"
            "2. Count pins BEFORE listing them. Look at the IC symbol and "
            "state the package pin count explicitly (e.g. 'This is a 28-pin "
            "DIP'). Your final table row count MUST match.\n"
            "3. For IC pinout requests, work in two passes:\n"
            "   - PASS 1: Scan the IC symbol's perimeter and list each pin "
            "as 'pin N: <exact text printed next to that pin>'. Do not skip, "
            "merge, reorder, or paraphrase labels.\n"
            "   - PASS 2: Render the requested table from your PASS 1 list. "
            "Verify the row count equals the pin count from step 2.\n"
            "4. If a label is illegible, cropped, or you are not confident, "
            "write '(unreadable)' for that pin — never guess.\n"
            "5. Net names, test point labels, and reference designators are "
            "only valid if PRINTED in this image. Do not import them from "
            "datasheet knowledge.\n"
            "6. After the table, briefly list any pins you marked unreadable "
            "so the user knows what to verify manually."
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
                            + (SCHEMATIC_ADDENDUM if schematic_mode else "")
                        ),
                    },
                    {"role": "user", "content": content_parts},
                ],
                "max_tokens": 16384,
                "stream": True,
                # Reasoning policy computed above based on model family:
                #   - {"enabled": False} for runaway-reasoning models that
                #     would otherwise burn all 16K tokens on hidden thinking
                #     (Qwen 3.5/3.6, Nemotron 3 reasoning variants).
                #   - {"exclude": True} for everything else: the model still
                #     reasons (preserving accuracy on hard visual tasks like
                #     schematics) but the reasoning trace is stripped from
                #     the streamed output so it doesn't leak into the answer.
                "reasoning": reasoning_param,
                # Optional per-model provider pin (computed above). Only
                # included when set, so default load-balancing is preserved
                # for models that don't need an explicit upstream provider.
                **({"provider": provider_param} if provider_param else {}),
            },
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            carry = ""
            finish_reason = None
            # ── Stream watchdog ──
            # OpenRouter sends ': OPENROUTER PROCESSING' SSE comment frames
            # as keepalives while upstream providers are working. These keep
            # the TCP connection alive (so requests.timeout never fires) but
            # don't carry any model output. If a provider gets stuck in a
            # long reasoning phase or hangs entirely, we'd wait forever.
            # Track wall-clock time since the last *meaningful* data chunk
            # (one with actual content) and abort if too long passes with
            # only keepalives or empty deltas.
            STREAM_IDLE_TIMEOUT_S = 120  # max seconds without a content token
            last_content_time = time.monotonic()
            stream_aborted = False
            try:
                for line in resp.iter_lines():
                    # Watchdog: check elapsed time since last meaningful chunk
                    if (
                        time.monotonic() - last_content_time
                        > STREAM_IDLE_TIMEOUT_S
                    ):
                        log.warning(
                            "VLM stream idle >%ds with no content tokens "
                            "(model=%s) — aborting. Upstream provider may "
                            "be stuck in a long reasoning phase.",
                            STREAM_IDLE_TIMEOUT_S,
                            model,
                        )
                        stream_aborted = True
                        break
                    if not line or not line.startswith(b"data: "):
                        # SSE comments (':' prefix) and blank keepalives:
                        # don't reset the watchdog — they're just heartbeats.
                        continue
                    data = line[6:]
                    if data == b"[DONE]":
                        break
                    parsed = json.loads(data)
                    choices = parsed.get("choices", [])
                    if not choices:
                        continue
                    # Track finish_reason — providers often only set it on the
                    # final chunk. Logged after the loop so we can diagnose
                    # "length" (max_tokens cutoff) vs "stop" (clean) vs others.
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                    delta = choices[0]["delta"].get("content", "")
                    if not delta:
                        continue
                    # Real content arrived — reset the watchdog.
                    last_content_time = time.monotonic()

                    text = carry + delta
                    carry = ""

                    # Hold back from the first unclosed [ or ( (multi-ref safe)
                    for opener in ("[", "("):
                        closer = "]" if opener == "[" else ")"
                        open_pos = text.find(opener)
                        while open_pos != -1:
                            close_pos = text.find(closer, open_pos)
                            if close_pos == -1:
                                carry = text[open_pos:]
                                text = text[:open_pos]
                                break
                            open_pos = text.find(opener, close_pos + 1)

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

            # If the watchdog killed the loop, surface a clear message to
            # the user instead of leaving them with a half-empty answer.
            if stream_aborted:
                yield (
                    f"\n\n*[Aborted: no content from `{model}` for "
                    f"{STREAM_IDLE_TIMEOUT_S}s — provider may be hung or "
                    f"stuck in extended reasoning. Try a faster model.]*"
                )

            # Log finish_reason so future truncations are diagnosable. Common
            # values: "stop" (clean), "length" (hit max_tokens — answer was
            # truncated), "content_filter", "error". When watchdog aborted,
            # finish_reason is unset (we never received a terminal chunk).
            if stream_aborted:
                log.warning(
                    "VLM stream aborted by idle watchdog (model=%s)", model
                )
            elif finish_reason and finish_reason != "stop":
                log.warning(
                    "VLM stream ended with finish_reason=%r (model=%s, "
                    "max_tokens=16384). Consider bumping max_tokens if 'length'.",
                    finish_reason,
                    model,
                )
            else:
                log.info("VLM stream ended cleanly (finish_reason=%r)", finish_reason)

    # ── streaming entry point ─────────────────────────────────────────

    def _pipe_stream(self, query: str):
        """Generator yielding streaming response chunks for a normal query."""
        try:
            # Parse label filters from query (e.g. "label:arduino pinout diagram")
            clean_query, label_filters = self._parse_label_filters(query)
            search_query = clean_query if clean_query else query

            # Step 1 — show retrieval status immediately
            if label_filters:
                filter_display = ", ".join(f"`{l}`" for l in label_filters)
                yield f"> 🔍 Searching documents filtered by labels: {filter_display}\n\n"
            else:
                yield "> 🔍 Searching all indexed documents…\n\n"

            hits = self._search(search_query, top_k=self.valves.TOP_K, label_filters=label_filters)
            if self.valves.SCORE_THRESHOLD > 0:
                hits = [h for h in hits if h.score >= self.valves.SCORE_THRESHOLD]

            if not hits:
                if label_filters:
                    filter_display = ", ".join(f"`{l}`" for l in label_filters)
                    yield (
                        f"No relevant pages found matching labels: {filter_display}.\n\n"
                        "Try broadening your search by removing the label filter, "
                        "or check the label names in the PDF Indexer dashboard."
                    )
                else:
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
            yield from self._stream_vlm(search_query, hits, cited)

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

        # List available labels
        if query.strip().lower() in ("/labels", "labels", "list labels"):
            return self._format_labels_list()

        # Normal query → streaming generator
        return self._pipe_stream(query)
