"""
title: ColQwen2 Visual RAG (CPU) – Multi-Vector MaxSim
author: adapted
version: 9.0
license: MIT
description: Visual RAG pipeline using ColQwen2 multi-vector MaxSim + Qdrant + OpenRouter VLM.
             Background indexing thread — queries are never blocked by indexing.
"""

import asyncio
import os, json, base64, io, logging, pathlib, hashlib, re, threading
from typing import List, Optional

import torch
from PIL import Image
from pydantic import BaseModel
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SearchParams,
    MultiVectorConfig, MultiVectorComparator,
)
from colpali_engine.models import ColQwen2, ColQwen2Processor

log = logging.getLogger("colpali-pipeline")
log.setLevel(logging.DEBUG)

STATE_FILE = "/app/pipelines/pipeline_state.json"
SCHEMA_VERSION = 2  # v1 = mean-pooled (broken), v2 = multi-vector MaxSim


class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str = "qdrant"
        QDRANT_PORT: int = 6333
        PDF_DIR: str = "/app/downloads"
        COLLECTION_NAME: str = "target_knowledge"
        TOP_K: int = 8
        SCORE_THRESHOLD: float = 0.0
        OPENROUTER_API_KEY: str = ""
        OPENROUTER_MODEL: str = "qwen/qwen3-vl-30b-a3b-instruct"
        # OPENROUTER_MODEL: str = "google/gemini-2.0-flash-001"
        SHOW_SOURCE_PAGES: bool = True
        IMAGE_SERVER_URL: str = "http://localhost:8081"
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

    def _save_page_image(self, page_img: Image.Image, filename: str, page_num: int) -> str:
        """Save page image to cache dir, return filename."""
        os.makedirs(self.valves.IMAGE_CACHE_DIR, exist_ok=True)
        safe_name = pathlib.Path(filename).stem
        img_filename = f"{safe_name}_p{page_num}.jpg"
        img_path = os.path.join(self.valves.IMAGE_CACHE_DIR, img_filename)
        page_img.save(img_path, format="JPEG", quality=85)
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
        if not os.path.exists(thumb_path):
            src_path = os.path.join(self.valves.IMAGE_CACHE_DIR, img_filename)
            img = Image.open(src_path)
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            img.save(thumb_path, format="JPEG", quality=70)
        return thumb_filename

    # ── model loading ─────────────────────────────────────────────────

    def _load_model(self):
        """Load ColQwen2 + connect Qdrant. Called once at startup via executor."""
        if self._initialized:
            return

        log.info("=== Model loading starting ===")

        self.valves.OPENROUTER_API_KEY = os.getenv(
            "OPENROUTER_API_KEY", self.valves.OPENROUTER_API_KEY
        )
        self.valves.OPENROUTER_MODEL = os.getenv(
            "OPENROUTER_MODEL", self.valves.OPENROUTER_MODEL
        )
        self.valves.COLLECTION_NAME = os.getenv(
            "TARGET_KNOWLEDGE", self.valves.COLLECTION_NAME
        )

        try:
            log.info("Loading ColQwen2 (CPU) …")
            self.model = ColQwen2.from_pretrained(
                "vidore/colqwen2-v1.0",
                torch_dtype=torch.float32,
                device_map="cpu",
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
            log.info("✓ ColQwen2 ready")

            self.qdrant = QdrantClient(
                host=self.valves.QDRANT_HOST, port=self.valves.QDRANT_PORT
            )
            log.info(f"✓ Qdrant connected at {self.valves.QDRANT_HOST}:{self.valves.QDRANT_PORT}")

            self._initialized = True
            log.info("=== Model loading complete ===")

        except Exception as e:
            log.error(f"!!! Model loading failed: {e}", exc_info=True)
            raise

    # ── background indexing ───────────────────────────────────────────

    def _start_background_index(self):
        """Spawn a daemon thread to run indexing, unless one is already running."""
        if self._index_thread and self._index_thread.is_alive():
            log.info("Indexing already in progress — skipping duplicate trigger")
            return
        self._index_thread = threading.Thread(
            target=self._background_index_worker,
            daemon=True,
            name="pdf-indexer",
        )
        self._index_thread.start()
        log.info("Background indexing thread started")

    def _background_index_worker(self):
        with self._index_lock:
            try:
                self._index_local_pdfs()
            except Exception as e:
                log.error(f"Background indexing failed: {e}", exc_info=True)

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
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            log.warning(f"No PDFs found in {pdf_dir}")
            return

        state = self._load_state()

        # ── Schema migration: wipe old mean-pooled index ────────────
        if state.get("schema_version", 1) != SCHEMA_VERSION:
            log.info(f"Schema v{state.get('schema_version',1)} → v{SCHEMA_VERSION}: re-indexing all PDFs")
            try:
                self.qdrant.delete_collection(collection)
            except Exception:
                pass
            state = {"schema_version": SCHEMA_VERSION, "indexed_files": [], "file_progress": {}}
            self._save_state(state)

        indexed = set(state.get("indexed_files", []))
        file_progress = state.get("file_progress", {})

        # Include both unstarted and partially-indexed files
        to_index = [p for p in pdfs if p.name not in indexed]

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

        global_page_id = self.qdrant.get_collection(collection).points_count

        for pdf_file in to_index:
            # Resume from the last completed page, or start from 1
            start_page = file_progress.get(pdf_file.name, 0) + 1
            log.info(f"  Indexing {pdf_file.name} (from page {start_page}) …")
            global_page_id = self._ingest(pdf_file.name, collection, global_page_id, start_page)
            indexed.add(pdf_file.name)
            state["indexed_files"] = list(indexed)
            state["schema_version"] = SCHEMA_VERSION
            state.get("file_progress", {}).pop(pdf_file.name, None)
            self._save_state(state)

        # Clear index_job on completion
        state["index_job"] = {}
        self._save_state(state)
        log.info("✓ All PDFs indexed")

    def _ingest(self, filename: str, collection: str, start_id: int, start_page: int = 1) -> int:
        from pdf2image import pdfinfo_from_path

        pdf_path = pathlib.Path(self.valves.PDF_DIR) / filename
        total_pages = pdfinfo_from_path(str(pdf_path))["Pages"]
        log.info(f"    {filename}: {total_pages} total pages")

        state = self._load_state()
        if "file_progress" not in state:
            state["file_progress"] = {}

        pid = start_id
        for page_num in range(start_page, total_pages + 1):
            # Convert one page at a time to avoid loading the full PDF into memory
            page_img = convert_from_path(
                str(pdf_path), dpi=200, first_page=page_num, last_page=page_num
            )[0].convert("RGB")

            img_filename = self._save_page_image(page_img, filename, page_num)

            batch = self.processor.process_images([page_img])
            with torch.no_grad():
                emb = self.model(**batch)

            multi_vec = emb[0].tolist()
            del page_img, batch, emb  # free memory immediately

            self.qdrant.upsert(collection, [PointStruct(
                id=pid,
                vector=multi_vec,
                payload={
                    "source": filename,
                    "page_number": page_num,
                    "image_filename": img_filename,
                },
            )])

            log.info(f"    page {page_num}/{total_pages}  id={pid}  patches={len(multi_vec)}")
            pid += 1

            # Checkpoint after every page: persist progress + live index_job status
            state["file_progress"][filename] = page_num
            state["index_job"] = {
                "active": True,
                "current_file": filename,
                "current_page": page_num,
                "total_pages": total_pages,
            }
            self._save_state(state)

        return pid

    # ── search with MULTI-VECTOR MaxSim ──────────────────────────────

    def _search(self, query: str, top_k: int = 5):
        batch = self.processor.process_queries([query])
        with torch.no_grad():
            emb = self.model(**batch)

        # ── Full multi-vector query (all query tokens) ──────────────
        q_vecs = emb[0].tolist()   # shape: (num_query_tokens, dim)
        log.info(f"Query encoded → {len(q_vecs)} tokens")

        return self.qdrant.query_points(
            collection_name=self.valves.COLLECTION_NAME,
            query=q_vecs,             # ← MULTI-VECTOR query
            limit=top_k,
            search_params=SearchParams(exact=True),
            with_payload=True,
        ).points

    # ── VLM call via OpenRouter ──────────────────────────────────────

    def _call_vlm(self, query: str, hits) -> tuple:
        import requests as _req

        api_key = self.valves.OPENROUTER_API_KEY
        model = self.valves.OPENROUTER_MODEL
        if not api_key:
            return "Error: OPENROUTER_API_KEY not set.", set()

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
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })

        resp = _req.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful document assistant specializing in technical "
                            "documents such as wiring diagrams, schematics, and manuals. "
                            "You are given page images retrieved from the user's documents, "
                            "each labeled [REF:N]. "
                            "Answer the question using only information visible in these pages. "
                            "You MUST cite every claim inline using [REF:N] "
                            "(e.g. 'Power flows through fuse F53 [REF:2].'). "
                            "When tracing paths, circuits, or flows that span multiple pages, "
                            "explicitly connect the information across pages in sequence to "
                            "build a complete end-to-end answer. "
                            "Only cite pages you actually used. "
                            "If a page is irrelevant, do not cite it."
                        ),
                    },
                    {"role": "user", "content": content_parts},
                ],
                "max_tokens": 2048,
            },
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]

        # Parse which REF indices were cited (1-based → 0-based), tolerant of spacing/case
        cited = {int(m) - 1 for m in re.findall(r'\[REF\s*:\s*(\d+)\]', answer, re.IGNORECASE)}

        # Replace [REF:N] variants with a linked [Page X] pointing to the full image
        for i, hit in enumerate(hits, 1):
            pg = hit.payload.get("page_number", "?")
            img_filename = hit.payload.get("image_filename", "")
            full_url = f"{self.valves.IMAGE_SERVER_URL}/{img_filename}" if img_filename else ""
            replacement = f"**[\\[Page {pg}\\]]({full_url})**" if full_url else f"Page {pg}"
            answer = re.sub(rf'\[REF\s*:\s*{i}\]', replacement, answer, flags=re.IGNORECASE)

        return answer, cited

    # ── main entry ───────────────────────────────────────────────────

    def pipe(self, body: dict, **kwargs) -> str:
        try:
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

            # User status query
            if query.strip().lower() in ("status", "indexing status", "/status"):
                return self._format_index_status()

            hits = self._search(query, top_k=self.valves.TOP_K)

            # Filter low-confidence results
            if self.valves.SCORE_THRESHOLD > 0:
                hits = [h for h in hits if h.score >= self.valves.SCORE_THRESHOLD]

            if not hits:
                return "No relevant pages found in the indexed documents."

            log.info(f"Retrieved {len(hits)} pages")
            for h in hits:
                log.info(
                    f"  → {h.payload.get('source')} p{h.payload.get('page_number')} "
                    f"score={h.score:.4f}"
                )

            answer, cited_indices = self._call_vlm(query, hits)

            # ── Source thumbnails via image-server ────────────────────
            if self.valves.SHOW_SOURCE_PAGES:
                cited_hits = [hits[i] for i in sorted(cited_indices) if i < len(hits)]
                if not cited_hits:
                    cited_hits = hits  # fallback if model didn't use REF format
                answer += "\n\n---\n\n**📄 Source Pages:**\n\n"
                headers = []
                divider = []
                images = []
                for hit in cited_hits:
                    img_filename = hit.payload.get("image_filename", "")
                    source = hit.payload.get("source", "unknown")
                    page = hit.payload.get("page_number", "?")
                    score = hit.score

                    if img_filename:
                        thumb_filename = self._make_thumbnail_file(img_filename)
                        full_url = f"{self.valves.IMAGE_SERVER_URL}/{img_filename}"
                        thumb_url = f"{self.valves.IMAGE_SERVER_URL}/{thumb_filename}"
                        headers.append(f"p{page} · {source} ({score:.2f})")
                        divider.append(":---:")
                        images.append(f"[![p{page}]({thumb_url})]({full_url})")
                if images:
                    answer += (
                        "| " + " | ".join(headers) + " |\n"
                        "| " + " | ".join(divider) + " |\n"
                        "| " + " | ".join(images)  + " |\n\n"
                    )

            return answer

        except Exception as e:
            log.error(f"pipe error: {e}", exc_info=True)
            return f"Error: {e}"










# """
# title: ColQwen2 Visual RAG (CPU)
# author: adapted
# version: 7.0
# license: MIT
# description: Visual RAG pipeline using ColQwen2 + Qdrant + OpenRouter VLM with source thumbnails.
# """

# import os, json, base64, io, logging, pathlib, time, hashlib
# from typing import List, Optional

# import torch
# from PIL import Image
# from pydantic import BaseModel
# from pdf2image import convert_from_path
# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     Distance, VectorParams, PointStruct, SearchParams
# )
# from colpali_engine.models import ColQwen2, ColQwen2Processor

# log = logging.getLogger("colpali-pipeline")
# log.setLevel(logging.DEBUG)

# TARGET_KNOWLEDGE = "target_knowledge"
# STATE_FILE = "/app/pipelines/pipeline_state.json"
# COLLECTION = TARGET_KNOWLEDGE
# THUMB_MAX_WIDTH = 400


# class Pipeline:
#     class Valves(BaseModel):
#         QDRANT_HOST: str = "qdrant"
#         QDRANT_PORT: int = 6333
#         PDF_DIR: str = "/app/downloads"
#         TOP_K: int = 3
#         OPENROUTER_API_KEY: str = ""
#         OPENROUTER_MODEL: str = "qwen/qwen3-vl-30b-a3b-instruct"
#         SHOW_SOURCE_PAGES: bool = True
#         IMAGE_SERVER_URL: str = "http://localhost:8081"
#         IMAGE_CACHE_DIR: str = "/app/pipelines/cache/images"

#     def __init__(self):
#         self.name = "ColQwen2 Visual RAG"
#         self.valves = self.Valves()
#         self.model = None
#         self.processor = None
#         self.qdrant = None
#         self._initialized = False

#     async def on_startup(self):
#         log.info("on_startup called – deferring heavy init to first query")

#     async def on_shutdown(self):
#         log.info("Pipeline shutdown")

#     # ── helpers ──────────────────────────────────────────────────────

#     def _load_state(self) -> dict:
#         if os.path.exists(STATE_FILE):
#             with open(STATE_FILE) as f:
#                 return json.load(f)
#         return {}

#     def _save_state(self, state: dict):
#         with open(STATE_FILE, "w") as f:
#             json.dump(state, f)

#     # Small thumbnail for inline preview row
#     def _make_thumbnail(self, img_b64: str, max_width: int = THUMB_MAX_WIDTH) -> str:
#         img_bytes = base64.b64decode(img_b64)
#         img = Image.open(io.BytesIO(img_bytes))
#         ratio = max_width / img.width
#         new_height = int(img.height * ratio)
#         img = img.resize((max_width, new_height), Image.LANCZOS)
#         buf = io.BytesIO()
#         img.save(buf, format="JPEG", quality=70)
#         return base64.b64encode(buf.getvalue()).decode("utf-8")
    
#     # ── Readable but lightweight click-to-view image ─────────────────
#     # Compressed enough to fit in Open WebUI's response limit.
#     # 800px wide, quality 50 → ~40-60KB per page vs ~1MB full-res.
#     def _make_readable(self, img_b64: str, max_width: int = 800) -> str:
#         img_bytes = base64.b64decode(img_b64)
#         img = Image.open(io.BytesIO(img_bytes))
#         if img.width <= max_width:
#             ratio = 1
#         else:
#             ratio = max_width / img.width
#         new_width = int(img.width * ratio)
#         new_height = int(img.height * ratio)
#         img = img.resize((new_width, new_height), Image.LANCZOS)
#         buf = io.BytesIO()
#         img.save(buf, format="JPEG", quality=50)
#         return base64.b64encode(buf.getvalue()).decode("utf-8")

#     # ── full init ────────────────────────────────────────────────────

#     def _full_init(self):
#         if self._initialized:
#             return

#         log.info("=== Full initialization starting ===")

#         self.valves.OPENROUTER_API_KEY = os.getenv(
#             "OPENROUTER_API_KEY", self.valves.OPENROUTER_API_KEY
#         )
#         self.valves.OPENROUTER_MODEL = os.getenv(
#             "OPENROUTER_MODEL", self.valves.OPENROUTER_MODEL
#         )

#         try:
#             log.info("Loading ColQwen2 (CPU) …")
#             self.model = ColQwen2.from_pretrained(
#                 "vidore/colqwen2-v1.0",
#                 torch_dtype=torch.float32,
#                 device_map="cpu",
#             ).eval()
#             self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
#             log.info("✓ ColQwen2 ready")

#             self.qdrant = QdrantClient(
#                 host=self.valves.QDRANT_HOST, port=self.valves.QDRANT_PORT
#             )
#             log.info(f"✓ Qdrant connected at {self.valves.QDRANT_HOST}:{self.valves.QDRANT_PORT}")

#             self._index_local_pdfs()
#             self._initialized = True
#             log.info("=== Full initialization complete ===")

#         except Exception as e:
#             log.error(f"!!! Initialization failed: {e}", exc_info=True)
#             raise

#     # ── indexing ─────────────────────────────────────────────────────

#     def _index_local_pdfs(self):
#         pdf_dir = pathlib.Path(self.valves.PDF_DIR)
#         pdfs = sorted(pdf_dir.glob("*.pdf"))
#         if not pdfs:
#             log.warning(f"No PDFs found in {pdf_dir}")
#             return

#         state = self._load_state()
#         indexed = set(state.get("indexed_files", []))
#         to_index = [p for p in pdfs if p.name not in indexed]

#         if not to_index:
#             log.info("All PDFs already indexed – skipping")
#             return

#         log.info(f"Found {len(to_index)} PDFs to index")

#         try:
#             self.qdrant.get_collection(COLLECTION)
#         except Exception:
#             dummy = self.processor.process_images([Image.new("RGB", (32, 32))])
#             with torch.no_grad():
#                 dim = self.model(**dummy).shape[-1]
#             self.qdrant.create_collection(
#                 COLLECTION,
#                 vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
#             )
#             log.info(f"Created Qdrant collection '{COLLECTION}' dim={dim}")

#         global_page_id = self.qdrant.get_collection(COLLECTION).points_count

#         for pdf_file in to_index:
#             log.info(f"  Indexing {pdf_file.name} …")
#             global_page_id = self._ingest(pdf_file.name, COLLECTION, global_page_id)
#             indexed.add(pdf_file.name)
#             state["indexed_files"] = list(indexed)
#             self._save_state(state)

#         log.info("✓ All PDFs indexed")

#     def _ingest(self, filename: str, collection: str, start_id: int) -> int:
#         pdf_path = pathlib.Path(self.valves.PDF_DIR) / filename
#         pages = convert_from_path(str(pdf_path), dpi=200)

#         points = []
#         for i, page_img in enumerate(pages):
#             page_img = page_img.convert("RGB")
#             batch = self.processor.process_images([page_img])
#             with torch.no_grad():
#                 emb = self.model(**batch)
#             mean_vec = emb[0].mean(dim=0).tolist()

#             buf = io.BytesIO()
#             page_img.save(buf, format="JPEG", quality=85)
#             img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

#             pid = start_id + i
#             points.append(
#                 PointStruct(
#                     id=pid,
#                     vector=mean_vec,
#                     payload={
#                         "source": filename,
#                         "page_number": i + 1,
#                         "image_b64": img_b64,
#                     },
#                 )
#             )
#             log.info(f"    page {i + 1}/{len(pages)}  id={pid}")

#         for point in points:
#             self.qdrant.upsert(collection, [point])
#         return start_id + len(pages)

#     # ── search (uses query_points for qdrant-client >= 1.12) ─────────

#     def _search(self, query: str, top_k: int = 3):
#         batch = self.processor.process_queries([query])
#         with torch.no_grad():
#             emb = self.model(**batch)
#         q_vec = emb[0].mean(dim=0).tolist()
#         return self.qdrant.query_points(
#             collection_name=COLLECTION,
#             query=q_vec,
#             limit=top_k,
#             search_params=SearchParams(exact=True),
#             with_payload=True,
#         ).points

#     # ── VLM call via OpenRouter ──────────────────────────────────────

#     def _call_vlm(self, query: str, hits) -> str:
#         import requests as _req

#         api_key = self.valves.OPENROUTER_API_KEY
#         model = self.valves.OPENROUTER_MODEL

#         if not api_key:
#             return "Error: OPENROUTER_API_KEY not set."

#         content_parts = [{"type": "text", "text": query}]
#         for hit in hits:
#             img_b64 = hit.payload.get("image_b64", "")
#             src = hit.payload.get("source", "?")
#             pg = hit.payload.get("page_number", "?")
#             if img_b64:
#                 content_parts.append({"type": "text", "text": f"[Page {pg} from {src}]"})
#                 content_parts.append({
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
#                 })

#         resp = _req.post(
#             "https://openrouter.ai/api/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json",
#             },
#             json={
#                 "model": model,
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a helpful assistant. Answer the user's question "
#                             "based on the document page images provided. Be specific "
#                             "and cite page numbers when possible."
#                         ),
#                     },
#                     {"role": "user", "content": content_parts},
#                 ],
#                 "max_tokens": 2048,
#             },
#             timeout=120,
#         )
#         resp.raise_for_status()
#         result = resp.json()
#         return result["choices"][0]["message"]["content"]

#     # ── main entry ───────────────────────────────────────────────────

#     def pipe(self, body: dict, **kwargs) -> str:
#         try:
#             self._full_init()

#             messages = body.get("messages", [])
#             query = messages[-1]["content"] if messages else ""
#             if not query:
#                 return "Please ask a question about your documents."

#             log.info(f"Query: {query}")

#             top_k = self.valves.TOP_K
#             hits = self._search(query, top_k=top_k)
#             if not hits:
#                 return "No relevant pages found."

#             log.info(f"Retrieved {len(hits)} pages")
#             for h in hits:
#                 log.info(
#                     f"  → {h.payload.get('source')} p{h.payload.get('page_number')} "
#                     f"score={h.score:.4f}"
#                 )

#             answer = self._call_vlm(query, hits)

#             # ── Source page thumbnails with click-to-view ─────────────
#             # Thumbnails are tiny base64 inline. Full images are served
#             # as files via image-server to avoid "Chunk too big".
#             if self.valves.SHOW_SOURCE_PAGES:
#                 os.makedirs(self.valves.IMAGE_CACHE_DIR, exist_ok=True)
#                 answer += '\n\n---\n\n**📄 Source Pages:**\n\n'
#                 for i, hit in enumerate(hits):
#                     img_b64 = hit.payload.get("image_b64", "")
#                     source = hit.payload.get("source", "unknown")
#                     page = hit.payload.get("page_number", "?")
#                     score = hit.score

#                     if img_b64:
#                         img_hash = hashlib.md5(img_b64[:1000].encode()).hexdigest()
#                         base_name = f"{source}_p{page}_{img_hash}"

#                         # Save full-res
#                         full_name = f"{base_name}_full.jpg"
#                         full_path = os.path.join(self.valves.IMAGE_CACHE_DIR, full_name)
#                         if not os.path.exists(full_path):
#                             with open(full_path, "wb") as f:
#                                 f.write(base64.b64decode(img_b64))

#                         # Save thumbnail
#                         thumb_name = f"{base_name}_thumb.jpg"
#                         thumb_path = os.path.join(self.valves.IMAGE_CACHE_DIR, thumb_name)
#                         if not os.path.exists(thumb_path):
#                             thumb_b64 = self._make_thumbnail(img_b64, max_width=150)
#                             with open(thumb_path, "wb") as f:
#                                 f.write(base64.b64decode(thumb_b64))

#                         full_url = f"{self.valves.IMAGE_SERVER_URL}/{full_name}"
#                         thumb_url = f"{self.valves.IMAGE_SERVER_URL}/{thumb_name}"

#                         # Pure URL markdown - guaranteed to render
#                         answer += (
#                             f'[![Page {page} - {source}]'
#                             f'({thumb_url})]'
#                             f'({full_url})\n\n'
#                         )
#             return answer

#         except Exception as e:
#             log.error(f"pipe error: {e}", exc_info=True)
#             return f"Error: {e}"