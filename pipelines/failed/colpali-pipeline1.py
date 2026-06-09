"""
title: ColQwen2 Visual RAG (CPU)
author: adapted
version: 7.0
license: MIT
description: Visual RAG pipeline using ColQwen2 + Qdrant + OpenRouter VLM with source thumbnails.
"""

import os, json, base64, io, logging, pathlib, time
from typing import List, Optional

import torch
from PIL import Image
from pydantic import BaseModel
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SearchParams
)
from colpali_engine.models import ColQwen2, ColQwen2Processor

log = logging.getLogger("colpali-pipeline")
log.setLevel(logging.DEBUG)

TARGET_KNOWLEDGE = "target_knowledge"
STATE_FILE = "/app/pipeline_state.json"
COLLECTION = TARGET_KNOWLEDGE
THUMB_MAX_WIDTH = 400


class Pipeline:
    class Valves(BaseModel):
        QDRANT_HOST: str = "qdrant"
        QDRANT_PORT: int = 6333
        PDF_DIR: str = "/app/downloads"
        TOP_K: int = 3
        OPENROUTER_API_KEY: str = ""
        OPENROUTER_MODEL: str = "qwen/qwen3-vl-30b-a3b-instruct"
        SHOW_SOURCE_PAGES: bool = True

    def __init__(self):
        self.name = "ColQwen2 Visual RAG"
        self.valves = self.Valves()
        self.model = None
        self.processor = None
        self.qdrant = None
        self._initialized = False

    async def on_startup(self):
        log.info("on_startup called – deferring heavy init to first query")

    async def on_shutdown(self):
        log.info("Pipeline shutdown")

    # ── helpers ──────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                return json.load(f)
        return {}

    def _save_state(self, state: dict):
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)

    def _make_thumbnail(self, img_b64: str, max_width: int = THUMB_MAX_WIDTH) -> str:
        """Resize a base64 image to a thumbnail, return as base64 JPEG."""
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── full init ────────────────────────────────────────────────────

    def _full_init(self):
        if self._initialized:
            return

        log.info("=== Full initialization starting ===")

        # Override valves from env vars
        self.valves.OPENROUTER_API_KEY = os.getenv(
            "OPENROUTER_API_KEY", self.valves.OPENROUTER_API_KEY
        )
        self.valves.OPENROUTER_MODEL = os.getenv(
            "OPENROUTER_MODEL", self.valves.OPENROUTER_MODEL
        )

        try:
            # ColQwen2
            log.info("Loading ColQwen2 (CPU) …")
            self.model = ColQwen2.from_pretrained(
                "vidore/colqwen2-v1.0",
                torch_dtype=torch.float32,
                device_map="cpu",
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
            log.info("✓ ColQwen2 ready")

            # Qdrant
            self.qdrant = QdrantClient(
                host=self.valves.QDRANT_HOST, port=self.valves.QDRANT_PORT
            )
            log.info(f"✓ Qdrant connected at {self.valves.QDRANT_HOST}:{self.valves.QDRANT_PORT}")

            # Index PDFs
            self._index_local_pdfs()
            self._initialized = True
            log.info("=== Full initialization complete ===")

        except Exception as e:
            log.error(f"!!! Initialization failed: {e}", exc_info=True)
            raise

    # ── indexing ─────────────────────────────────────────────────────

    def _index_local_pdfs(self):
        pdf_dir = pathlib.Path(self.valves.PDF_DIR)
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            log.warning(f"No PDFs found in {pdf_dir}")
            return

        state = self._load_state()
        indexed = set(state.get("indexed_files", []))
        to_index = [p for p in pdfs if p.name not in indexed]

        if not to_index:
            log.info("All PDFs already indexed – skipping")
            return

        log.info(f"Found {len(to_index)} PDFs to index")

        # Ensure collection
        try:
            self.qdrant.get_collection(COLLECTION)
        except Exception:
            dummy = self.processor.process_images([Image.new("RGB", (32, 32))])
            with torch.no_grad():
                dim = self.model(**dummy).shape[-1]
            self.qdrant.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info(f"Created Qdrant collection '{COLLECTION}' dim={dim}")

        global_page_id = self.qdrant.get_collection(COLLECTION).points_count

        for pdf_file in to_index:
            log.info(f"  Indexing {pdf_file.name} …")
            global_page_id = self._ingest(pdf_file.name, COLLECTION, global_page_id)
            indexed.add(pdf_file.name)
            state["indexed_files"] = list(indexed)
            self._save_state(state)

        log.info("✓ All PDFs indexed")

    def _ingest(self, filename: str, collection: str, start_id: int) -> int:
        pdf_path = pathlib.Path(self.valves.PDF_DIR) / filename
        pages = convert_from_path(str(pdf_path), dpi=200)

        points = []
        for i, page_img in enumerate(pages):
            page_img = page_img.convert("RGB")
            batch = self.processor.process_images([page_img])
            with torch.no_grad():
                emb = self.model(**batch)
            mean_vec = emb[0].mean(dim=0).tolist()

            buf = io.BytesIO()
            page_img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            pid = start_id + i
            points.append(
                PointStruct(
                    id=pid,
                    vector=mean_vec,
                    payload={
                        "source": filename,
                        "page_number": i + 1,
                        "image_b64": img_b64,
                    },
                )
            )
            log.info(f"    page {i + 1}/{len(pages)}  id={pid}")

        self.qdrant.upsert(collection, points)
        return start_id + len(pages)

    # ── search ───────────────────────────────────────────────────────

    def _search(self, query: str, top_k: int = 3):
        batch = self.processor.process_queries([query])
        with torch.no_grad():
            emb = self.model(**batch)
        q_vec = emb[0].mean(dim=0).tolist()
        return self.qdrant.search(
            COLLECTION,
            query_vector=q_vec,
            limit=top_k,
            search_params=SearchParams(exact=True),
            with_payload=True,
        )

    # ── VLM call via OpenRouter ──────────────────────────────────────

    def _call_vlm(self, query: str, hits) -> str:
        import requests as _req

        api_key = self.valves.OPENROUTER_API_KEY
        model = self.valves.OPENROUTER_MODEL

        if not api_key:
            return "Error: OPENROUTER_API_KEY not set."

        content_parts = [{"type": "text", "text": query}]
        for hit in hits:
            img_b64 = hit.payload.get("image_b64", "")
            src = hit.payload.get("source", "?")
            pg = hit.payload.get("page_number", "?")
            if img_b64:
                content_parts.append({"type": "text", "text": f"[Page {pg} from {src}]"})
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
                            "You are a helpful assistant. Answer the user's question "
                            "based on the document page images provided. Be specific "
                            "and cite page numbers when possible."
                        ),
                    },
                    {"role": "user", "content": content_parts},
                ],
                "max_tokens": 2048,
            },
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]

    # ── main entry ───────────────────────────────────────────────────

    def pipe(self, body: dict, **kwargs) -> str:
        try:
            self._full_init()

            messages = body.get("messages", [])
            query = messages[-1]["content"] if messages else ""
            if not query:
                return "Please ask a question about your documents."

            log.info(f"Query: {query}")

            # Retrieve
            top_k = self.valves.TOP_K
            hits = self._search(query, top_k=top_k)
            if not hits:
                return "No relevant pages found."

            log.info(f"Retrieved {len(hits)} pages")
            for h in hits:
                log.info(
                    f"  → {h.payload.get('source')} p{h.payload.get('page_number')} "
                    f"score={h.score:.4f}"
                )

            # Generate answer via OpenRouter VLM
            answer = self._call_vlm(query, hits)

            # Append source page thumbnails
            if self.valves.SHOW_SOURCE_PAGES:
                answer += "\n\n---\n\n**📄 Retrieved Source Pages:**\n\n"
                for i, hit in enumerate(hits):
                    img_b64 = hit.payload.get("image_b64", "")
                    source = hit.payload.get("source", "unknown")
                    page = hit.payload.get("page_number", "?")
                    score = hit.score

                    if img_b64:
                        thumb_b64 = self._make_thumbnail(img_b64)
                        answer += (
                            f"**Page {page}** from `{source}` "
                            f"*(relevance: {score:.2f})*\n\n"
                        )
                        answer += f"![Page {page}](data:image/jpeg;base64,{thumb_b64})\n\n"

            return answer

        except Exception as e:
            log.error(f"pipe error: {e}", exc_info=True)
            return f"Error: {e}"