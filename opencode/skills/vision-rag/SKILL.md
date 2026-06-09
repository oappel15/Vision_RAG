---
name: vision-rag
description: Use when the user wants to search PDF documents visually (diagrams, schematics, tables, images), check if a PDF is indexed in Vision RAG, upload a PDF for indexing, or query indexed documents. Also use when the user references a PDF file and needs information extracted from it — especially visual content that cannot be read as plain text. Trigger keywords include "search PDF", "index this PDF", "is this indexed", "look up in the schematic", "what does the diagram show", "Vision RAG", "label this PDF".
---

# Vision RAG — Visual Document Search

This skill connects OpenCode to the Vision RAG system, which indexes PDF
documents as page images and searches them using multi-vector visual
embeddings (ColQwen2) + Vision LLM analysis. It can extract information
from **images, diagrams, tables, schematics, wiring diagrams, and any
visual content** in PDFs — not just text.

## Architecture

- **Qdrant** vector database stores multi-vector page embeddings
- **ColQwen2** encodes each PDF page as a grid of patch embeddings
- **Vision LLM** (Gemini/GPT/Claude) reads matched page images and answers
- **pdf-ingest** (port 8082) handles file upload, labels, status
- **pipelines** (port 9099) handles search and indexing

## Available MCP Tools

You have these tools available via the `vision-rag` MCP server:

| Tool | Purpose |
|------|---------|
| `vision_rag_status` | Check indexing status, list indexed files and their labels |
| `vision_rag_labels` | List all available labels for search filtering |
| `vision_rag_is_indexed` | Check if a specific PDF is already indexed |
| `vision_rag_upload` | Upload a PDF to the index with optional labels |
| `vision_rag_search` | Search documents with optional label filters |
| `vision_rag_set_labels` | Add/update labels on an indexed document |

## Standard Workflow

### When the user wants information from a PDF:

1. **Check if indexed**: Call `vision_rag_is_indexed` with the PDF filename.
2. **If not indexed**: Call `vision_rag_upload` with the file path and
   appropriate labels. Wait for indexing (check with `vision_rag_status`).
3. **Search**: Call `vision_rag_search` with the user's question and
   the document's label to scope results to that specific file.
4. **Return the result**: The search response includes the VLM's answer
   with citations to specific pages. Embed this in your response.

### When the user asks about a PDF in their project tree:

```
User: "What sensors are shown in the Toyota wiring diagram?"

1. Check: vision_rag_is_indexed("Toyota.pdf")
   → YES, indexed with labels: [Toyota]

2. Search: vision_rag_search(
     query="list all sensors in the wiring diagram",
     labels=["Toyota"]
   )
   → Returns VLM analysis of matched pages with sensor list

3. Return the answer to the user with the extracted data.
```

### When the user wants to index a new PDF:

```
User: "Index the schematic at ./docs/board_v2.pdf"

1. Upload: vision_rag_upload(
     filepath="/absolute/path/to/docs/board_v2.pdf",
     labels=["board_v2", "schematic", "project-name"]
   )

2. Check progress: vision_rag_status()
   → Shows indexing progress

3. Confirm to user when done.
```

## Label Best Practices

Labels enable scoped searches. Always apply meaningful labels:

- **Filename stem** is automatic (e.g., `Toyota.pdf` gets label `Toyota`)
- **Project name**: label PDFs with the project they belong to
- **Category**: `schematic`, `datasheet`, `manual`, `report`
- **Source**: `Confluence` (auto-applied for Confluence pages)

When searching, always specify labels to scope results:
- `labels=["Toyota"]` — only Toyota docs
- `labels=["schematic"]` — only schematics
- `labels=["Toyota", "schematic"]` — must match both (AND logic)
- No labels = search everything (YOLO mode)

## Search Tips

- The search works on **visual content**: diagrams, tables, images, graphs
- Each PDF page is a separate searchable unit
- The VLM reads the actual page image — it can see layout, colors, symbols
- For dense schematics, ask specific questions rather than broad ones
- Label filtering is case-insensitive

## Error Handling

- If the service is unreachable, Vision RAG Docker containers may not be running
- If indexing seems stuck, check `vision_rag_status` — large pages take time
- If search returns empty, try without label filters to verify documents exist
- The pipeline auto-indexes on startup, so newly uploaded files will be processed
