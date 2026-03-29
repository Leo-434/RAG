# GraphRAG — Knowledge Graph-Enhanced AI Q&A System

<div align="center">

![GraphRAG](https://img.shields.io/badge/GraphRAG-Knowledge%20Base-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI Q&A system powered by Knowledge Graph + Vector Search, featuring multi-turn conversations, hybrid retrieval, and interactive graph visualization.**

[中文](./README.md) · English

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
- [License](#license)

---

## Overview

GraphRAG is an AI-powered question-answering system that deeply integrates **vector semantic retrieval** with a **Neo4j knowledge graph**. It automatically parses PDF documents, extracts entities and relationships to build a knowledge graph, and answers natural language questions via three retrieval strategies. All conversation history and document metadata are persisted in SQLite, surviving backend restarts.

```
PDF Document → MinerU Parse → Text Chunking ─┬─ ChromaDB Vector Store
                                              └─ Entity Extraction → Neo4j Knowledge Graph

User Question → Hybrid Retrieval (Vector + Graph RRF) → LangChain Agent → Streaming Answer
```

---

## Key Features

### 🔍 Three Retrieval Modes

| Mode | Description |
|------|-------------|
| **Hybrid** | Vector semantic search + Graph fulltext search, merged via **Reciprocal Rank Fusion (RRF)**. Best overall accuracy. |
| **Vector Only** | ChromaDB similarity search only. Best for open-ended conceptual questions. |
| **Graph Only** | Neo4j fulltext index only. Best for precise entity/relationship lookups. |

### 💬 Multi-Turn Conversations
- Conversation history persisted to SQLite — restored automatically on restart
- Sidebar shows all past conversations with timestamps; click to resume
- Auto-titles generated from the first user message
- Full conversation history passed to the LLM agent as context for follow-up questions

### 📄 Document Management
- PDF upload with async background processing — UI stays responsive
- Real-time status tracking: `uploading → ingesting → ready / failed`
- Per-document stats: chunk count, entity count, relationship count
- Cascading delete: removes Neo4j nodes, ChromaDB embeddings, disk file, and metadata atomically

### 🕸 Knowledge Graph Visualization
- Filter entities by type: PERSON, ORGANIZATION, PRODUCT, CONCEPT
- Interactive canvas: drag, zoom, fullscreen mode
- Double-click any node to expand its 1-hop neighbor subgraph
- Fulltext keyword search to locate entities instantly
- Filter graph view by specific documents

### ⚡ Streaming Answers (SSE)
- Token-by-token streaming via Server-Sent Events
- Stop generation at any time
- Source citations shown after answer completes (document chunks / graph entities)

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────┐
│          Frontend (React 18 + Vite)          │
│   ChatPage   DocumentsPage   GraphPage       │
│   Zustand State Management + SSE Streaming   │
└─────────────────┬───────────────────────────┘
                  │ REST API / SSE
┌─────────────────▼───────────────────────────┐
│         Backend (FastAPI + Python 3.10+)     │
│                                              │
│  /api/query      → AnswerService             │
│  /api/documents  → IngestionService          │
│  /api/graph      → GraphService              │
│  /api/conversations → ConversationService    │
│                                              │
│  HybridRetrievalService (RRF Fusion)         │
│  LangChain Agents (create_agent)             │
└──────┬──────────────┬────────────┬───────────┘
       │              │            │
   ┌───▼───┐    ┌─────▼────┐  ┌───▼─────┐
   │ Neo4j │    │ ChromaDB │  │ SQLite  │
   │ Graph │    │ Vectors  │  │Metadata │
   └───────┘    └──────────┘  └─────────┘
```

### Tech Stack

**Backend**

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI + Uvicorn | 0.115+ |
| LLM | DashScope (Qwen3-Max) | — |
| Embedding Model | DashScope text-embedding-v4 | — |
| Agent Framework | LangChain + LangGraph | 1.2+ / 1.0+ |
| Graph Database | Neo4j | 5.x |
| Vector Database | ChromaDB | 0.5+ |
| Conversation Store | SQLite (aiosqlite) | — |
| PDF Parsing | MinerU API (VLM mode) | — |
| Data Validation | Pydantic v2 | 2.0+ |
| Logging | structlog | 24.0+ |

**Frontend**

| Component | Technology | Version |
|-----------|-----------|---------|
| UI Framework | React | 18.3.1 |
| Routing | React Router | 7.13.0 |
| State Management | Zustand | 5.0.12 |
| UI Components | Radix UI + Tailwind CSS | 1.x / 4.x |
| Build Tool | Vite | 6.3.5 |
| Notifications | Sonner | 2.0.3 |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- A running Neo4j instance (local or cloud)
- [DashScope API Key](https://dashscope.aliyuncs.com) (Alibaba Cloud — provides Qwen LLM + embeddings)
- [MinerU API Key](https://mineru.net) (PDF parsing service)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/GraphRAG.git
cd GraphRAG
```

### 2. Start the Backend

```bash
cd backend

# Install Python dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys and database credentials

# Start the development server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

### 3. Start the Frontend

```bash
cd frontend

# Install Node dependencies
npm install

# Start the dev server
npm run dev
```

Access the UI at: **http://localhost:5173**

### 4. Workflow

1. Open http://localhost:5173, go to the **Documents** page
2. Upload a PDF — wait for its status to show `ready` (a few minutes depending on document size)
3. Switch to the **Chat** page and start asking questions
4. Explore the extracted entities and relationships on the **Knowledge Graph** page

---

## Configuration

Create a `.env` file inside `backend/` using `.env.example` as a template:

```bash
# ── LLM & Embeddings (DashScope / Alibaba Cloud) ──
DASHSCOPE_API_KEY=sk-your-api-key-here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-max-2026-01-23
EMBEDDING_MODEL=text-embedding-v4

# ── PDF Parsing (MinerU) ───────────────────────────
MINERU_API_KEY=your-mineru-api-key-here
MINERU_BASE_URL=https://mineru.net/api/v4
MINERU_MODEL_VERSION=vlm

# ── Graph Database (Neo4j) ─────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# ── Vector Database (ChromaDB) ─────────────────────
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION=document_chunks

# ── Ingestion Parameters ───────────────────────────
CHUNK_SIZE=400
CHUNK_OVERLAP=60
EMBED_BATCH_SIZE=8
LANGEXTRACT_MAX_CHAR_BUFFER=3000

# ── Retrieval Parameters ───────────────────────────
DEFAULT_TOP_K=5
RRF_K=60                    # Reciprocal Rank Fusion k parameter

# ── File Upload ────────────────────────────────────
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_MB=50

# ── CORS ───────────────────────────────────────────
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

---

## API Reference

Full interactive docs available at `http://localhost:8000/docs`

### Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check (Neo4j + ChromaDB status) |
| `POST` | `/api/documents/upload` | Upload PDF; triggers async ingestion pipeline |
| `GET` | `/api/documents` | List all documents with ingestion status |
| `GET` | `/api/documents/{doc_id}` | Get metadata for a single document |
| `DELETE` | `/api/documents/{doc_id}` | Delete document and all associated data |
| `POST` | `/api/query` | Non-streaming Q&A (returns complete answer) |
| `POST` | `/api/query/stream` | SSE streaming Q&A (token-by-token output) |
| `GET` | `/api/graph/stats` | Graph statistics (node/relation counts by type) |
| `GET` | `/api/graph/entities` | Paginated entity list with type/doc filters |
| `GET` | `/api/graph/entities/{name}/neighbors` | 1-hop neighbor subgraph for an entity |
| `GET` | `/api/graph/relationships` | Paginated relationship list |
| `POST` | `/api/graph/search` | Fulltext keyword search on entities |
| `GET` | `/api/conversations` | List all conversation history |
| `POST` | `/api/conversations` | Create a new conversation |
| `GET` | `/api/conversations/{conv_id}` | Get conversation with all messages |
| `DELETE` | `/api/conversations/{conv_id}` | Permanently delete conversation and messages |

### Streaming Q&A Example

**Request:**
```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who are the founders of OpenAI?",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "conversation_id": "optional-conv-id"
  }'
```

**SSE Response:**
```
event: token
data: {"content": "The founders of OpenAI include"}

event: token
data: {"content": " Sam Altman, Greg Brockman, Ilya Sutskever..."}

event: sources
data: {"sources": [{"source_type": "vector", "content": "...", "filename": "openai.pdf"}, ...]}

event: done
data: {}
```

**QueryRequest Schema:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | `string` | required | The user's question |
| `retrieval_mode` | `hybrid` \| `vector_only` \| `graph_only` | `hybrid` | Retrieval strategy |
| `top_k` | `int` (1–20) | `5` | Number of sources to retrieve |
| `conversation_id` | `string` | `null` | Pass to maintain conversation context |

---

## Project Structure

```
GraphRAG/
├── backend/                              # FastAPI backend
│   ├── app/
│   │   ├── main.py                       # App entry point + lifespan management
│   │   ├── config.py                     # Pydantic settings (reads .env)
│   │   ├── dependencies.py               # FastAPI dependency injectors
│   │   ├── models/                       # Pydantic request/response schemas
│   │   │   ├── document.py
│   │   │   ├── query.py
│   │   │   ├── graph.py
│   │   │   └── conversation.py
│   │   ├── routers/                      # REST API route handlers
│   │   │   ├── health.py
│   │   │   ├── documents.py
│   │   │   ├── query.py
│   │   │   ├── graph.py
│   │   │   └── conversations.py
│   │   ├── services/                     # Business logic
│   │   │   ├── ingestion_service.py      # Document processing pipeline
│   │   │   ├── vector_service.py         # ChromaDB read/write
│   │   │   ├── graph_service.py          # Neo4j read/write
│   │   │   ├── answer_service.py         # LangChain Agent Q&A + streaming
│   │   │   ├── hybrid_retrieval_service.py  # RRF fusion retrieval
│   │   │   ├── conversation_service.py   # Conversation history (SQLite async)
│   │   │   ├── document_registry_service.py # Document metadata (SQLite sync)
│   │   │   └── entity_extraction_service.py # LLM-based entity extraction
│   │   └── utils/
│   │       ├── logger.py                 # Structured logging (structlog)
│   │       ├── rrf.py                    # Reciprocal Rank Fusion algorithm
│   │       └── mineru_parser.py          # MinerU API client wrapper
│   ├── .env.example                      # Configuration template (no secrets)
│   ├── pyproject.toml                    # Python project metadata + deps
│   └── uv.lock                           # Locked dependency versions
│
├── frontend/                             # React frontend
│   ├── src/app/
│   │   ├── App.tsx                       # Root component
│   │   ├── routes.ts                     # React Router v7 route config
│   │   ├── api.ts                        # Typed API client layer
│   │   ├── store.ts                      # Zustand global state
│   │   ├── mock-data.ts                  # Mock data for offline development
│   │   └── components/
│   │       ├── Layout.tsx                # Main layout (sidebar + content)
│   │       ├── Sidebar.tsx               # Conversation list + navigation
│   │       ├── ChatPage.tsx              # Chat interface with streaming
│   │       ├── DocumentsPage.tsx         # Document management UI
│   │       ├── GraphPage.tsx             # Interactive graph visualization
│   │       ├── SimpleMarkdown.tsx        # Lightweight Markdown renderer
│   │       └── ui/                       # 40+ Radix UI base components
│   └── package.json
│
├── .gitignore
├── README.md                             # Chinese documentation
└── README_EN.md                          # English documentation (this file)
```

---

## Data Flow

### Document Ingestion Pipeline

```
① POST /api/documents/upload
   └─ Save file to disk + create SQLite record (status=uploading)

② Background task: run_pipeline()
   ├─ MinerUParser.parse(pdf_file) → Markdown text
   ├─ RecursiveCharacterTextSplitter → list of chunks
   ├─ [Parallel execution]
   │   ├─ VectorService.embed_and_store(chunks) → ChromaDB
   │   └─ EntityExtractionService.extract(markdown)
   │       └─ GraphService.write_graph_data(entities) → Neo4j
   └─ SQLite: update status=ready, record chunk/entity/relation counts

③ GET /api/documents/{doc_id}  ← Frontend polls every 5 seconds
```

### Q&A Flow

```
① POST /api/query/stream
   {question, retrieval_mode, conversation_id, top_k}

② Load conversation history from SQLite (if conversation_id provided)

③ HybridRetrievalService.retrieve(question, mode)
   ├─ hybrid:      Vector search + Graph fulltext → RRF merge
   ├─ vector_only: VectorService.similarity_search()
   └─ graph_only:  GraphService.fulltext_search()

④ AnswerService.stream_answer()
   └─ LangChain Agent (create_agent)
       ├─ Tool calls: semantic_search / entity_graph_search / ...
       └─ Token-by-token output → SSE event: token

⑤ Emit SSE event: sources
   Persist messages to SQLite
   Emit SSE event: done
```

### Data Storage Layout

| Store | What it Contains | Location |
|-------|-----------------|----------|
| **Neo4j** | Entity nodes (PERSON, ORG, PRODUCT, CONCEPT) + relationships with doc_id | Neo4j instance |
| **ChromaDB** | Text chunks + vector embeddings, metadata (doc_id, filename, chunk_index) | `./data/chroma_db/` |
| **SQLite** | Conversation history (messages + sources) + document metadata | `./data/conversations.db` |
| **Disk** | Uploaded PDF files | `./data/uploads/{doc_id}_{filename}` |

---

## License

This project is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
