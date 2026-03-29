# GraphRAG 知识图谱增强问答系统

<div align="center">

![GraphRAG](https://img.shields.io/badge/GraphRAG-Knowledge%20Base-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**基于知识图谱增强的 AI 问答系统，支持混合检索、多轮对话与可视化知识图谱探索**

[English](./README_EN.md) · 中文

</div>

---

## 目录

- [系统概述](#系统概述)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [API 文档](#api-文档)
- [项目结构](#项目结构)
- [数据流说明](#数据流说明)
- [开源协议](#开源协议)

---

## 系统概述

GraphRAG 是一个将**向量语义检索**与**知识图谱**深度融合的 AI 问答系统。它能自动解析 PDF 文档，提取实体与关系构建知识图谱，并通过三种检索模式回答自然语言问题，所有对话历史与文档元数据均持久化存储，重启服务后数据不丢失。

```
PDF 文档 → MinerU 解析 → 文本分块 ─┬─ ChromaDB 向量存储
                                    └─ 实体提取 → Neo4j 知识图谱

用户提问 → 混合检索（向量 + 图谱 RRF） → LangChain Agent → 流式回答
```

---

## 核心功能

### 🔍 三种检索模式
| 模式 | 说明 |
|------|------|
| **混合检索** | 向量语义检索 + 图谱全文检索，通过 RRF（倒数秩融合）算法合并排名，效果最佳 |
| **语义检索** | 仅使用 ChromaDB 向量相似度检索，适合开放性问答 |
| **图谱检索** | 仅使用 Neo4j 全文索引，适合精确实体查询 |

### 💬 多轮对话
- 对话历史持久化存储于 SQLite，重启后自动还原
- 侧边栏显示历史对话列表，支持切换与删除
- 对话标题自动从第一条问题生成
- 每轮对话携带完整历史上下文，支持追问

### 📄 文档管理
- 支持 PDF 上传，异步后台处理，不阻塞用户操作
- 实时追踪进度：`uploading → ingesting → ready / failed`
- 显示文档的 chunk 数、实体数、关系数
- 删除时级联清理：Neo4j 节点、ChromaDB 向量、磁盘文件、元数据

### 🕸 知识图谱可视化
- 实体类型筛选（PERSON / ORGANIZATION / PRODUCT / CONCEPT）
- 可交互画布：拖拽、缩放、全屏
- 双击节点展开一跳邻居子图
- 全文关键词搜索定位实体
- 可按文档筛选图谱范围

### ⚡ 流式回答
- Token 逐字流式输出，基于 SSE（Server-Sent Events）
- 支持中途停止生成
- 回答完成后展示引用来源（文档片段 / 图谱实体）

---

## 技术架构

### 整体架构图

```
┌─────────────────────────────────────────────┐
│              前端 (React 18 + Vite)          │
│  ChatPage  DocumentsPage  GraphPage          │
│  Zustand 状态管理 + SSE 流式处理             │
└──────────────────┬──────────────────────────┘
                   │ REST API / SSE
┌──────────────────▼──────────────────────────┐
│           后端 (FastAPI + Python 3.10+)      │
│                                              │
│  /api/query     → AnswerService              │
│  /api/documents → IngestionService           │
│  /api/graph     → GraphService               │
│  /api/conversations → ConversationService    │
│                                              │
│  HybridRetrievalService (RRF 融合)           │
│  LangChain Agents (create_agent)             │
└──────┬──────────────┬──────────┬────────────┘
       │              │          │
   ┌───▼───┐    ┌─────▼───┐  ┌──▼──────┐
   │ Neo4j │    │ChromaDB │  │ SQLite  │
   │图谱存储│   │向量存储  │  │元数据   │
   └───────┘    └─────────┘  └─────────┘
```

### 技术栈

**后端**
| 组件 | 技术 | 版本 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 0.115+ |
| LLM | DashScope (Qwen3-Max) | — |
| 嵌入模型 | DashScope text-embedding-v4 | — |
| Agent 框架 | LangChain + LangGraph | 1.2+ / 1.0+ |
| 图数据库 | Neo4j | 5.x |
| 向量数据库 | ChromaDB | 0.5+ |
| 对话持久化 | SQLite (aiosqlite) | — |
| PDF 解析 | MinerU API | VLM 模式 |
| 数据校验 | Pydantic v2 | 2.0+ |

**前端**
| 组件 | 技术 | 版本 |
|------|------|------|
| 框架 | React | 18.3.1 |
| 路由 | React Router | 7.13.0 |
| 状态管理 | Zustand | 5.0.12 |
| UI 组件库 | Radix UI + Tailwind CSS | 1.x / 4.x |
| 构建工具 | Vite | 6.3.5 |
| 通知 | Sonner | 2.0.3 |

---

## 快速开始

### 前置条件

- Python 3.10+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) 包管理器（`pip install uv`）
- Neo4j 数据库（本地或云端）
- DashScope API Key（[申请地址](https://dashscope.aliyuncs.com)）
- MinerU API Key（[申请地址](https://mineru.net)）

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/GraphRAG.git
cd GraphRAG
```

### 2. 启动后端

```bash
cd backend

# 安装依赖
uv sync

# 复制并填写配置
cp .env.example .env
# 编辑 .env，填入 API Key 和数据库连接信息

# 启动开发服务器
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 3. 启动前端

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端访问地址：http://localhost:5173

### 4. 使用流程

1. 打开 http://localhost:5173，进入**文档库**页面
2. 上传 PDF 文档，等待状态变为 `ready`（需要几分钟，视文档大小而定）
3. 切换到**问答**页面，直接输入问题即可开始对话
4. 在**知识图谱**页面探索从文档中提取的实体与关系

---

## 环境配置

在 `backend/` 目录下创建 `.env` 文件（参考 `.env.example`）：

```bash
# ── LLM & 嵌入（DashScope / 阿里云）──────────────
DASHSCOPE_API_KEY=sk-your-api-key-here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-max-2026-01-23
EMBEDDING_MODEL=text-embedding-v4

# ── PDF 解析（MinerU）────────────────────────────
MINERU_API_KEY=your-mineru-api-key-here
MINERU_BASE_URL=https://mineru.net/api/v4
MINERU_MODEL_VERSION=vlm

# ── 图数据库（Neo4j）─────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# ── 向量数据库（ChromaDB）────────────────────────
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION=document_chunks

# ── 文件处理参数 ──────────────────────────────────
CHUNK_SIZE=400
CHUNK_OVERLAP=60
EMBED_BATCH_SIZE=8
LANGEXTRACT_MAX_CHAR_BUFFER=3000

# ── 检索参数 ──────────────────────────────────────
DEFAULT_TOP_K=5
RRF_K=60

# ── 文件上传 ──────────────────────────────────────
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_MB=50

# ── CORS ──────────────────────────────────────────
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

---

## API 文档

完整 Swagger 文档：`http://localhost:8000/docs`

### 主要端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 系统健康检查（Neo4j + ChromaDB 状态） |
| `POST` | `/api/documents/upload` | 上传 PDF，触发异步处理流水线 |
| `GET` | `/api/documents` | 获取所有文档列表及状态 |
| `DELETE` | `/api/documents/{doc_id}` | 删除文档及关联数据 |
| `POST` | `/api/query` | 非流式问答（返回完整回答） |
| `POST` | `/api/query/stream` | SSE 流式问答（逐 token 输出） |
| `GET` | `/api/graph/stats` | 知识图谱统计信息 |
| `GET` | `/api/graph/entities` | 实体列表（分页 + 过滤） |
| `GET` | `/api/graph/entities/{name}/neighbors` | 实体一跳邻居子图 |
| `POST` | `/api/graph/search` | 全文关键词搜索实体 |
| `GET` | `/api/conversations` | 获取所有对话历史 |
| `POST` | `/api/conversations` | 新建对话 |
| `DELETE` | `/api/conversations/{conv_id}` | 删除对话及消息 |

### 流式问答示例

```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "OpenAI 的创始人有哪些？",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "conversation_id": "your-conv-id"
  }'
```

SSE 事件格式：
```
event: token
data: {"content": "OpenAI 的创始人包括"}

event: token
data: {"content": "Sam Altman、Greg Brockman..."}

event: sources
data: {"sources": [...]}

event: done
data: {}
```

---

## 项目结构

```
GraphRAG/
├── backend/                        # FastAPI 后端
│   ├── app/
│   │   ├── main.py                 # 应用入口 + 生命周期管理
│   │   ├── config.py               # Pydantic 配置（读取 .env）
│   │   ├── dependencies.py         # FastAPI 依赖注入
│   │   ├── models/                 # Pydantic 数据模型
│   │   │   ├── document.py
│   │   │   ├── query.py
│   │   │   ├── graph.py
│   │   │   └── conversation.py
│   │   ├── routers/                # REST API 路由
│   │   │   ├── health.py
│   │   │   ├── documents.py
│   │   │   ├── query.py
│   │   │   ├── graph.py
│   │   │   └── conversations.py
│   │   ├── services/               # 业务逻辑层
│   │   │   ├── ingestion_service.py      # 文档处理流水线
│   │   │   ├── vector_service.py         # ChromaDB 操作
│   │   │   ├── graph_service.py          # Neo4j 操作
│   │   │   ├── answer_service.py         # LangChain Agent 问答
│   │   │   ├── hybrid_retrieval_service.py # RRF 融合检索
│   │   │   ├── conversation_service.py   # 对话历史持久化
│   │   │   ├── document_registry_service.py # 文档元数据持久化
│   │   │   └── entity_extraction_service.py # 实体提取
│   │   └── utils/
│   │       ├── logger.py
│   │       ├── rrf.py              # 倒数秩融合算法
│   │       └── mineru_parser.py    # MinerU API 封装
│   ├── .env.example                # 环境变量模板
│   ├── pyproject.toml              # 项目依赖
│   └── uv.lock                     # 依赖锁定文件
│
├── frontend/                       # React 前端
│   ├── src/app/
│   │   ├── App.tsx                 # 根组件
│   │   ├── routes.ts               # 路由配置
│   │   ├── api.ts                  # API 请求层
│   │   ├── store.ts                # Zustand 全局状态
│   │   └── components/
│   │       ├── Layout.tsx          # 主布局（侧边栏 + 内容区）
│   │       ├── Sidebar.tsx         # 对话列表 + 导航
│   │       ├── ChatPage.tsx        # 问答界面
│   │       ├── DocumentsPage.tsx   # 文档管理界面
│   │       ├── GraphPage.tsx       # 知识图谱可视化
│   │       ├── SimpleMarkdown.tsx  # Markdown 渲染
│   │       └── ui/                 # 基础 UI 组件库
│   └── package.json
│
├── .gitignore
├── README.md                       # 中文文档（本文件）
└── README_EN.md                    # English documentation
```

---

## 数据流说明

### 文档处理流水线

```
①  POST /api/documents/upload
    └─ 保存文件到磁盘 + SQLite 记录状态=uploading

②  后台任务 run_pipeline()
    ├─ MinerUParser.parse(pdf) → Markdown 文本
    ├─ RecursiveCharacterTextSplitter → 文本块列表
    ├─ [并行执行]
    │   ├─ VectorService.embed_and_store() → ChromaDB
    │   └─ EntityExtractionService.extract() → {nodes, edges}
    │       └─ GraphService.write_graph_data() → Neo4j
    └─ SQLite 更新状态=ready，记录 chunk/entity/relation 数量

③  GET /api/documents/{doc_id}  ← 前端每 5 秒轮询
```

### 问答流程

```
①  POST /api/query/stream  {question, retrieval_mode, conversation_id}

②  加载对话历史  ← ConversationService(SQLite)

③  HybridRetrievalService.retrieve()
    ├─ (hybrid)  Vector + Graph → RRF 融合
    ├─ (vector)  VectorService.similarity_search()
    └─ (graph)   GraphService.fulltext_search()

④  AnswerService.stream_answer()
    └─ LangChain Agent (create_agent)
        ├─ 工具调用（semantic_search / entity_graph_search / ...）
        └─ 逐 token 输出 → SSE event: token

⑤  输出 sources → SSE event: sources
    保存消息到 SQLite → SSE event: done
```

---

## 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

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
