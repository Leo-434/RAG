"""LangChain 1.x Agent-based answer service with SSE streaming.

All retrieval modes (hybrid / vector_only / graph_only) use create_agent.
No LCEL chains — every LLM call goes through a tool-equipped agent.
Multi-turn conversation history is passed as the messages list.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.models.query import QueryRequest, QueryResponse, SourceItem
from app.services.graph_service import GraphService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.services.conversation_service import ConversationService

log = get_logger(__name__)

# ── System prompts per mode ────────────────────────────────────────────────────

_SYSTEM_HYBRID = """\
你是一个专业的知识库问答助手，擅长从文档和知识图谱中检索信息来回答问题。

可用工具：
- semantic_search        — 向量语义检索，适合开放性问题、概念理解、事件描述
- entity_graph_search    — 实体关系图谱检索，适合查询某实体的关系、属性
- investment_search      — 专项投资关系检索，适合查询投融资数据
- fulltext_keyword_search — 关键词全文检索，适合精确名称定位

策略：先用最合适的工具，复杂问题可组合多个工具。
如信息不足请说明，不要捏造事实。用中文回答。"""

_SYSTEM_VECTOR = """\
你是一个文档问答助手，通过向量语义检索从文档库中找到相关内容后回答问题。
请使用 semantic_search 工具检索文档，再综合检索结果给出准确回答。
如资料不足请说明，不要捏造事实。用中文回答。"""

_SYSTEM_GRAPH = """\
你是一个知识图谱问答助手，通过查询 Neo4j 图谱中的实体和关系来回答问题。
可用工具：
- entity_graph_search    — 查询实体的直接关系（1跳子图）
- fulltext_keyword_search — 全文检索定位实体

先用 fulltext_keyword_search 定位实体，再用 entity_graph_search 展开关系。
如信息不足请说明，不要捏造事实。用中文回答。"""


def _build_history_messages(history: list[dict]) -> list:
    """Convert raw dict history to LangChain message objects."""
    result = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


class AnswerService:
    def __init__(
        self,
        settings: Settings,
        hybrid_service: HybridRetrievalService,
        graph_service: GraphService,
    ):
        self._settings = settings
        self._hybrid = hybrid_service
        self._graph = graph_service
        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
            temperature=0,
            max_tokens=2048,
        )
        # Build one agent per retrieval mode — all use create_agent, no LCEL
        self._hybrid_agent = self._build_hybrid_agent()
        self._vector_agent = self._build_vector_agent()
        self._graph_agent  = self._build_graph_agent()

    # ── Agent builders (called once at __init__) ──────────────────────────────

    def _build_hybrid_agent(self):
        """Agent with all four retrieval tools for hybrid mode."""
        vector_svc = self._hybrid._vector
        graph = self._graph

        @tool
        def semantic_search(query: str) -> str:
            """向量语义检索文档片段。适合开放性问题、概念理解、事件背景描述。
            参数 query: 自然语言查询文本。"""
            results = vector_svc.similarity_search(query, k=5)
            return "\n\n".join(r.content or "" for r in results) or "未找到相关内容。"

        @tool
        def entity_graph_search(entity_name: str) -> str:
            """在 Neo4j 知识图谱中查询实体及其直接关系（1跳子图）。
            适合查询某人/机构/产品的关联关系、职位、所属组织等。
            参数 entity_name: 实体名称，如 "苹果公司"、"史蒂夫·乔布斯"。"""
            return graph.entity_graph_context(entity_name)

        @tool
        def investment_search(org_name: str) -> str:
            """精确查询 Neo4j 中某组织的投资关系，包括投资方、被投方、金额、年份。
            适合融资问题、战略合作投资金额查询。
            参数 org_name: 组织名称。"""
            return graph.investment_search(org_name)

        @tool
        def fulltext_keyword_search(keyword: str) -> str:
            """使用 Neo4j 全文索引检索包含指定关键词的实体。
            适合定位特定名称相关的实体，辅助后续 entity_graph_search。
            参数 keyword: 搜索关键词。"""
            results = graph.fulltext_search(keyword, limit=10)
            if not results:
                return "未找到相关实体。"
            return "\n".join(f"{r['name']} ({r['label']})" for r in results)

        return create_agent(
            model=self._llm,
            tools=[semantic_search, entity_graph_search, investment_search, fulltext_keyword_search],
            system_prompt=_SYSTEM_HYBRID,
        )

    def _build_vector_agent(self):
        """Agent with only semantic_search for vector-only mode."""
        vector_svc = self._hybrid._vector

        @tool
        def semantic_search(query: str) -> str:
            """向量语义检索文档片段。输入查询文本，返回最相关的文档段落。
            参数 query: 自然语言查询文本。"""
            results = vector_svc.similarity_search(query, k=5)
            return "\n\n".join(r.content or "" for r in results) or "未找到相关内容。"

        return create_agent(
            model=self._llm,
            tools=[semantic_search],
            system_prompt=_SYSTEM_VECTOR,
        )

    def _build_graph_agent(self):
        """Agent with entity_graph_search + fulltext_keyword_search for graph-only mode."""
        graph = self._graph

        @tool
        def entity_graph_search(entity_name: str) -> str:
            """在 Neo4j 知识图谱中查询实体及其直接关系（1跳子图）。
            参数 entity_name: 实体名称。"""
            return graph.entity_graph_context(entity_name)

        @tool
        def fulltext_keyword_search(keyword: str) -> str:
            """使用 Neo4j 全文索引定位包含关键词的实体。
            参数 keyword: 搜索关键词。"""
            results = graph.fulltext_search(keyword, limit=10)
            if not results:
                return "未找到相关实体。"
            return "\n".join(f"{r['name']} ({r['label']})" for r in results)

        return create_agent(
            model=self._llm,
            tools=[entity_graph_search, fulltext_keyword_search],
            system_prompt=_SYSTEM_GRAPH,
        )

    def _get_agent(self, mode: str):
        """Select agent by retrieval mode."""
        return {
            "hybrid":      self._hybrid_agent,
            "vector_only": self._vector_agent,
            "graph_only":  self._graph_agent,
        }.get(mode, self._hybrid_agent)

    # ── Non-streaming answer ──────────────────────────────────────────────────

    async def answer(
        self,
        request: QueryRequest,
        start_time_ms: int,
        history: Optional[list[dict]] = None,
    ) -> QueryResponse:
        agent = self._get_agent(request.retrieval_mode.value)
        history_msgs = _build_history_messages(history or [])

        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": history_msgs + [HumanMessage(content=request.question)]},
        )
        messages = result.get("messages", [])
        final_answer = messages[-1].content if messages else "无法生成回答。"

        # Also fetch source snippets for citation display in the UI
        sources = await self._hybrid.retrieve(
            request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode.value,
        )

        latency = int(time.time() * 1000) - start_time_ms
        return QueryResponse(answer=final_answer, sources=sources, latency_ms=latency)

    # ── SSE streaming ─────────────────────────────────────────────────────────

    async def stream_answer(
        self,
        request: QueryRequest,
        history: Optional[list[dict]] = None,
        conv_svc: Optional["ConversationService"] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Yield SSE-formatted strings:
          event: token   data: {"content": "..."}
          event: sources data: {"sources": [...]}
          event: done    data: {}
          event: error   data: {"message": "..."}
        """
        try:
            agent = self._get_agent(request.retrieval_mode.value)
            history_msgs = _build_history_messages(history or [])
            full_response = ""

            # Stream final-answer tokens via astream_events (LangChain 1.x / version="v2")
            async for event in agent.astream_events(
                {"messages": history_msgs + [HumanMessage(content=request.question)]},
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        full_response += chunk.content
                        payload = json.dumps({"content": chunk.content}, ensure_ascii=False)
                        yield f"event: token\ndata: {payload}\n\n"

            # Fetch sources for citation display after the answer stream ends
            sources = await self._hybrid.retrieve(
                request.question,
                top_k=request.top_k,
                mode=request.retrieval_mode.value,
            )
            sources_data = json.dumps(
                {"sources": [s.model_dump() for s in sources]},
                ensure_ascii=False,
            )
            yield f"event: sources\ndata: {sources_data}\n\n"
            yield "event: done\ndata: {}\n\n"

            # Persist messages to conversation after stream completes
            if request.conversation_id and conv_svc and full_response:
                await conv_svc.add_message(request.conversation_id, "user", request.question)
                await conv_svc.add_message(
                    request.conversation_id,
                    "assistant",
                    full_response,
                    sources=sources,
                )
                # Auto-set title from first user question
                conv = await conv_svc.get_conversation(request.conversation_id)
                if conv and conv.title == "新对话" and len(conv.messages) <= 2:
                    await conv_svc.update_title(request.conversation_id, request.question[:50])

        except Exception as e:
            log.error("answer_service.stream_error", error=str(e))
            error_data = json.dumps({"message": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"
