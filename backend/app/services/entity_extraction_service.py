"""Entity/relationship extraction service using DashScope (OpenAI-compatible) API."""

import asyncio
import json
import re
from typing import Any

import openai

from app.config import Settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# --- Prompt -----------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一个知识图谱实体关系提取专家。
从用户给出的文本中提取实体和关系，严格按以下JSON格式输出，不要加任何解释：

{
  "entities": [
    {
      "name": "实体名称",
      "type": "PERSON|ORGANIZATION|PRODUCT|CONCEPT",
      "description": "对该实体的一句话描述（可选）"
    }
  ],
  "relationships": [
    {
      "source": "源实体名称（必须与entities中某个name完全一致）",
      "target": "目标实体名称（必须与entities中某个name完全一致）",
      "type": "WORKS_FOR|FOUNDED|INVESTED_IN|DEVELOPED|PART_OF|RELATED_TO|LEADS|USES|REGULATES"
    }
  ]
}

注意：
- 只提取文本中明确提到的实体和关系，不要推断
- 实体name必须是文本中出现的原始名称
- relationships中的source/target必须是entities中已有的name
- 只返回JSON，不要有```json```标记或其他内容
"""


# --- Helpers ----------------------------------------------------------

def _preprocess_html_tables(text: str) -> str:
    """Convert HTML tables to pipe-separated plain text."""
    def table_to_text(m: re.Match) -> str:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), re.DOTALL | re.IGNORECASE)
        lines = []
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
            cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            lines.append(" | ".join(cleaned))
        return "\n".join(lines)

    return re.sub(r"<table[^>]*>.*?</table>", table_to_text, text, flags=re.DOTALL | re.IGNORECASE)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, breaking on paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = re.split(r"\n{2,}", text)
    current = ""

    for para in paragraphs:
        # If a single paragraph exceeds max_chars, hard-split it
        if len(para) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
        elif len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping any markdown fences."""
    content = content.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()
    return json.loads(content)


# --- Service ----------------------------------------------------------

class EntityExtractionService:
    def __init__(self, settings: Settings):
        self._client = openai.OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
        )
        self._settings = settings

    def extract(self, markdown: str) -> dict:
        """
        Extract entities and relationships from markdown text.
        Returns {"nodes": [...], "edges": [...]}.
        """
        text = _preprocess_html_tables(markdown)
        log.info("entity_extraction.start", chars=len(text))

        chunks = _chunk_text(text, max_chars=self._settings.LANGEXTRACT_MAX_CHAR_BUFFER)
        log.info("entity_extraction.chunks", count=len(chunks))

        all_nodes: dict[str, dict] = {}   # name → node (dedup)
        all_edges: dict[tuple, dict] = {}  # (src, tgt, type) → edge (dedup)

        for i, chunk in enumerate(chunks):
            try:
                result = self._extract_chunk(chunk)
                for ent in result.get("entities", []):
                    name = ent.get("name", "").strip()
                    if name:
                        all_nodes.setdefault(name, {"name": name, "label": ent.get("type", "CONCEPT")})
                        # Merge extra fields
                        for k, v in ent.items():
                            if k not in ("name", "type") and v:
                                all_nodes[name][k] = v
                for rel in result.get("relationships", []):
                    src = rel.get("source", "").strip()
                    tgt = rel.get("target", "").strip()
                    rtype = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
                    if src and tgt:
                        all_edges[(src, tgt, rtype)] = {"source": src, "target": tgt, "type": rtype}
            except Exception as e:
                log.warning("entity_extraction.chunk_failed", chunk_index=i, error=str(e))

        nodes = list(all_nodes.values())
        edges = list(all_edges.values())
        log.info("entity_extraction.done", nodes=len(nodes), edges=len(edges))
        return {"nodes": nodes, "edges": edges}

    def _extract_chunk(self, text: str) -> dict[str, Any]:
        """Call the LLM to extract entities and relationships from one chunk."""
        response = self._client.chat.completions.create(
            model=self._settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"请从以下文本中提取实体和关系：\n\n{text}"},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return _parse_json_response(content)

    async def async_extract(self, markdown: str) -> dict:
        return await asyncio.to_thread(self.extract, markdown)
