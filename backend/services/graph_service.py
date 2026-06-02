# services/graph_service.py
"""
GraphRAG 服务层。

设计目标:
  与 rag_service.py 保持完全一致的对外接口形状,使 app.py 的 /chat 路由
  可以用同一段 SSE 转发逻辑处理两种模式,只需根据 mode 字段切换 retrieve
  和 answer_stream 的具体实现。

对外暴露:
  - retrieve_graph(question, file_id)            → (citations, context_text)
  - answer_stream_graph(question, citations, ...) → AsyncGenerator[dict, None]
  - invalidate_agent_cache(file_id)              → 清缓存

新增事件类型(2024-Q):
  answer_stream_graph 会在 token 推送之前,先 yield 一个:
    {"type": "graph_query", "data": {"cypher": "...", "node_count": N}}
  前端 GraphView 收到后立即切换图谱视图,展示本次问答涉及的子图。

实现策略:
  - 流式: A1 伪流式(agent.invoke 一次性返回完整 answer, 按 chunk 模拟流式)
  - 历史: B1 暂不接入多轮(每次问答独立)
  - citation 适配: C1 在本层做字段映射, 前端无感切换
  - 图片处理: 复用 services.image_utils
  - Neo4j driver: 模块级单例
  - Agent: 按 file_id 缓存
  - LLM: 复用 rag_service._get_llm()
"""
from __future__ import annotations

import os
import json
import asyncio
import traceback
from typing import Any, AsyncGenerator, Dict, List, Tuple, Optional

from neo4j import GraphDatabase

# 复用 rag_service 的 LLM 工厂——保证 PDF 模式和图谱模式用同一个模型/配置
from services.rag_service import _get_llm

# 图谱 Agent 装配入口
from services.agent import build_graphrag_agent, agent_query

# 图片 URL 工具,与 rag_service 共享
from services.image_utils import (
    rewrite_image_urls,
    extract_image_urls,
    normalize_table_img_paths,
)


# =============================================================================
# 配置
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "20472036")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

TOP_K = 3
ENABLE_SUMMARY_GEN = True
FORCE_REBUILD_SUMMARY = False

PSEUDO_STREAM_CHUNK_SIZE = 8
PSEUDO_STREAM_DELAY_SEC = 0.015

# 图谱可视化 Cypher 的全局节点上限(防止节点爆炸)
GRAPH_CYPHER_LIMIT = 500


# =============================================================================
# 模块级单例
# =============================================================================

_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

_agent_cache: Dict[str, Tuple[Any, list]] = {}


def _get_or_build_agent(file_id: str) -> Tuple[Any, list]:
    """按 file_id 缓存 Agent。"""
    if file_id not in _agent_cache:
        llm = _get_llm()
        agent, tools = build_graphrag_agent(
            file_id=file_id,
            driver=_driver,
            llm=llm,
            top_k=TOP_K,
            enable_summary_generation=ENABLE_SUMMARY_GEN,
            force_rebuild_summary=FORCE_REBUILD_SUMMARY,
        )
        _agent_cache[file_id] = (agent, tools)
        print(f"[graph_service] Agent built and cached for file_id={file_id}")
    return _agent_cache[file_id]


def invalidate_agent_cache(file_id: Optional[str] = None) -> None:
    if file_id is None:
        _agent_cache.clear()
        print("[graph_service] All agent caches invalidated.")
    else:
        _agent_cache.pop(file_id, None)
        print(f"[graph_service] Agent cache invalidated for file_id={file_id}")


# =============================================================================
# Citation 字段适配
# =============================================================================

def _normalize_citation(raw: dict, idx: int) -> dict:
    """把图谱原生 citation 适配为前端期望结构。"""
    file_id = raw.get("fileId", "")
    ctype = raw.get("type", "unknown")

    title = (
        raw.get("title")
        or raw.get("section_name")
        or raw.get("table_name")
        or "(未命名)"
    )
    raw_markdown = raw.get("markdown", "") or ""
    summary = raw.get("summary", "") or ""

    markdown = rewrite_image_urls(raw_markdown, file_id) if file_id else raw_markdown
    inline_images = extract_image_urls(raw_markdown, file_id) if file_id else []
    table_imgs_normalized = (
        normalize_table_img_paths(raw.get("table_img_path"), file_id)
        if file_id else []
    )

    snippet_source = summary if summary else markdown
    snippet_short = snippet_source.strip()

    all_images: List[Dict[str, str]] = []
    seen_urls = set()
    for img in inline_images + table_imgs_normalized:
        if img["url"] not in seen_urls:
            seen_urls.add(img["url"])
            all_images.append(img)

    table_imgs_url_list: List[str] = [img["url"] for img in table_imgs_normalized]

    return {
        "citation_id": f"{file_id}-g{idx}",
        "fileId": file_id,
        "rank": raw.get("rank", idx),
        "page": 1,
        "pages": [1],
        "snippet": snippet_short,
        "score": 0.0,
        "previewUrl": "",
        "images": all_images,

        "source": "graph",
        "type": ctype,
        "title": title,
        "markdown": markdown,
        "summary": summary,
        "section_name": raw.get("section_name", ""),
        "section_path": raw.get("section_path", ""),
        "structure_info": raw.get("structure_info", ""),
        "level": raw.get("level"),
        "table_name": raw.get("table_name", ""),
        "table_img_path": table_imgs_url_list,
    }


# =============================================================================
# ✨ 图谱可视化 Cypher 构造
# =============================================================================

def _build_graph_cypher(payload: dict) -> Tuple[Optional[str], int]:
    """
    根据 Agent 收集到的 graph_payload 构造一段 Cypher,
    供前端 NeoVis 直接执行渲染"本次问答涉及的子图"。

    Args:
        payload: {
            "section_ids":  list[str],   # Section.entity_id
            "chunk_ids":    list[str],   # Chunk.entity_id
            "table_names":  list[str],   # Table.entity_name
            "row_ids":      list[str],   # TableRow.row_id
        }

    Returns:
        (cypher, total_node_count):
          - cypher 为可直接传给 NeoVis 的字符串;若 payload 全空则返回 None
          - total_node_count 是 4 类节点 id 数的总和(便于前端 UI 展示)

    Cypher 设计:
        1. 用 WHERE ... IN [...] 分别匹配 4 类节点
        2. 把所有命中节点 collect 为 nodes 集合
        3. OPTIONAL MATCH 找出"两端都在 nodes 内"的边
        4. RETURN n1, r, n2,限制 LIMIT 500

    安全:
        - 使用 json.dumps(..., ensure_ascii=False) 序列化字符串数组,
          能自动处理 entity_id 里可能的 " ' \\ 等字符
        - JSON 字符串字面量与 Cypher 列表字面量在大多数情况下兼容
          (双引号包裹的字符串,反斜杠转义形式一致)
    """
    section_ids = payload.get("section_ids") or []
    chunk_ids = payload.get("chunk_ids") or []
    table_names = payload.get("table_names") or []
    row_ids = payload.get("row_ids") or []

    total = len(section_ids) + len(chunk_ids) + len(table_names) + len(row_ids)
    if total == 0:
        return None, 0

    conditions: List[str] = []

    def _to_cypher_list(items: List[str]) -> str:
        # 用 JSON 序列化,自动处理引号和特殊字符
        return json.dumps(items, ensure_ascii=False)

    if section_ids:
        conditions.append(
            f"(n:Section AND n.entity_id IN {_to_cypher_list(section_ids)})"
        )
    if chunk_ids:
        conditions.append(
            f"(n:Chunk AND n.entity_id IN {_to_cypher_list(chunk_ids)})"
        )
    if table_names:
        conditions.append(
            f"(n:Table AND n.entity_name IN {_to_cypher_list(table_names)})"
        )
    if row_ids:
        conditions.append(
            f"(n:TableRow AND n.row_id IN {_to_cypher_list(row_ids)})"
        )

    where_clause = " OR ".join(conditions)

    cypher = (
        f"MATCH (n) WHERE {where_clause} "
        f"WITH collect(DISTINCT n) AS nodes "
        f"UNWIND nodes AS n1 "
        f"OPTIONAL MATCH (n1)-[r]-(n2) "
        f"WHERE n2 IN nodes "
        f"RETURN n1, r, n2 "
        f"LIMIT {GRAPH_CYPHER_LIMIT}"
    )

    return cypher, total


# =============================================================================
# 答案缓存
# =============================================================================
# 为什么需要这个:
#   rag_service 是"先检索再生成";图谱 Agent 是"一步到位"
#   retrieve 阶段就把 answer + graph_payload 拿到了,
#   answer_stream 阶段消费这两份缓存

_pending_answers: Dict[Tuple[str, str], str] = {}
_pending_graph_cyphers: Dict[Tuple[str, str], Optional[Tuple[str, int]]] = {}


# =============================================================================
# 主流程
# =============================================================================

async def retrieve_graph(question: str, file_id: str) -> Tuple[List[dict], str]:
    """图谱检索入口(同 rag_service.retrieve 的形状)。"""
    cache_key = (file_id, question)

    if not file_id:
        _pending_answers[cache_key] = (
            "图谱模式需要先选择一个已构建图谱的文档。"
            "请上传 PDF 或从历史库中选择文件。"
        )
        _pending_graph_cyphers[cache_key] = None
        return [], "graph_mode"

    try:
        agent, tools = _get_or_build_agent(file_id)
        result = await asyncio.to_thread(agent_query, agent, tools, question)
    except Exception as e:
        traceback.print_exc()
        _pending_answers[cache_key] = f"图谱查询出错:{e}"
        _pending_graph_cyphers[cache_key] = None
        return [], "graph_mode"

    answer = result.get("answer", "") or "(图谱 Agent 未产出回答)"
    raw_citations = result.get("citations", []) or []
    graph_payload = result.get("graph_payload") or {}

    # 字段适配
    citations = [_normalize_citation(c, i) for i, c in enumerate(raw_citations, start=1)]

    # ✨ 拼接图谱可视化 Cypher
    cypher, node_count = _build_graph_cypher(graph_payload)
    if cypher:
        print(
            f"[graph_service] 本次问答涉及 {node_count} 个节点 "
            f"(sections={len(graph_payload.get('section_ids', []))}, "
            f"chunks={len(graph_payload.get('chunk_ids', []))}, "
            f"tables={len(graph_payload.get('table_names', []))}, "
            f"rows={len(graph_payload.get('row_ids', []))})"
        )
    else:
        print("[graph_service] 本次问答未收集到图谱节点,跳过 graph_query 推送")

    # 暂存,等 answer_stream_graph 消费
    _pending_answers[cache_key] = answer
    _pending_graph_cyphers[cache_key] = (cypher, node_count) if cypher else None

    return citations, "graph_mode"


async def answer_stream_graph(
    question: str,
    citations: List[dict],
    context_text: str,
    branch: str,
    session_id: Optional[str] = None,
    file_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    图谱流式生成入口。

    事件顺序:
      1. ✨ graph_query  ← 新增:让前端先切换图谱视图
      2. citation(可能多个)
      3. token(可能多个)
      4. done

    Args:
        question, citations, context_text, branch: 与 rag_service.answer_stream 一致
        session_id: 暂未使用
        file_id:    用于从模块级字典中取暂存的 answer/cypher
    """
    cache_key = (file_id or "", question)

    # ---- 1) ✨ 先推送 graph_query 事件,让前端立即切换图谱 ----
    pending_cypher = _pending_graph_cyphers.pop(cache_key, None)
    if pending_cypher:
        cypher, node_count = pending_cypher
        yield {
            "type": "graph_query",
            "data": {"cypher": cypher, "node_count": node_count},
        }

    # ---- 2) 再推送 citations ----
    if citations:
        for c in citations:
            yield {"type": "citation", "data": c}

    # ---- 3) 取暂存 answer ----
    answer = _pending_answers.pop(cache_key, None)

    if answer is None:
        yield {
            "type": "error",
            "data": {
                "message": "图谱模式内部错误:未找到本轮问答的答案缓存。"
                           "请检查 retrieve_graph 是否被先调用。"
            },
        }
        yield {"type": "done", "data": {"used_retrieval": False}}
        return

    # ---- 4) 伪流式吐 token ----
    try:
        for i in range(0, len(answer), PSEUDO_STREAM_CHUNK_SIZE):
            chunk = answer[i : i + PSEUDO_STREAM_CHUNK_SIZE]
            yield {"type": "token", "data": chunk}
            await asyncio.sleep(PSEUDO_STREAM_DELAY_SEC)
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "data": {"message": f"流式输出错误:{e}"}}
        return

    # ---- 5) 完成 ----
    yield {
        "type": "done",
        "data": {"used_retrieval": bool(citations)},
    }