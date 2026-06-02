"""
agent.py

GraphRAG Agent 工厂入口。
"""
from __future__ import annotations
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent  # ✨ 用 LangChain 1.x 的新入口

from services.tool import (
    build_detail_retrieval_tool,
    build_summary_retrieval_tool,
)


SYSTEM_PROMPT = """你是一个文档问答助手,根据可用工具的 docstring 选择合适工具回答用户问题。

通用规则:
- 严格基于工具返回的 XML 内容回答,禁止凭训练知识补全文档内容
- 工具返回值开头的 <!-- 处理规则 --> 必须遵守
- 遇到 <warning> 时告知用户,不要伪造内容
- 单次调用信息立即回答,避免重复调用
"""


def build_graphrag_agent(
    file_id: str,
    driver: Any,
    llm: Any,
    *,
    top_k: int = 3,
    enable_summary_generation: bool = True,
    force_rebuild_summary: bool = False,
    level_override: Optional[int] = None,
    system_prompt: Optional[str] = None,
):
    """
    构造绑定到指定文档的 GraphRAG Agent。

    Returns:
        (agent, tools) —— agent 可调用 .invoke / .stream,
                         tools 用于外部读取 citations
    """
    tools = [
        build_detail_retrieval_tool(
            file_id=file_id,
            driver=driver,
            top_k=1,
        ),
        build_summary_retrieval_tool(
            file_id=file_id,
            driver=driver,
            llm=llm,
            top_k=top_k,
            enable_summary_generation=enable_summary_generation,
            force_rebuild=force_rebuild_summary,
            level_override=level_override,
        ),
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt or SYSTEM_PROMPT,
    )

    return agent, tools


def ask(agent, tools, question: str) -> dict:
    """问答入口,返回 {answer, citations}。"""
    for t in tools:
        t.clear_state()

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    answer = result["messages"][-1].content
    citations = []
    for t in tools:
        citations.extend(t.get_last_citations())

    return {"answer": answer, "citations": citations}

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def truncate(text: str, limit: int = 500) -> str:
    """超过 limit 字符则截断并加省略号。"""
    return text[:limit] + "..." if len(text) > limit else text


def agent_query(agent, tools, question: str) -> dict:
    """
    调用 GraphRAG Agent 处理一个问题。

    Args:
        agent: build_graphrag_agent 返回的 agent 实例
        tools: build_graphrag_agent 返回的 tools 列表(用于读 citations)
        question: 用户问题

    Returns:
        {
            "question": str,              # 原问题(便于日志)
            "answer": str,                # LLM 基于 XML 生成的最终回答
            "citations": list[dict],      # 从工具闭包读出的结构化引用(给前端)
            "evidence": list[dict],       # agent 工具调用记录(给开发者/调试)
        }
    """
    # ---- 1) 清空工具上次残留的 citations 状态 ----
    for t in tools:
        t.clear_state()

    # ---- 2) 调用 agent ----
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    messages = result["messages"]

    # ---- 3) 提取最终回答(最后一条消息的内容)----
    answer = messages[-1].content

    # ---- 4) 收集结构化 citations(从工具闭包,绕过 LLM)----
    citations = []
    for t in tools:
        citations.extend(t.get_last_citations())

    # ---- 5) 提取工具调用记录 evidence ----
    evidence = []
    for msg in messages:
        # 5.1 AIMessage 带 tool_calls: agent 决定调用工具
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                evidence.append({
                    "type": "call",
                    "tool": tc["name"],
                    "args": tc["args"],
                })
        # 5.2 ToolMessage: 工具返回结果
        elif isinstance(msg, ToolMessage):
            evidence.append({
                "type": "result",
                "tool": msg.name,
                "content": msg.content,
            })

    # ---- 6) 合并所有工具的 graph_payload(图谱节点 id 集合)----
    merged_graph_payload = {
        "section_ids": [],
        "chunk_ids": [],
        "table_names": [],
        "row_ids": [],
    }
    for t in tools:
        getter = getattr(t, "get_last_graph_payload", None)
        if getter is None:
            continue
        try:
            payload = getter() or {}
        except Exception:
            continue
        for k in merged_graph_payload:
            merged_graph_payload[k].extend(payload.get(k, []) or [])

    # 去重保序
    merged_graph_payload = {
        k: list(dict.fromkeys(v)) for k, v in merged_graph_payload.items()
    }

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "evidence": evidence,
        "graph_payload": merged_graph_payload,   # ✨ 新增
    }