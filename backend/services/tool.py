"""
tools.py

GraphRAG Agent 的双轨检索工具定义。

设计要点:
  1. 工具通过工厂函数(build_*_tool)生成,闭包注入 file_id / driver / llm,
     避免全局状态,支持多文档并发。
  2. 仅暴露 2 个高层工具给 Agent:
       - detail_retrieval_tool  (细节检索: text + table + table_row)
       - summary_retrieval_tool (总结类检索: 含 level 自判定 + 自底向上摘要)
     vector_search 在两者内部调用,Agent 不直接看到。
  3. 工具内部所有异常都被捕获并以 <error>...</error> 形式返回给 Agent
     (Agent 拿到异常对象会 crash,拿到字符串能继续工作)。
  4. 输出统一使用 XML 风格标签包裹结构化字段:
       - 属性值统一用双引号,降低与正文撇号的冲突
       - 不在 tool 输出中夹带 "系统提示" 文本; 严格性约束放在
         agent.SYSTEM_PROMPT 中,职责分离

  5. ✨ 新增:graph_payload 收集
     每次工具调用时,把本次涉及的图谱节点 id 收集到 last_call_state["graph_payload"],
     供外层(graph_service)拼成"本次问答的子图 Cypher"推送给前端可视化。
     - section_ids: Section 节点的 entity_id 列表
     - chunk_ids:   Chunk 节点的 entity_id 列表
     - table_names: Table 节点的 entity_name 列表(Table 没有独立的 entity_id)
     - row_ids:     TableRow 节点的 row_id 列表
"""
from __future__ import annotations
from typing import Any, Optional
from langchain_core.tools import tool

# === 基础检索与图谱操作依赖 ===
from services.rag_pipeline import retrieve_entities          # FAISS 向量检索
from services.graph_search import (                                   # Neo4j 细节表检索
    get_full_tables_from_hits,
    get_table_context_with_chunks,
)
from services.graph_search_section import get_section_context_with_summary  # Neo4j 宏观摘要检索

# === 本包内的依赖 ===
from services.level_detector import decide_level
from services.summarizer import (
    table_summarizer,
    chunk_summarizer,
    section_summarizer,
)


# ============================================================
# 工具内部:graph_payload 操作的小工具
# ============================================================

def _new_graph_payload() -> dict:
    """工具开始时清空 / 初始化 graph_payload。"""
    return {
        "section_ids": [],
        "chunk_ids": [],
        "table_names": [],
        "row_ids": [],
    }


def _dedup_payload(payload: dict) -> dict:
    """收尾去重,保留顺序。"""
    return {k: list(dict.fromkeys(v)) for k, v in payload.items()}


# ============================================================
# 工具 1: 细节检索 (text + table + table_row)
# ============================================================

def build_detail_retrieval_tool(
    file_id: str,
    driver: Any,
    top_k: int = 3,
):
    """
    构造细节检索工具(合并 chunk + table)。
    
    内部并行检索:
      - text 命中     → 段落原文 [图谱: Chunk + 它的父 Section]
      - table 命中    → get_table_context_with_chunks [图谱: 命中 Chunk + 同 Chunk 下所有 Table]
      - table_row 命中 → get_full_tables_from_hits [图谱: 命中 Table + 该 Table 下所有 TableRow]
    """
    
    # ✨ 同时维护 citations 和 graph_payload
    last_call_state: dict = {
        "citations": [],
        "graph_payload": _new_graph_payload(),
    }
    
    OUTPUT_RULE_HINT = (
        '<!-- 输出规则:\n'
        '\n'
        '下方 <context_item> 标签包裹的内容是检索到的文档片段,你必须基于这些内容回答用户问题。\n'
        'type 属性标识三种来源:\n'
        '  - full_table_data: 表格的具体行数据(精确数值)\n'
        '  - table_chunk:     表格 + 周边段落上下文(信息量最全)\n'
        '  - text_chunk:      段落原文(不含表格的文字描述)\n'
        '\n'
        '【主源选择】按问题类型匹配一种 type 作为主要回答依据,其他作为补充:\n'
        '\n'
        '  ① 表格数值/内容查询 → 优先 full_table_data\n'
        '     场景: "X的标准范围""Y的允许极限""Z的偏差/精度/具体数值"\n'
        '     处理: 以 full_table_data 内容为主回答,精确引用数值\n'
        '\n'
        '  ② 表格概括/上下文 → 优先 table_chunk\n'
        '     场景: "总结表X的方法""表X包括什么类型/有什么要点"\n'
        '     处理: 以 table_chunk 内容为主回答\n'
        '\n'
        '  ③ 段落细节(不涉及表格)→ 优先 text_chunk\n'
        '     场景: "X的要求""Y的工艺流程""Z的适用范围"\n'
        '     处理: 以 text_chunk 内容为主回答,引用段落原文\n'
        '\n'
        '  ④ 无匹配 → 综合所有 context_item 的内容\n'
        '\n'
        '【格式】数值字段原样保留,禁改写/压缩;多 context_item 用 --- 分隔。\n'
        '-->'
    )
    
    @tool(parse_docstring=False)
    def detail_retrieval_tool(question: str) -> str:
        """【细节/段落/表格必用】段落、参数、表格的精确检索,回答"具体内容是什么"类问题。

        必用场景(只要属于以下任一,都用本工具):
        - 段落细节: "X的要求""Y的工艺流程、工艺规范""Z的适用范围"
        - 数值范围: "X的标准范围""Y的允许极限""Z的偏差、精度"
        - 表格概括: "总结表X的方法""表X包括什么类型/分别有什么要点"
          (只要问题涉及"某张表/某类表"的内容,使用本工具)

        不适用场景(请改用 summary_retrieval_tool):
        - 整章/整节的综述: "某章主要讲了什么""某节的要求、工艺规范概述"

        参数:
            question: 用户的段落细节和表格相关问题
        """
        last_call_state["citations"] = []
        last_call_state["graph_payload"] = _new_graph_payload()
        graph_payload = last_call_state["graph_payload"]
        
        # ---- 1) 向量检索(一次性拿到所有类型的命中)----
        try:
            hits = retrieve_entities(file_id, question, top_k=top_k)
        except Exception as e:
            return f"<error>向量检索失败: {e}</error>"
        
        # 提前提取原始 hits 列表
        table_hits = hits.get("table", []) or []
        table_row_hits = hits.get("table_row", []) or []
        
        parts: list[str] = []
        citations: list[dict] = []
        rank_counter = 0
        
        # ---- 2) chunk 命中:段落原文 ----
        # ✨ 图谱展示:命中的 Chunk + 它们的父 Section
        if hits.get("text"):
            # 仅处理 chunk 类型,其他 text 实体忽略
            chunk_hits = [
                h for h in hits["text"]
                if h.get("entity_type") == "chunk"
            ]
            for i, h in enumerate(chunk_hits, 1):
                rank_counter += 1
                title = h.get("entity_name", "")
                content = (h.get("page_content", "") or "").strip()
                
                # ✨ 收集图谱节点:chunk 自身 + 它的父 section
                chunk_entity_id = h.get("entity_id")
                if chunk_entity_id:
                    graph_payload["chunk_ids"].append(chunk_entity_id)
                # 父 section 通过 section_path 反推不够安全(可能存的是 path),
                # 这里只收 chunk_id,父 section 的关系由前端 Cypher 的 OPTIONAL MATCH 补全
                # —— 但因为我们的 Cypher 限定"边的两端都在 nodes 集合里",
                # 单收 chunk 没法画出 Section→Chunk 这条边。所以也要把父 section 收进来。
                parent_section_path = h.get("section_path", "")
                if parent_section_path:
                    # 父 section 的 entity_id 通常就是它的 section_path(根据你的图谱结构推断)
                    # 例:chunk "10 检查员资格及职责" 的 section_path =
                    #    "第2章 涂装专业 > 第34节 ... > 10 检查员资格及职责"
                    # 父 section 是这个路径的前缀,但 chunk 自身又对应 section path 的末段...
                    # 为简单起见,父 section_id 改由前端 Cypher 自己 OPTIONAL MATCH 出去,
                    # 这里只确保父 section 的 id 出现在 section_ids 即可。
                    #
                    # 取 section_path 的前缀(去掉最末一段) → 父 Section 的 path
                    parts_path = parent_section_path.split(" > ")
                    if len(parts_path) >= 2:
                        parent_path = " > ".join(parts_path[:-1])
                        graph_payload["section_ids"].append(parent_path)
                
                parts.append(
                    f'<context_item type="text_chunk" index="{i}">\n'
                    f'  <title>{title}</title>\n'
                    f'  <content>\n{content}\n  </content>\n'
                    f'</context_item>'
                )
                
                citations.append({
                    "fileId": file_id,
                    "rank": rank_counter,
                    "type": "text_chunk",
                    "title": title,
                    "table_name": "",
                    "table_img_path": [],  # 统一为 list[str]
                    "markdown": content,
                })
        
        # ---- 3) table 命中:整 chunk + 同 chunk 内所有表 ----
        # ✨ 图谱展示:命中 Table + 它所在 Chunk + 该 Chunk 下所有 Table
        if hits.get("table"):
            try:
                table_ctxs = get_table_context_with_chunks(driver, hits)
            except Exception as e:
                table_ctxs = []
                parts.append(f"<error>表格上下文检索失败: {e}</error>")
            
            if table_ctxs:
                # 只在有结果时才计算 img_paths
                table_img_paths = list({
                    h.get("table_img_path", "")
                    for h in table_hits
                    if h.get("table_img_path")
                })
                
                for i, t in enumerate(table_ctxs, 1):
                    rank_counter += 1
                    chunk_title = t.get("chunk_title", "")
                    full_text = t.get("full_formatted_text", "") or ""
                    
                    # ✨ 收集图谱节点
                    # 注:get_table_context_with_chunks 的 raw_data 里有完整结构
                    raw = t.get("raw_data", {}) or {}
                    # 它 RETURN c.entity_name AS chunk_title,但我们需要 chunk 的 entity_id
                    # 如果 raw_data 里有 chunk_entity_id 字段更好,否则我们只能用 chunk_title
                    # 临时方案:Cypher 没返回 entity_id,这里 chunk_id 暂时跳过(由 Table 出发的
                    # CONTAINS_TABLE 反向边会被 OPTIONAL MATCH 补全)
                    # 但 table_names 必须收!
                    tables_data = raw.get("tables_data") or []
                    for tbl in tables_data:
                        tname = tbl.get("table_name")
                        if tname:
                            graph_payload["table_names"].append(tname)
                    
                    parts.append(
                        f'<context_item type="table_chunk" index="{i}">\n'
                        f'  <title>{chunk_title}</title>\n'
                        f'  <content>\n{full_text}\n  </content>\n'
                        f'</context_item>'
                    )
                    
                    citations.append({
                        "fileId": file_id,
                        "rank": rank_counter,
                        "type": "table_chunk",
                        "title": chunk_title,
                        "table_name": "",
                        "table_img_path": table_img_paths,  # list[str]
                        "markdown": full_text,
                    })
        
        # ---- 4) table_row 命中:回溯整张表 ----
        # ✨ 图谱展示:命中行所属的 Table + 该 Table 下所有 TableRow
        if hits.get("table_row"):
            try:
                full_tables = get_full_tables_from_hits(driver, hits)
            except Exception as e:
                full_tables = []
                parts.append(f"<error>表格行回溯失败: {e}</error>")
            
            if full_tables:
                # 按 parent_entity_name 聚合 img_path(同一张表的多行去重)
                img_path_by_table = {
                    row.get("parent_entity_name", ""): row.get("table_img_path", "")
                    for row in table_row_hits
                    if row.get("parent_entity_name") and row.get("table_img_path")
                }
                
                for i, t in enumerate(full_tables, 1):
                    rank_counter += 1
                    table_name = t.get("table_name", "")
                    full_content = t.get("full_text_content", "") or ""
                    img_path = img_path_by_table.get(table_name, "")
                    
                    # ✨ 收集图谱节点:Table + 该 Table 下的所有 TableRow
                    if table_name:
                        graph_payload["table_names"].append(table_name)
                    # full_rows 形如 [{index, text, row_id}, ...] (Cypher 已加 row_id 字段)
                    for row in t.get("full_rows", []) or []:
                        rid = row.get("row_id")
                        if rid:
                            graph_payload["row_ids"].append(rid)
                    
                    parts.append(
                        f'<context_item type="full_table_data" index="{i}">\n'
                        f'  <table_name>{table_name}</table_name>\n'
                        f'  <table_rows>\n{full_content}\n  </table_rows>\n'
                        f'</context_item>'
                    )
                    
                    citations.append({
                        "fileId": file_id,
                        "rank": rank_counter,
                        "type": "full_table_data",
                        "title": table_name,
                        "table_name": table_name,
                        "table_img_path": [img_path] if img_path else [],  # 统一为 list[str]
                        "markdown": full_content,
                    })
        
        # ---- 5) 全空 ----
        if not parts or not citations:
            last_call_state["citations"] = []
            last_call_state["graph_payload"] = _new_graph_payload()
            return (
                "<warning>未检索到任何相关的细节内容。"
                "如果问题涉及宏观规范或章节概要,请改用 summary_retrieval_tool。</warning>"
            )
        
        last_call_state["citations"] = citations
        last_call_state["graph_payload"] = _dedup_payload(graph_payload)
        return OUTPUT_RULE_HINT + "\n" + "\n".join(parts)
    
    def _get_last_citations():
        """返回最近一次工具调用产生的 citations 副本"""
        return list(last_call_state["citations"])
    
    def _get_last_graph_payload():
        """✨ 返回最近一次工具调用收集到的图谱节点 id 集合"""
        return dict(last_call_state.get("graph_payload") or {})
    
    def _clear_state():
        """清空闭包容器中的所有状态"""
        last_call_state["citations"] = []
        last_call_state["graph_payload"] = _new_graph_payload()
    
    object.__setattr__(detail_retrieval_tool, 'get_last_citations', _get_last_citations)
    object.__setattr__(detail_retrieval_tool, 'get_last_graph_payload', _get_last_graph_payload)
    object.__setattr__(detail_retrieval_tool, 'clear_state', _clear_state)
    
    return detail_retrieval_tool

# ============================================================
# 工具 2: 总结类检索 (含 level 自判定 + 自底向上摘要)
# ============================================================

def build_summary_retrieval_tool(
    file_id: str,
    driver: Any,
    llm: Any,
    top_k: int = 3,
    enable_summary_generation: bool = True,
    force_rebuild: bool = False,
    level_override: Optional[int] = None,
):
    """
    构造总结类检索工具。

    返回的工具额外暴露访问器:
        tool.get_last_citations()     -> list[dict]   最近一次调用的结构化引用数据
        tool.get_last_graph_payload() -> dict         最近一次调用涉及到的图谱节点 id 集合
        tool.clear_state()                            清空闭包状态
    """
    table_sum = table_summarizer(llm) if enable_summary_generation else None
    chunk_sum = chunk_summarizer(llm) if enable_summary_generation else None
    section_sum = section_summarizer(llm) if enable_summary_generation else None

    last_call_state: dict = {
        "citations": [],
        "graph_payload": _new_graph_payload(),
    }

    # 主分支输出规则:描述与实际 XML 对齐
    OUTPUT_RULE = (
        '<!-- 处理规则(必须遵守):\n'
        '  1. metadata.name 作小标题, metadata.path 作副标题\n'
        '  2. 内容输出: 优先用 <macro_summary> 标签内容原样输出(禁止改写);\n'
        '     若该标签不存在或为空, 改用 <raw_content> 标签内容原样输出;\n'
        '     若两标签均不存在, 输出"该章节暂无内容"\n'
        '  3. 多个 section_context 用 --- 分隔\n'
        '-->'
    )

    # fallback 分支输出规则:补强角色定义,与主分支风格统一
    FALLBACK_RULE = (
        '<!-- 处理规则(必须遵守):\n'
        '  下方 <context_item> 是相关内容片段,必须基于这些片段回答用户问题。\n'
        '  1. 用 title 作小标题, 根据content回答用户问题,\n'
        '  2. 多个 context_item 用 --- 分隔\n'
        '-->'
    )

    @tool(parse_docstring=False)
    def summary_retrieval_tool(question: str) -> str:
        """【章节概要必用】整篇章节的文字综述检索,回答"概述/总结/摘要"类问题。

        必用场景(只要属于以下任一,都用本工具):

        - 显式章节询问: "第3章主要讲了什么""3.2节的概要"
        - 主题+概要后缀: "XX工艺规范的概要""XX作业要求的概述""XX焊接工艺的要点"
          (用主题标题内容询问 + "概要/综述/总结/摘要" 等后缀,通常指对应章节/小节的宏观回答)

        不适用场景(请改用 detail_retrieval_tool):
        - 任何涉及"表/表格"的问题(无论问数值还是问概括)
          例: "总结表X的方法""X的标准范围""X包括什么类型"
        - 段落细节查询: "X的要求""Y的工艺流程""Z的适用范围"

        参数:
            question: 用户的章节/综述类问题
        """
        # 重置闭包状态
        last_call_state["citations"] = []
        last_call_state["graph_payload"] = _new_graph_payload()
        graph_payload = last_call_state["graph_payload"]

        # ---- 1) 决定层级 ----
        if level_override is not None:
            level = level_override
            print(f"[summary_retrieval_tool] 使用 level_override={level}")
        else:
            level = decide_level(question, llm=llm)

        # ---- 2) 向量检索 ----
        try:
            hits = retrieve_entities(file_id, question, top_k=top_k)
        except Exception as e:
            return f"<error>向量检索失败: {e}</error>"

        # ---- 3) 章节检索 ----
        try:
            results = get_section_context_with_summary(
                driver=driver,
                hits=hits,
                level=level,
                table_summarizer=table_sum,
                chunk_summarizer=chunk_sum,
                section_summarizer=section_sum,
                force_rebuild=force_rebuild,
            )
        except Exception as e:
            return f"<error>章节检索失败: {e}</error>"

        # ---- 4) 命中为空: fallback 到 chunk ----
        # ✨ 图谱展示:命中的 chunk + 它的父 section
        if not results:
            chunk_hits = [
                h for h in hits.get("text", [])
                if h.get("entity_type") == "chunk"
            ]
            if not chunk_hits:
                last_call_state["graph_payload"] = _new_graph_payload()
                return (
                    f'<warning level="{level}">'
                    f'未检索到任何相关章节内容。'
                    f'</warning>'
                )

            # fallback 分支:用统一风格的规则提示,语义中性的 fallback_notice 标签
            parts = [
                FALLBACK_RULE,
                f'<fallback_notice level="{level}">'
                f'未直接命中宏观章节,以下是相关内容片段:'
                f'</fallback_notice>'
            ]

            fallback_citations: list[dict] = []
            for i, h in enumerate(chunk_hits, 1):
                title = h.get("name", "") or h.get("entity_name", "")
                content = (h.get("content", "") or h.get("page_content", "") or "").strip()

                # ✨ 收集图谱节点
                chunk_entity_id = h.get("entity_id")
                if chunk_entity_id:
                    graph_payload["chunk_ids"].append(chunk_entity_id)
                # 父 section
                section_path = h.get("section_path", "")
                if section_path:
                    parts_path = section_path.split(" > ")
                    if len(parts_path) >= 2:
                        graph_payload["section_ids"].append(" > ".join(parts_path[:-1]))

                parts.append(
                    f'<context_item type="fallback_chunk" index="{i}">\n'
                    f'  <title>{title}</title>\n'
                    f'  <content>{content}</content>\n'
                    f'</context_item>'
                )

                fallback_citations.append({
                    "fileId": file_id,
                    "rank": i,
                    "level": level,  # 记录原本想检索的层级
                    "type": "fallback_chunk",
                    "section_name": title,
                    "section_path": "",
                    "structure_info": "",
                    "summary": "",
                    "markdown": content,
                })

            last_call_state["citations"] = fallback_citations
            last_call_state["graph_payload"] = _dedup_payload(graph_payload)
            return "\n".join(parts)

        # ---- 5) 主分支:拼接输出 ----
        # ✨ 图谱展示:命中 target_section + 子树所有 Section + 所有 Chunk
        parts: list[str] = [OUTPUT_RULE]
        citations: list[dict] = []

        for i, r in enumerate(results, 1):
            target = r.get("target_section", {}) or {}
            stats = r.get("subtree_stats", {}) or {}

            section_name = target.get("entity_name", "(未命名)")
            section_path = target.get("section_path", "")
            sub_sections = stats.get("sections", 0)
            sub_chunks = stats.get("chunks", 0)
            structure_info = f"包含 {sub_sections} 个子章节,{sub_chunks} 个文本块"
            summary_text = r.get("summary") or ""
            markdown_text = r.get("markdown") or ""

            # ✨ 收集图谱节点:子树所有 section + 所有 chunk
            # r 里还需要带出 subtree 数据。这要求 graph_search_section.get_section_context_with_summary
            # 在 results.append() 时把 subtree 也带出来。
            # 如果没带出来,降级:只收 target_section 这一个节点。
            subtree_data = r.get("subtree") or {}
            subtree_sections = subtree_data.get("sections", [])
            subtree_chunks_map = subtree_data.get("chunks", {})

            if subtree_sections:
                for sec in subtree_sections:
                    sid = sec.get("entity_id")
                    if sid:
                        graph_payload["section_ids"].append(sid)
            else:
                # 降级:至少把 target 本身加进去
                tid = target.get("entity_id")
                if tid:
                    graph_payload["section_ids"].append(tid)

            if subtree_chunks_map:
                for sec_id, chunks_list in subtree_chunks_map.items():
                    for ch in chunks_list:
                        cid = ch.get("entity_id")
                        if cid:
                            graph_payload["chunk_ids"].append(cid)

            # --- XML 拼接 ---
            parts.append(f'<section_context index="{i}" level="{level}">')
            parts.append('  <metadata>')
            parts.append(f'    <name>{section_name}</name>')
            parts.append(f'    <path>{section_path}</path>')
            parts.append(f'    <structure_info>{structure_info}</structure_info>')
            parts.append('  </metadata>')

            if summary_text:
                parts.append(f'  <macro_summary>\n{summary_text}\n  </macro_summary>')

            if markdown_text:
                parts.append(f'  <raw_content>\n{markdown_text}\n  </raw_content>')

            parts.append('</section_context>')

            # --- 同步构建 citation ---
            citations.append({
                "fileId": file_id,
                "rank": i,
                "level": level,
                "type": "section",
                "section_name": section_name,
                "section_path": section_path,
                "sub_sections": sub_sections,
                "sub_chunks": sub_chunks,
                "structure_info": structure_info,
                "summary": summary_text,
                "markdown": markdown_text,
            })

        last_call_state["citations"] = citations
        last_call_state["graph_payload"] = _dedup_payload(graph_payload)
        return "\n".join(parts)

    def _get_last_citations():
        """返回最近一次工具调用产生的 citations 副本"""
        return list(last_call_state["citations"])

    def _get_last_graph_payload():
        """✨ 返回最近一次工具调用收集到的图谱节点 id 集合"""
        return dict(last_call_state.get("graph_payload") or {})

    def _clear_state():
        """清空闭包容器中的所有状态"""
        last_call_state["citations"] = []
        last_call_state["graph_payload"] = _new_graph_payload()

    object.__setattr__(summary_retrieval_tool, 'get_last_citations', _get_last_citations)
    object.__setattr__(summary_retrieval_tool, 'get_last_graph_payload', _get_last_graph_payload)
    object.__setattr__(summary_retrieval_tool, 'clear_state', _clear_state)

    return summary_retrieval_tool