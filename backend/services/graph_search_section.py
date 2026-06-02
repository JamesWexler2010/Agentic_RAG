"""
graph_search_section.py

基于 Neo4j 图谱的 Section 检索 + 自底向上摘要生成模块

功能:
1. 路径截取定位(locate_target_section)
2. 子树遍历 + 多模态内容获取(fetch_subtree_with_content)
3. Chunk → Markdown 格式化(chunk_to_markdown)
4. 自底向上摘要生成(build_summaries_bottom_up)
5. 主流程入口(get_section_context_with_summary)
"""
from typing import Dict, List, Any, Optional, Callable, TypedDict

from neo4j import Driver


# ============================================================
# 类型定义(便于 IDE 提示)
# ============================================================

class SectionDict(TypedDict, total=False):
    """Section 元数据"""
    entity_id: str
    entity_name: str
    section_path: str
    depth: int
    parent_id: Optional[str]
    summary: Optional[str]


class ChunkDict(TypedDict, total=False):
    """Chunk 元数据 + 多模态内容"""
    entity_id: str
    title: str
    content: str
    section_path: str
    summary: Optional[str]
    images: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]


class SubtreeDict(TypedDict):
    """子树查询结果"""
    sections: List[SectionDict]
    chunks: Dict[str, List[ChunkDict]]


# 类型别名：摘要器函数签名
# 接收: table_name(str), table_rows(str) -> 返回: 摘要(str)
TableSummarizer = Callable[[str, str], str]
# 接收: chunk_markdown(str) -> 返回: 摘要(str)
ChunkSummarizer = Callable[[str], str]
# 接收: child_summaries(List[str]), section_name(str) -> 返回: 摘要(str)
SectionSummarizer = Callable[[List[str], str], str]


# 允许写入摘要的节点标签(防注入白名单)
_ALLOWED_LABELS = {"Chunk", "Section"}


# ============================================================
# [新增] summary 清洗工具
# ============================================================

# 视为"假空值"的字符串(全小写比较)
_EMPTY_SUMMARY_TOKENS = frozenset({"none", "null", "nan", "n/a", "na", "undefined"})


def _normalize_summary(value: Any) -> Optional[str]:
    """
    把 summary 字段统一规整成 None 或有效的非空字符串。
    
    用于堵住所有"假空值":
      - None / Neo4j NULL                      → None
      - 非字符串对象(sqlalchemy.null 等)       → None
      - 空字符串 "" / 纯空白 "   "             → None
      - 字符串字面量 "None"/"null"/"nan"/"N/A" → None
      - 有效字符串                              → 返回 strip() 后的版本
    
    这样下游所有 `if summary:` 或 `if isinstance(s, str) and s.strip():` 
    判断都能正确工作。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in _EMPTY_SUMMARY_TOKENS:
        return None
    return stripped


def _clean_section_dict(sec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """对一个 section dict 的 summary 字段做清洗(原地)。"""
    if sec is None:
        return None
    if "summary" in sec:
        sec["summary"] = _normalize_summary(sec["summary"])
    return sec


def _clean_chunk_dict(chunk: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """对一个 chunk dict 的 summary 字段做清洗(原地)。"""
    if chunk is None:
        return None
    if "summary" in chunk:
        chunk["summary"] = _normalize_summary(chunk["summary"])
    return chunk


# ============================================================
# 模块 1:路径截取定位
# ============================================================

def locate_target_section(
    driver: Driver, 
    hit_entity_id: str, 
    level: int
) -> Optional[SectionDict]:
    """
    根据命中 section 的 entity_id 和目标层级 level,
    截取 section_path 定位上层 section。

    支持 level 范围:1-6
        level=1  最顶层(章)
        level=2  节/附录
        level=3  节内一级
        level=4  x.x
        level=5  x.x.x
        level=6  x.x.x.x
        
        - level >= 实际路径段数 时返回命中节点本身
        - level <= 0 时返回最顶层
        - 算法不依赖具体取值范围,只按 section_path 截取前 level 段。
          扩展到更多层级时本函数无需修改。

    参数:
        driver: Neo4j 连接
        hit_entity_id: 命中节点的 entity_id
        level: 目标层级(1-6)

    返回:
        定位到的 Section 字典;找不到时返回 None。
        找不到通常意味着 hit_entity_id 对应的不是 Section 节点(可能是 Chunk)。
    """
    cypher = """
    MATCH (hit:Section {entity_id: $hit_entity_id})
    WITH hit, split(hit.section_path, ' > ') AS parts
    WITH hit, parts,
         CASE 
           WHEN $level >= size(parts) THEN hit.section_path
           WHEN $level <= 0 THEN parts[0]
           ELSE reduce(s = head(parts), x IN parts[1..$level] | s + ' > ' + x)
         END AS target_path
    MATCH (target:Section {section_path: target_path})
    RETURN target.entity_id   AS entity_id,
           target.entity_name AS entity_name,
           target.section_path AS section_path,
           target.depth       AS depth,
           target.summary     AS summary
    """
    with driver.session() as session:
        record = session.run(
            cypher, 
            hit_entity_id=hit_entity_id, 
            level=level
        ).single()
        if not record:
            return None
        # [改] 出口处清洗 summary
        return _clean_section_dict(dict(record))


# [新增] 按 entity_id 直接取 Section 本身——配合 level=None 分支使用
def _fetch_section_by_id(
    driver: Driver,
    entity_id: str,
) -> Optional[SectionDict]:
    """
    根据 entity_id 直接取出 Section 节点本身,**不做任何 lift**。
    
    与 locate_target_section 的区别:
      - locate_target_section: 按 level 截取 section_path → 往上找祖先
      - _fetch_section_by_id:  命中节点是什么就返回什么(用户没指定层级时使用)
    
    用途:
      level_detector.decide_level() 返回 None 时,
      get_section_context_with_summary() 走"命中节点就是 target"分支,
      此函数负责把 hit_id 包装成同样的 SectionDict 结构。
    
    返回:
        Section 节点对应的字典,找不到时返回 None。
        (如果节点存在但 label 不是 Section,例如是 Chunk,也返回 None;
         这是预期行为,因为后续 fetch_subtree_with_content 需要 Section 作为子树根)
    """
    cypher = """
    MATCH (s:Section {entity_id: $entity_id})
    RETURN s.entity_id   AS entity_id,
           s.entity_name AS entity_name,
           s.section_path AS section_path,
           s.depth       AS depth,
           s.summary     AS summary
    """
    with driver.session() as session:
        record = session.run(cypher, entity_id=entity_id).single()
        if not record:
            return None
        return _clean_section_dict(dict(record))


# ============================================================
# 模块 2:子树遍历 + 多模态内容获取
# ============================================================

def fetch_subtree_with_content(
    driver: Driver, 
    root_entity_id: str
) -> SubtreeDict:
    """
    从根 Section 出发，获取整棵子树的：
    - 所有 Section 节点（含层级关系，用于自底向上摘要）
    - 所有叶 Chunk 及其多模态内容（图片、表格）

    返回结构:
        {
            "sections": [{entity_id, entity_name, section_path, 
                          depth, parent_id, summary}, ...],
            "chunks": {section_entity_id: [chunk_dict, ...]}
        }
    """
    cypher = """
    // Step 1: 获取子树所有 Section(含父子关系)
    MATCH (root:Section {entity_id: $root_id})
    OPTIONAL MATCH (root)-[:HAS_CHILD*0..]->(sub:Section)
    WITH collect(DISTINCT sub) + collect(DISTINCT root) AS all_sections_raw
    UNWIND all_sections_raw AS sec
    WITH DISTINCT sec
    WHERE sec IS NOT NULL
    OPTIONAL MATCH (parent:Section)-[:HAS_CHILD]->(sec)
    WITH collect({
        entity_id: sec.entity_id,
        entity_name: sec.entity_name,
        section_path: sec.section_path,
        depth: sec.depth,
        parent_id: parent.entity_id,
        summary: sec.summary
    }) AS sections
    
    // Step 2: 获取每个 Section 下的 Chunk 及多模态内容
    UNWIND sections AS s
    OPTIONAL MATCH (sec:Section {entity_id: s.entity_id})-[:HAS_CHUNK]->(c:Chunk)
    
    // 图片
    OPTIONAL MATCH (c)-[:CONTAINS_IMAGE]->(img:Image)
    WITH sections, s, c, collect(DISTINCT {
        img_key: img.img_key, 
        img_paths: img.img_paths,
        name: img.entity_name
    }) AS images
    
    // 表格 + 表格行
    OPTIONAL MATCH (c)-[:CONTAINS_TABLE]->(t:Table)
    OPTIONAL MATCH (t)-[:HAS_ROW]->(tr:TableRow)
    WITH sections, s, c, images, t, tr
    ORDER BY t.entity_name, tr.row_index ASC
    WITH sections, s, c, images, t, 
         collect(CASE WHEN tr IS NOT NULL 
                 THEN {idx: tr.row_index, txt: tr.row_text} 
                 ELSE null END) AS rows_raw
    WITH sections, s, c, images,
         collect(CASE WHEN t IS NOT NULL THEN {
             name: t.entity_name,
             table_path: t.table_path,
             html: t.table_body,
             img_path: t.table_img_path,
             rows: [r IN rows_raw WHERE r IS NOT NULL]
         } ELSE null END) AS tables_raw
    
    WITH sections, s, 
         collect(CASE WHEN c IS NOT NULL THEN {
             entity_id: c.entity_id,
             title: c.entity_name,
             content: c.content,
             section_path: c.section_path,
             summary: c.summary,
             images: [img IN images WHERE img.img_key IS NOT NULL],
             tables: [tbl IN tables_raw WHERE tbl IS NOT NULL]
         } ELSE null END) AS chunks_raw
    
    WITH sections, s, [ch IN chunks_raw WHERE ch IS NOT NULL] AS chunks
    
    RETURN sections AS all_sections, 
           collect({section_id: s.entity_id, chunks: chunks}) AS section_chunks
    """
    
    with driver.session() as session:
        record = session.run(cypher, root_id=root_entity_id).single()
        if not record:
            return {"sections": [], "chunks": {}}
        
        # [改] 出口处清洗所有 section 的 summary
        all_sections = record["all_sections"] or []
        for sec in all_sections:
            _clean_section_dict(sec)
        
        # [改] 出口处清洗所有 chunk 的 summary
        chunks_map: Dict[str, List[ChunkDict]] = {}
        for sc in record["section_chunks"]:
            if sc["chunks"]:
                for chunk in sc["chunks"]:
                    _clean_chunk_dict(chunk)
                chunks_map[sc["section_id"]] = sc["chunks"]
        
        return {
            "sections": all_sections,
            "chunks": chunks_map
        }


# ============================================================
# 模块 3:Chunk → Markdown 格式化
# ============================================================

def chunk_to_markdown(chunk: ChunkDict) -> str:
    """将单个 chunk 及其多模态内容转为 Markdown 格式"""
    blocks = []
    
    # 1. 基础信息组装
    title = chunk.get('title', '').strip()
    section_path = chunk.get('section_path', '').strip()
    
    meta_lines = []
    if title:
        meta_lines.append(f"**内容块名称**:{title}")
    if section_path:
        meta_lines.append(f"**章节路径**:{section_path}")
        
    if meta_lines:
        blocks.append("\n".join(meta_lines))
    
    # 2. 正文内容处理
    content = chunk.get('content', '').strip()
    if content:
        blocks.append(content)
        
    # 3. 相关图片处理
    images = chunk.get('images') or []
    if images:
        img_lines = ["**相关图片**:"]
        for img in images:
            img_name = img.get('name', '未命名图片').strip()
            img_key = img.get('img_key', '').strip()
            # 只有当 img_key 存在时才生成图片链接
            if img_key:
                img_lines.append(f"- ![{img_name}]({img_key})")
        if len(img_lines) > 1:
            blocks.append("\n".join(img_lines))
            
    # 4. 相关表格处理
    tables = chunk.get('tables') or []
    for tbl in tables:
        tbl_name = tbl.get('name', '未命名表格').strip()
        tbl_blocks = [f"**表格:{tbl_name}**"]
        
        # 优先级 1: 使用自然语言格式的行列表
        if tbl.get('rows'):
            # 提取非空的文本行
            row_lines = [
                f"- {r.get('txt', '').strip()}" 
                for r in tbl['rows'] 
                if isinstance(r, dict) and r.get('txt', '').strip()
            ]
            if row_lines:
                tbl_blocks.append("\n".join(row_lines))
                
        # 优先级 2: 如果没有任何行数据，使用 HTML 兜底
        elif tbl.get('html'):
            html_content = tbl['html'].strip()
            if html_content:
                tbl_blocks.append(html_content)
                
        # 将单个表格的标题和内容合并，作为一个整体区块
        if len(tbl_blocks) > 1:
            blocks.append("\n\n".join(tbl_blocks))
            
    # 统一使用双换行符组装当前 chunk 的所有信息
    return "\n\n".join(blocks)


# ============================================================
# 模块 4:自底向上生成摘要并写回
# ============================================================

def write_summary(
    driver: Driver, 
    label: str, 
    entity_id: str, 
    summary: str
) -> bool:
    """
    将摘要写回 Section 或 Chunk 节点
    
    参数:
        driver: Neo4j 连接
        label: "Chunk" 或 "Section"（防注入白名单）
        entity_id: 节点的唯一标识
        summary: 摘要文本
    
    返回:
        True 表示写入成功，False 表示节点不存在或失败
    
    异常:
        ValueError: 如果 label 不在白名单中
    """
    # 防注入：label 必须在白名单
    if label not in _ALLOWED_LABELS:
        raise ValueError(
            f"label 必须是 {_ALLOWED_LABELS} 之一,实际为 {label!r}"
        )
    
    # [改] 写入前清洗,确保不会把 "None" 字符串等脏数据写进图谱
    clean_summary = _normalize_summary(summary)
    if clean_summary is None:
        print(f"⚠️  拒绝写入空摘要: {label}({entity_id})")
        return False
    
    cypher = f"""
    MATCH (n:{label} {{entity_id: $entity_id}})
    SET n.summary = $summary,
        n.summary_updated_at = datetime()
    RETURN n.entity_id AS id
    """
    try:
        with driver.session() as session:
            result = session.run(
                cypher, 
                entity_id=entity_id, 
                summary=clean_summary
            ).single()
            if result is None:
                print(f"⚠️  节点不存在: {label}({entity_id})")
                return False
            return True
    except Exception as e:
        print(f"❌ 写入失败 {label}({entity_id}): {e}")
        return False


def build_summaries_bottom_up(
    driver: Driver,
    subtree: SubtreeDict,
    table_summarizer: TableSummarizer,
    chunk_summarizer: ChunkSummarizer,
    section_summarizer: SectionSummarizer,
    force_rebuild: bool = False
) -> Dict[str, str]:
    """
    自底向上生成摘要:
    1. 先为每个 Chunk 生成摘要(基于纯净上下文)
    2. 再按 depth 倒序,为每个 Section 聚合其子 Section 摘要 + 直属 Chunk 摘要
    """
    sections = subtree["sections"]
    chunks_map = subtree["chunks"]
    summary_cache: Dict[str, str] = {}
    
    # 预建 entity_id → entity_name 映射
    id_to_name: Dict[str, str] = {
        s["entity_id"]: s.get("entity_name", "") for s in sections
    }
    
    # ============ Step 1: 为所有 Chunk 生成摘要(叶节点) ============
    for chunks in chunks_map.values():
        for chunk in chunks:
            cid = chunk["entity_id"]
            
            # [改] 经过 fetch_subtree_with_content 的清洗后,这里 summary 必为 str 或 None
            existing = chunk.get("summary")
            if existing and not force_rebuild:
                summary_cache[cid] = existing
                print(f"[Chunk 跳过] {chunk.get('title', cid)} 已有摘要")
                continue
            
            # 生成新摘要
            try:
                # 构建专供摘要使用的“纯净上下文”
                summary_context = [f"**内容块名称**：{chunk.get('title', '')}"]
                
                # 1. 拼接正文
                if chunk.get('content'):
                    summary_context.append(chunk['content'])
                
                # 2. 处理图片：剥离 URL，仅保留名称提示大模型这里有图,如果不是要URL，没有必要再次显示一边图名称。
                
                # 3. 处理表格：调用专属的 table_summarizer
                for tbl in chunk.get('tables', []):
                    tbl_name = tbl.get('name', '未命名表格')
                    if tbl.get('rows'):
                        # 提取纯文本行
                        tbl_rows_text = "\n".join([f"- {r['txt']}" for r in tbl['rows']])
                        
                        # 【核心修改】：使用专属表格摘要器提炼核心规范和极值
                        tbl_summary = table_summarizer(tbl_name, tbl_rows_text) 
                        summary_context.append(f"** {tbl_name} 的核心内容摘要**:\n{tbl_summary}")
                    elif tbl.get('html'):
                        summary_context.append(f"**包含表格**:{tbl_name} (详细数据见原文)")
                
                # 4. 最终合并：正文 + 图片名 + 表格摘要
                final_md_for_summary = "\n\n".join(summary_context)
                
                # 生成整个 Chunk 的最终摘要
                summary = chunk_summarizer(final_md_for_summary) 
                
                # [改] 生成的 summary 也清洗一遍再处理
                summary = _normalize_summary(summary)
                if summary is None:
                    print(f"⚠️  [Chunk 生成空摘要] {chunk.get('title', cid)}")
                    continue
                
                if write_summary(driver, "Chunk", cid, summary):
                    summary_cache[cid] = summary
                    print(f"✅ [Chunk 生成] {chunk.get('title', cid)}")
                else:
                    print(f"⚠️  [Chunk 写入失败] {chunk.get('title', cid)}")
            except Exception as e:
                print(f"❌ [Chunk 异常] {chunk.get('title', cid)}: {e}")
    #print(f"{list(summary_cache.values())}")  # 输出已生成摘要的 Chunk ID 列表
    # ============ Step 2: 按 depth 倒序为 Section 生成摘要 ============
    children_map: Dict[str, List[str]] = {}
    for sec in sections:
        pid = sec.get("parent_id")
        if pid:
            children_map.setdefault(pid, []).append(sec["entity_id"])
    
    sorted_sections = sorted(
        sections, 
        key=lambda x: x.get("depth", 0), 
        reverse=True
    )
    
    for sec in sorted_sections:
        sid = sec["entity_id"]
        sec_name = sec.get("entity_name", sid)
        
        # [改] 经过 fetch_subtree_with_content 的清洗,这里直接 truthy 就够了
        existing = sec.get("summary")
        if existing and not force_rebuild:
            summary_cache[sid] = existing
            print(f"⏭️  [Section 跳过] {sec_name} 已有摘要")
            continue
        
        child_summaries: List[str] = []
        for chunk in chunks_map.get(sid, []):
            cs = summary_cache.get(chunk["entity_id"])
            if cs:
                child_summaries.append(f"[内容块: {chunk.get('title', '')}] {cs}")
        
        for child_sid in children_map.get(sid, []):
            cs = summary_cache.get(child_sid)
            if cs:
                child_name = id_to_name.get(child_sid, "")
                child_summaries.append(f"[子章节: {child_name}] {cs}")
        
        if not child_summaries:
            print(f"⚠️  [Section 无内容] {sec_name} 跳过")
            continue
        
        try:
            summary = section_summarizer(child_summaries, sec_name)
            # [改] 生成的 summary 清洗
            summary = _normalize_summary(summary)
            if summary is None:
                print(f"⚠️  [Section 生成空摘要] {sec_name}")
                continue
            
            if write_summary(driver, "Section", sid, summary):
                summary_cache[sid] = summary
                print(f"✅ [Section 生成] {sec_name} (depth={sec.get('depth')})")
            else:
                print(f"⚠️  [Section 写入失败] {sec_name}")
        except Exception as e:
            print(f"❌ [Section 异常] {sec_name}: {e}")
    
    return summary_cache


# ============================================================
# 模块 5:主流程入口
# ============================================================

def get_section_context_with_summary(
    driver: Driver,
    hits: Dict[str, List[Dict[str, Any]]],
    level: Optional[int],   # ✨ int → Optional[int]: None 表示"用命中节点本身,不 lift"
    table_summarizer: Optional[TableSummarizer] = None,
    chunk_summarizer: Optional[ChunkSummarizer] = None,
    section_summarizer: Optional[SectionSummarizer] = None,
    force_rebuild: bool = False
) -> List[Dict[str, Any]]:
    """
    主流程：
    1. 从 hits 提取 section 命中
    2. ✨ 根据 level 决定 target:
       - level 是 int → 调 locate_target_section 按层级 lift(原行为)
       - level 是 None → 命中节点本身就是 target,不做 lift
                         (用户未在问题里指定具体层级时由 decide_level 返回 None)
    3. 子树遍历 + 多模态扩展，生成 Markdown
    4. 检查摘要是否存在，按需自底向上生成（含表格独立提取）

    参数:
        driver: Neo4j 连接
        hits: 检索命中结果，格式 {"text": [...], "section": [...]}
        level: 目标层级。
            - 1/2/3/4: 按层级往上 lift(适用于"问第3章/第N节"这类明确指定层级的问题)
            - None:    不 lift,命中节点本身就是 target
                       (适用于"问平面度要求"这类只问内容、未指定层级的问题)
        table_summarizer/chunk_summarizer/section_summarizer: 大模型调用函数
            （若其中任意一个为 None，则仅获取上下文，不触发摘要生成逻辑）
        force_rebuild: 是否强制重建已有摘要
    
    返回:
        每个命中且去重后的 target section 对应一个结果字典：
        [{
            "hit_section_id": str,
            "target_section": SectionDict,
            "markdown": str,
            "summary": Optional[str],
            "subtree_stats": {"sections": int, "chunks": int}
        }, ...]
    """
    # ============ Step 1: 提取 section 命中 ============
    all_text_hits = hits.get("text", [])
    section_hits = [
        item for item in all_text_hits 
        if item.get("entity_type") == "section"
    ]
    section_hits.extend(hits.get("section", []))
    
    # 获取原始命中的 Section ID 列表
    section_ids = list(dict.fromkeys(
            s.get("entity_id") for s in section_hits if s.get("entity_id")
    ))
    print(f"[1/5] 原始命中 Section: {len(section_ids)} 个")
    
    if not section_ids:
        return []
    
    results = []
    
    # [优化]：用于记录已经处理过的上层 Target 节点，防止重复生成
    processed_target_ids = set()
    
    for hit_id in section_ids:
        # ============ Step 2: 根据 level 决定 target ============
        # ✨ level 为 None: 命中节点本身就是 target,不做 lift
        # ✨ level 是 int: 走原 locate_target_section 按层级 lift
        if level is None:
            target = _fetch_section_by_id(driver, hit_id)
            if not target:
                # 命中的 entity_id 不是 Section(可能是 Chunk),按预期跳过
                # —— 这种 hit 在原始流程里也会被 locate_target_section 卡掉
                print(f"[2/5] {hit_id} 不是 Section(可能是 Chunk),跳过")
                continue
            print(
                f"[2/5] {hit_id} → 直接使用命中节点(无 lift): "
                f"{target['section_path']} (depth={target['depth']})"
            )
        else:
            # locate_target_section 内部已经清洗过 summary,这里拿到的是干净数据
            target = locate_target_section(driver, hit_id, level)
            if not target:
                print(f"[2/5] 无法为 {hit_id} 定位 level={level} 的上层节点")
                continue
            print(
                f"[2/5] {hit_id} → 定位到: {target['section_path']} "
                f"(depth={target['depth']})"
            )

        target_id = target["entity_id"]
        
        # [优化]：如果多个底层命中点指向了同一个父节点（例如命中了 1.1 和 1.2，目标都是第 1 章），直接跳过
        if target_id in processed_target_ids:
            print(f"[2/5] 归属于已处理过的 target: {target['section_path']},跳过重复处理")
            continue
            
        processed_target_ids.add(target_id)
        
        # ============ Step 3: 获取子树 + 多模态内容 ============
        # fetch_subtree_with_content 内部已经清洗过所有 section/chunk 的 summary
        subtree = fetch_subtree_with_content(driver, target_id)
        section_count = len(subtree["sections"])
        chunk_count = sum(len(v) for v in subtree["chunks"].values())
        print(
            f"[3/5] 子树: {section_count} 个 Section, "
            f"{chunk_count} 个 Chunk"
        )
        
        # ============ Step 4: 拼接 Markdown ============
        markdown_content = _build_full_markdown(target, subtree)
        print(f"[4/5] Markdown 已拼接 ({len(markdown_content)} 字符)")
        
        # ============ Step 5: 摘要逻辑 ============
        # [改] target 已经清洗过,这里 target_summary 必为 str(非空) 或 None
        target_summary = target.get("summary")
        
        # [修改]：必须确保三个 Summarizer 都传入了，才启动自底向上生成逻辑
        if (not target_summary or force_rebuild) and table_summarizer and chunk_summarizer and section_summarizer:
            print(f"[5/5] 触发摘要生成(自底向上)...")
            summary_cache = build_summaries_bottom_up(
                driver=driver, 
                subtree=subtree, 
                table_summarizer=table_summarizer,
                chunk_summarizer=chunk_summarizer, 
                section_summarizer=section_summarizer, 
                force_rebuild=force_rebuild
            )
            target_summary = summary_cache.get(target_id)
        elif target_summary:
            print(f"[5/5] 命中已有摘要,直接复用")
        else:
            print(f"[5/5] 跳过摘要(未提供完整的 summarizer 组件)")
        
        results.append({
            "hit_section_id": hit_id,
            "target_section": target,
            "markdown": markdown_content,
            "summary": target_summary,
            "subtree_stats": {
                "sections": section_count,
                "chunks": chunk_count
            },
            "subtree": subtree,   # ✨ 让 tool.py 能从这里取所有 section/chunk 的 entity_id
        })
    
    return results


def _build_full_markdown(
    target: SectionDict, 
    subtree: SubtreeDict
) -> str:
    """构造完整子树的 Markdown 文档(内部辅助函数)"""
    blocks = []
    
    # 1. 顶部标题和路径
    entity_name = target.get('entity_name', '未命名节点').strip()
    section_path = target.get('section_path', '未知路径').strip()
    blocks.append(f"**{entity_name}**\n\n**当前节点路径**:{section_path}")
    blocks.append("---")
    
    # 2. 遍历 Chunks
    # 使用 subtree.get() 增加容错，避免键不存在时报错
    for chunks in subtree.get("chunks", {}).values():
        for chunk in chunks:
            chunk_md = chunk_to_markdown(chunk)
            if chunk_md:  # 确保只有非空的 chunk 才被加入
                blocks.append(chunk_md)
                blocks.append("---")
    
    # 3. 移除末尾多余的分割线
    if blocks and blocks[-1] == "---":
        blocks.pop()
        
    # 统一使用双换行符隔开各个区块，这是 Markdown 最标准的排版方式
    return "\n\n".join(blocks)