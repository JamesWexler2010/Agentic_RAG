"""
知识图谱构建

节点唯一约束：
  - Section   : entity_id（完整 section_path）
  - Chunk     : entity_id
  - Image     : img_key（取 img_paths[0] 作为代理键，img_paths 数组作为属性保留）
  - Table     : table_img_path
  - TableRow  : table_img_path + "::" + row_index
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase

# ── 正则 ─────────────────────────────────────────────────────────
IMAGE_RE = re.compile(r"\[\[图片：(.+?)\]\]")
TABLE_RE = re.compile(r"\[\[表格：(.+?)\]\]")


# ════════════════════════════════════════════════════════════════════
# 数据加载与预处理
# ════════════════════════════════════════════════════════════════════

def load_entities(file_id: str) -> dict:
    base = Path("data") / file_id
    text_entities = json.loads((base / "text_entities.json").read_text(encoding="utf-8"))

    sections = [e for e in text_entities if e["entity_type"] == "section"]
    chunks   = [e for e in text_entities if e["entity_type"] == "chunk"]

    # 找出与 chunk 同路径且 content 为空的叶子 section，后续跳过这些节点
    # 此类 section 的内容完全由对应 chunk 承载，保留 section 节点无实际价值
    chunk_paths = {ck["section_path"] for ck in chunks}
    merged_section_ids = {
        sec["entity_id"]
        for sec in sections
        if sec["entity_id"] in chunk_paths
        and sec.get("content", "") == ""
    }

    filtered_sections = [s for s in sections if s["entity_id"] not in merged_section_ids]

    print(f"   叶子Section合并：{len(merged_section_ids)} 个叶子Section节点被跳过，对应Chunk直接挂载到父Section")

    return {
        "sections":           filtered_sections,
        "chunks":             chunks,
        "merged_section_ids": merged_section_ids,
        "images":  json.loads((base / "image_entities.json").read_text(encoding="utf-8")),
        "tables":  json.loads((base / "table_entities.json").read_text(encoding="utf-8")),
        "rows":    json.loads((base / "table_sub_entities.json").read_text(encoding="utf-8")),
    }


# ════════════════════════════════════════════════════════════════════
# 初始化约束与索引
# ════════════════════════════════════════════════════════════════════

def init_constraints(session):
    """幂等：IF NOT EXISTS 保证重复执行安全"""
    ddls = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Section)  REQUIRE n.entity_id      IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk)    REQUIRE n.entity_id      IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Image)    REQUIRE n.img_key        IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Table)    REQUIRE n.table_img_path IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:TableRow) REQUIRE n.row_id         IS UNIQUE",
        # 普通索引，加速 MATCH
        "CREATE INDEX IF NOT EXISTS FOR (n:Image)    ON (n.entity_name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:Table)    ON (n.entity_name)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TableRow) ON (n.table_img_path)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TableRow) ON (n.row_index)",
    ]
    for ddl in ddls:
        session.run(ddl)


# ════════════════════════════════════════════════════════════════════
# 1. Section 树
# ════════════════════════════════════════════════════════════════════

def build_section_tree(session, sections: list):
    # upsert 节点
    session.run("""
        UNWIND $rows AS row
        MERGE (n:Section {entity_id: row.entity_id})
        SET n.entity_name  = row.entity_name,
            n.depth        = row.depth,
            n.section_path = row.section_path
    """, rows=sections)

    # 按 section_path 推断父子关系，建 HAS_CHILD 边
    edges = []
    for sec in sections:
        parts = sec["section_path"].split(" > ")
        if len(parts) > 1:
            edges.append({
                "parent_id": " > ".join(parts[:-1]),
                "child_id":  sec["entity_id"],
            })
    if edges:
        session.run("""
            UNWIND $edges AS e
            MATCH (p:Section {entity_id: e.parent_id})
            MATCH (c:Section {entity_id: e.child_id})
            MERGE (p)-[:HAS_CHILD]->(c)
        """, edges=edges)


# ════════════════════════════════════════════════════════════════════
# 2. Chunk
# ════════════════════════════════════════════════════════════════════

def build_chunks(session, chunks: list, merged_section_ids: set):
    # upsert 节点
    session.run("""
        UNWIND $rows AS row
        MERGE (n:Chunk {entity_id: row.entity_id})
        SET n.entity_name  = row.entity_name,
            n.content      = row.content,
            n.depth        = row.depth,
            n.section_path = row.section_path
    """, rows=chunks)

    # 建 Section -> Chunk 的 HAS_CHUNK 边
    for ck in chunks:
        path = ck["section_path"]

        if path in merged_section_ids:
            # 叶子 Section 已被跳过，直接连接到父 Section
            parts = path.split(" > ")
            if len(parts) > 1:
                parent_id = " > ".join(parts[:-1])
                session.run("""
                    MATCH (sec:Section {entity_id: $parent_id})
                    MATCH (ck:Chunk    {entity_id: $chunk_id})
                    MERGE (sec)-[:HAS_CHUNK]->(ck)
                """, parent_id=parent_id, chunk_id=ck["entity_id"])
        else:
            # 普通情况：连接到对应 Section
            session.run("""
                MATCH (sec:Section {entity_id: $section_path})
                MATCH (ck:Chunk    {entity_id: $chunk_id})
                MERGE (sec)-[:HAS_CHUNK]->(ck)
            """, section_path=path, chunk_id=ck["entity_id"])


# ════════════════════════════════════════════════════════════════════
# 3. Image
# ════════════════════════════════════════════════════════════════════

def build_images(session, images: list, chunks: list):
    """
    一个图片实体对应一个 Image 节点，img_paths 数组作为属性保留（分图场景）。
    唯一键：img_key，取 img_paths[0] 作为代理键。
    entity_name 作为普通属性，同名图片有多个节点。

    Chunk -> Image 关系通过消耗队列建立：
    - 队列按 entity_name 分组
    - chunk 内去重（同一 chunk 多次引用同名图只消耗一次）
    - 跨 chunk 各自独立消耗（不同 chunk 的同名图分别取队列中的下一个）
    """
    # 为每个图片实体生成代理键
    enriched = [
        {
            **img,
            "img_key": img["img_paths"][0],
        }
        for img in images
    ]

    # upsert Image 节点
    session.run("""
        UNWIND $rows AS row
        MERGE (n:Image {img_key: row.img_key})
        SET n.entity_name = row.entity_name,
            n.img_paths   = row.img_paths
    """, rows=enriched)

    # 构建消耗队列：entity_name -> [img_key, ...]
    img_queues = defaultdict(list)
    for row in enriched:
        img_queues[row["entity_name"]].append(row["img_key"])

    # 遍历 chunk，解析占位符，建 CONTAINS_IMAGE 边
    edges = []
    for ck in chunks:
        content = ck.get("content", "")

        # 连续同名占位符视为同一张图（去重）
        # 占位符之间间隔文本 strip 后为空 → 连续；非空 → 打断连续性
        prev_img = None
        parts = IMAGE_RE.split(content)
        deduped = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip():
                    prev_img = None
            else:
                name = part
                if name != prev_img:
                    deduped.append(name)
                prev_img = name

        for img_name in deduped:
            if img_queues.get(img_name):
                target_key = img_queues[img_name].pop(0)
                edges.append({
                    "chunk_id": ck["entity_id"],
                    "img_key":  target_key,
                })
            else:
                print(f"未找到图片或队列已耗尽：chunk={ck['entity_id']}  图名={img_name}")

    if edges:
        session.run("""
            UNWIND $edges AS e
            MATCH (ck:Chunk {entity_id: e.chunk_id})
            MATCH (img:Image {img_key: e.img_key})
            MERGE (ck)-[:CONTAINS_IMAGE]->(img)
        """, edges=edges)

    # 孤儿图提示
    for name, remaining in img_queues.items():
        if remaining:
            print(f"孤儿图提示：[{name}] 还有 {len(remaining)} 张未被任何 Chunk 引用。")


# ════════════════════════════════════════════════════════════════════
# 4. Table
# ════════════════════════════════════════════════════════════════════

def build_tables(session, tables: list, chunks: list):
    """
    唯一键：table_img_path

    Chunk -> Table 关系通过消耗队列建立，逻辑与 Image 相同：
    - 队列按 entity_name 分组
    - chunk 内去重
    - 跨 chunk 各自独立消耗
    """
    # upsert Table 节点
    session.run("""
        UNWIND $rows AS row
        MERGE (n:Table {table_img_path: row.table_img_path})
        SET n.entity_name  = row.entity_name,
            n.table_path   = row.table_path,
            n.table_body   = row.table_body,
            n.parent_table = row.parent_table
    """, rows=tables)

    # 构建消耗队列：entity_name -> [table_img_path, ...]
    table_queues = defaultdict(list)
    for t in tables:
        table_queues[t["entity_name"]].append(t["table_img_path"])

    # 遍历 chunk，解析占位符，建 CONTAINS_TABLE 边
    edges = []
    for ck in chunks:
        content = ck.get("content", "")
        raw_names = TABLE_RE.findall(content)

        # chunk 内去重，保留首次出现顺序
        deduped = list(dict.fromkeys(raw_names))

        for tbl_name in deduped:
            if table_queues.get(tbl_name):
                target_img_path = table_queues[tbl_name].pop(0)
                edges.append({
                    "chunk_id":       ck["entity_id"],
                    "table_img_path": target_img_path,
                })
            else:
                print(f"未找到表格或队列已耗尽：chunk={ck['entity_id']}  表名={tbl_name}")

    if edges:
        session.run("""
            UNWIND $edges AS e
            MATCH (ck:Chunk {entity_id: e.chunk_id})
            MATCH (tbl:Table {table_img_path: e.table_img_path})
            MERGE (ck)-[:CONTAINS_TABLE]->(tbl)
        """, edges=edges)

    # 孤儿表提示
    for name, remaining in table_queues.items():
        if remaining:
            print(f"孤儿表提示：[{name}] 还有 {len(remaining)} 个物理表格未被任何 Chunk 引用。")


# ════════════════════════════════════════════════════════════════════
# 5. TableRow
# ════════════════════════════════════════════════════════════════════

def build_table_rows(session, rows: list):
    """
    唯一键：table_img_path + "::" + row_index（拼接复合键）
    父表通过 table_img_path 精确匹配，避免同名表歧义。
    相邻行间建 NEXT_ROW 链。
    """
    enriched = [
        {
            **row,
            "row_id": f"{row['table_img_path']}::{row['row_index']}",
        }
        for row in rows
    ]

    # upsert TableRow 节点
    session.run("""
        UNWIND $rows AS row
        MERGE (r:TableRow {row_id: row.row_id})
        SET r.row_text           = row.row_text,
            r.row_index          = row.row_index,
            r.table_img_path     = row.table_img_path,
            r.parent_entity_name = row.parent_entity_name
    """, rows=enriched)

    # Table -> TableRow（用 table_img_path 精确匹配父表）
    session.run("""
        UNWIND $rows AS row
        MATCH (tbl:Table   {table_img_path: row.table_img_path})
        MATCH (r:TableRow  {row_id: row.row_id})
        MERGE (tbl)-[:HAS_ROW]->(r)
    """, rows=enriched)

    # 行间顺序链：按 table_img_path 分组，组内按 row_index 排序，相邻建 NEXT_ROW 边
    groups = defaultdict(list)
    for row in enriched:
        groups[row["table_img_path"]].append(row)

    seq_edges = []
    for _, group_rows in groups.items():
        sorted_rows = sorted(group_rows, key=lambda r: r["row_index"])
        for i in range(len(sorted_rows) - 1):
            seq_edges.append({
                "from_id": sorted_rows[i]["row_id"],
                "to_id":   sorted_rows[i + 1]["row_id"],
            })

    if seq_edges:
        session.run("""
            UNWIND $edges AS e
            MATCH (r1:TableRow {row_id: e.from_id})
            MATCH (r2:TableRow {row_id: e.to_id})
            MERGE (r1)-[:NEXT_ROW]->(r2)
        """, edges=seq_edges)


# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════

def build_knowledge_graph(
    file_id: str,
    neo4j_uri:  str   = "bolt://localhost:7687",
    neo4j_auth: tuple = ("neo4j", "20472036"),
):
    print(f"📂 加载数据  file_id={file_id}")
    data = load_entities(file_id)
    print(f"   Section={len(data['sections'])}  Chunk={len(data['chunks'])}  "
          f"Image={len(data['images'])}  Table={len(data['tables'])}  Row={len(data['rows'])}")

    driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    with driver.session() as s:
        print("0. 初始化约束 & 索引...")
        init_constraints(s)

        print("1. 构建 Section 树...")
        build_section_tree(s, data["sections"])

        print("2. 构建 Chunk 节点...")
        build_chunks(s, data["chunks"], data["merged_section_ids"])

        print("3. 构建 Image 节点及关系...")
        build_images(s, data["images"], data["chunks"])

        print("4. 构建 Table 节点及关系...")
        build_tables(s, data["tables"], data["chunks"])

        print("5. 构建 TableRow 节点及行序关系...")
        build_table_rows(s, data["rows"])

    driver.close()
    print("✅ 知识图谱构建完成")


if __name__ == "__main__":
    build_knowledge_graph(file_id="f_55l2wt09")