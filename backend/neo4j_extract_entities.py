"""
KG 检索流程：
  1. 向量检索（FAISS）命中实体：text / table / table_row / image
  2. 取出实体的唯一键（与 json_to_neo4j.py 的约束对齐）
  3. 在 Neo4j 中 MATCH 节点，并扩展必要上下文
       - Chunk   → 父 Section + 关联 Image/Table
       - Table   → 所在 Chunk + 全部 Row
       - TableRow → 父 Table + 前后邻居 Row + 所在 Chunk
       - Image   → 引用它的 Chunk
"""

from __future__ import annotations
from typing import Any
from neo4j import GraphDatabase

# 复用你项目里已有的加载器
from services.text_index_service import (      # ← 改成你的实际模块路径
    load_text_entity_vs
)

from services.table_index_service import (      # ← 改成你的实际模块路径
    load_table_vs,
    load_table_sub_entity_vs,
    load_image_vs,
)

# ════════════════════════════════════════════════════════════════════
# 1. 向量召回
# ════════════════════════════════════════════════════════════════════

def retrieve_entities(file_id: str, question: str, top_k: int = 5) -> dict:
    """
    从四个 FAISS 索引各取 top_k，返回分类后的命中列表。
    每条命中都带上进入 Neo4j 所需的唯一键 + 相似度得分。
    """
    hits: dict[str, list[dict]] = {
        "text":      [],
        "table":     [],
        "table_row": [],
        "image":     [],
    }

    # ---- text entity（Section / Chunk 共用索引）----
    for doc, score in load_text_entity_vs(file_id).similarity_search_with_score(question, k=top_k):
        md = doc.metadata
        hits["text"].append({
            "entity_id":   md["entity_id"],
            "entity_type": md.get("entity_type"),   # "section" / "chunk"
            "score":       float(score),
        })

    # ---- table ----
    for doc, score in load_table_vs(file_id).similarity_search_with_score(question, k=top_k):
        md = doc.metadata
        hits["table"].append({
            "table_img_path": md["table_img_path"],
            "entity_name":    md.get("entity_name"),
            "score":          float(score),
        })

    # ---- table sub-entity（行）----
    for doc, score in load_table_sub_entity_vs(file_id).similarity_search_with_score(question, k=top_k):
        md = doc.metadata
        hits["table_row"].append({
            "table_img_path": md["table_img_path"],
            "row_index":      md["row_index"],
            "row_id":         f"{md['table_img_path']}::{md['row_index']}",
            "score":          float(score),
        })

    # ---- image ----
    for doc, score in load_image_vs(file_id).similarity_search_with_score(question, k=top_k):
        md = doc.metadata
        hits["image"].append({
            "entity_name": md["entity_name"],
            "score":       float(score),
        })

    return hits


# ════════════════════════════════════════════════════════════════════
# 2. Cypher 语句
# ════════════════════════════════════════════════════════════════════

# 2.1 Section / Chunk —— 同一索引可能命中两种标签
#     · Chunk   : 取父 Section、关联的 Image / Table
#     · Section : 取下辖 Chunk（数量多时按需截断）
CYPHER_TEXT = """
UNWIND $ids AS eid
MATCH (n) WHERE (n:Chunk OR n:Section) AND n.entity_id = eid
OPTIONAL MATCH (parent:Section)-[:HAS_CHUNK]->(n)            // 仅对 Chunk 命中
OPTIONAL MATCH (n)-[:CONTAINS_IMAGE]->(img:Image)            // 仅对 Chunk 命中
OPTIONAL MATCH (n)-[:CONTAINS_TABLE]->(tbl:Table)            // 仅对 Chunk 命中
OPTIONAL MATCH (n)-[:HAS_CHUNK]->(child_ck:Chunk)            // 仅对 Section 命中
RETURN
    labels(n)                                             AS labels,
    n.entity_id                                           AS entity_id,
    n.entity_name                                         AS entity_name,
    n.content                                             AS content,
    n.section_path                                        AS section_path,
    n.depth                                               AS depth,
    parent.section_path                                   AS parent_section,
    collect(DISTINCT img   {.entity_name, .img_paths})    AS images,
    collect(DISTINCT tbl   {.entity_name, .table_img_path}) AS tables,
    collect(DISTINCT child_ck {.entity_id, .entity_name}) AS child_chunks
"""

# 2.2 Table —— 拉全表行 + 所在 Chunk
CYPHER_TABLE = """
UNWIND $paths AS p
MATCH (tbl:Table {table_img_path: p})
OPTIONAL MATCH (ck:Chunk)-[:CONTAINS_TABLE]->(tbl)
OPTIONAL MATCH (tbl)-[:HAS_ROW]->(r:TableRow)
WITH tbl,
     collect(DISTINCT ck {.entity_id, .section_path})        AS chunks,
     collect(r {.row_index, .row_text})                      AS raw_rows
RETURN
    tbl {.table_img_path, .entity_name, .table_body, .parent_table} AS table,
    chunks,
    [x IN raw_rows WHERE x.row_index IS NOT NULL | x]        AS rows
"""

# 2.3 TableRow —— 自身 + 父 Table + 前后邻接 + 所在 Chunk
CYPHER_ROW = """
UNWIND $row_ids AS rid
MATCH (r:TableRow {row_id: rid})
OPTIONAL MATCH (tbl:Table)-[:HAS_ROW]->(r)
OPTIONAL MATCH (prev:TableRow)-[:NEXT_ROW]->(r)
OPTIONAL MATCH (r)-[:NEXT_ROW]->(nxt:TableRow)
OPTIONAL MATCH (ck:Chunk)-[:CONTAINS_TABLE]->(tbl)
RETURN
    r   {.row_id, .row_index, .row_text, .table_img_path}      AS row,
    tbl {.table_img_path, .entity_name}                         AS table,
    prev {.row_index, .row_text}                                AS prev_row,
    nxt  {.row_index, .row_text}                                AS next_row,
    collect(DISTINCT ck {.entity_id, .section_path})            AS chunks
"""

# 2.4 Image —— 被哪些 Chunk 引用
CYPHER_IMAGE = """
UNWIND $names AS name
MATCH (img:Image {entity_name: name})
OPTIONAL MATCH (ck:Chunk)-[:CONTAINS_IMAGE]->(img)
RETURN
    img {.entity_name, .img_paths}                      AS image,
    collect(DISTINCT ck {.entity_id, .section_path})    AS chunks
"""


# ════════════════════════════════════════════════════════════════════
# 3. Neo4j 查询封装
# ════════════════════════════════════════════════════════════════════

class KGRetriever:
    def __init__(
        self,
        uri:  str = "bolt://localhost:7687",
        auth: tuple = ("neo4j", "20472036"),
    ):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 按实体类型各走一条 Cypher ----

    def fetch_text(self, entity_ids: list[str]) -> list[dict]:
        if not entity_ids:
            return []
        with self.driver.session() as s:
            return [rec.data() for rec in s.run(CYPHER_TEXT, ids=entity_ids)]

    def fetch_tables(self, table_img_paths: list[str]) -> list[dict]:
        if not table_img_paths:
            return []
        with self.driver.session() as s:
            return [rec.data() for rec in s.run(CYPHER_TABLE, paths=table_img_paths)]

    def fetch_rows(self, row_ids: list[str]) -> list[dict]:
        if not row_ids:
            return []
        with self.driver.session() as s:
            return [rec.data() for rec in s.run(CYPHER_ROW, row_ids=row_ids)]

    def fetch_images(self, names: list[str]) -> list[dict]:
        if not names:
            return []
        with self.driver.session() as s:
            return [rec.data() for rec in s.run(CYPHER_IMAGE, names=names)]


# ════════════════════════════════════════════════════════════════════
# 4. 端到端管道
# ════════════════════════════════════════════════════════════════════

def retrieve(
    file_id: str,
    question: str,
    top_k: int = 5,
    neo4j_uri:  str = "bolt://localhost:7687",
    neo4j_auth: tuple = ("neo4j", "20472036"),
) -> dict:
    """向量召回 → 图谱定位 → 返回聚合结果"""
    # 1) 向量召回
    hits = retrieve_entities(file_id, question, top_k=top_k)

    # 2) 去重取键
    text_ids  = list({h["entity_id"]      for h in hits["text"]})
    tbl_paths = list({h["table_img_path"] for h in hits["table"]})
    row_ids   = list({h["row_id"]         for h in hits["table_row"]})
    img_names = list({h["entity_name"]    for h in hits["image"]})

    # 3) 图谱 MATCH
    with KGRetriever(neo4j_uri, neo4j_auth) as kg:
        return {
            "question":   question,
            "text":       kg.fetch_text(text_ids),
            "tables":     kg.fetch_tables(tbl_paths),
            "table_rows": kg.fetch_rows(row_ids),
            "images":     kg.fetch_images(img_names),
            "vector_hits": hits,          # 原始向量分数，便于排序 / 调试
        }


# ════════════════════════════════════════════════════════════════════
# CLI 测试
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from pprint import pprint
    result = retrieve(
        file_id = "f_55l2wt09",
        question = "舱壁垂直度的标准偏差限制需同时满足哪两个条件？",
        top_k    = 5,
    )
    pprint(result, width=140, sort_dicts=False)