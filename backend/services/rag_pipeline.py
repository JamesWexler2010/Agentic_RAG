# services/rag_pipeline.py
"""
GraphRAG 流水线 - Step 1: 多索引向量检索

策略: 串行查询 4 个 FAISS 索引,各自独立 top-K,按桶返回。
错误隔离: 索引文件不存在静默跳过(预期情况),其他异常打 traceback。
保留全量信息: 每条命中保留完整 metadata + page_content + score。
"""
from __future__ import annotations
import json
import traceback
from typing import Any

# 按你实际的 import 路径调整
from services.table_index_service import (
    load_table_vs,
    load_table_sub_entity_vs,
    load_image_vs,
)

from services.text_index_service import (
    load_text_entity_vs,
)


def retrieve_entities(
    file_id: str,
    question: str,
    top_k: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """
    从四个 FAISS 索引各取 top_k,返回分桶后的命中列表。
    每条命中保留完整 metadata + page_content + score,
    并按各实体类型补全图谱主键(用于后续图遍历)。

    text 桶比较特殊:索引内同时存有 chunk 和 section,
    会按 entity_type 分别检索,各取 top_k 条。

    Args:
        file_id:  文档 ID
        question: 用户问题
        top_k:    每个索引/每个 entity_type 各取 top_k 条

    Returns:
        {
            "text":      [{entity_id, entity_type, ...metadata, page_content, score}],
            "table":     [{entity_name, table_img_path, ...metadata, page_content, score}],
            "table_row": [{row_id, table_img_path, row_index, row_text, ...metadata, score}],
            "image":     [{entity_name, img_paths(list), ...metadata, page_content, score}],
        }
        某个桶为空可能是该文档没建该索引,也可能是没召回到任何结果。
    """
    hits: dict[str, list[dict[str, Any]]] = {
        "text":      [],
        "table":     [],
        "table_row": [],
        "image":     [],
    }

    # ============ text_entities (Section / Chunk 共用,各取 top_k) ============
    try:
        vs = load_text_entity_vs(file_id)

        for etype in ("chunk", "section"):
            try:
                results = vs.similarity_search_with_score(
                    question,
                    k=top_k,
                    filter={"entity_type": etype},
                )
            except Exception as e:
                print(f"[retrieve_entities] text_entities filter={etype} 检索失败: {e}")
                traceback.print_exc()
                continue

            for doc, score in results:
                md = dict(doc.metadata)
                hits["text"].append({
                    **md,
                    "page_content": doc.page_content,
                    "score":        float(score),
                })

    except FileNotFoundError:
        print(f"[retrieve_entities] text_entities 索引不存在,跳过 (file_id={file_id})")
    except Exception as e:
        print(f"[retrieve_entities] text_entities 加载失败: {e}")
        traceback.print_exc()

    # ============ tables ============
    try:
        vs = load_table_vs(file_id)
        for doc, score in vs.similarity_search_with_score(question, k=top_k):
            md = dict(doc.metadata)
            hits["table"].append({
                **md,                              # entity_name / entity_type / table_path / table_body / table_img_path / parent_table
                "page_content": doc.page_content,  # entity_name (如 "表1")
                "score":        float(score),
            })
    except FileNotFoundError:
        print(f"[retrieve_entities] tables 索引不存在,跳过 (file_id={file_id})")
    except Exception as e:
        print(f"[retrieve_entities] tables 检索失败: {e}")
        traceback.print_exc()

    # ============ table_sub_entities (TableRow) ============
    try:
        vs = load_table_sub_entity_vs(file_id)
        for doc, score in vs.similarity_search_with_score(question, k=top_k):
            md = dict(doc.metadata)
            # row_text 在 page_content 里,不在 metadata 里
            row_text = doc.page_content
            # row_id 现场拼装,与图谱中 :TableRow {row_id} 一致
            row_id = f"{md['table_img_path']}::{md['row_index']}"
            hits["table_row"].append({
                **md,                  # parent_table_path / parent_entity_name / row_index / table_img_path
                "row_id":       row_id,
                "row_text":     row_text,
                "page_content": doc.page_content,
                "score":        float(score),
            })
    except FileNotFoundError:
        print(f"[retrieve_entities] table_sub_entities 索引不存在,跳过 (file_id={file_id})")
    except Exception as e:
        print(f"[retrieve_entities] table_sub_entities 检索失败: {e}")
        traceback.print_exc()

    # ============ images ============
    try:
        vs = load_image_vs(file_id)
        for doc, score in vs.similarity_search_with_score(question, k=top_k):
            md = dict(doc.metadata)
            # img_paths 在建索引时序列化成了 JSON 字符串,这里反序列化回 list
            img_paths_raw = md.get("img_paths", "[]")
            try:
                img_paths = json.loads(img_paths_raw) if isinstance(img_paths_raw, str) else img_paths_raw
            except json.JSONDecodeError:
                img_paths = []
                print(f"[retrieve_entities] image img_paths 解析失败: {img_paths_raw!r}")
            hits["image"].append({
                **md,                              # entity_name / entity_type / img_paths(原始JSON字符串会被下面覆盖)
                "img_paths":    img_paths,         # 覆盖为 list
                "page_content": doc.page_content,  # entity_name
                "score":        float(score),
            })
    except FileNotFoundError:
        print(f"[retrieve_entities] images 索引不存在,跳过 (file_id={file_id})")
    except Exception as e:
        print(f"[retrieve_entities] images 检索失败: {e}")
        traceback.print_exc()

    return hits


# ============ 调试辅助 ============
def format_hits_summary(hits: dict[str, list[dict]]) -> str:
    """把分桶结果格式化成可读字符串,用于调试。"""
    lines = []
    for bucket, items in hits.items():
        if not items:
            lines.append(f"【{bucket}】(无召回)")
            continue
        lines.append(f"【{bucket}】(命中 {len(items)} 条):")
        for i, h in enumerate(items, 1):
            content = (h.get("page_content") or "").replace("\n", " ")[:100]  # 截断长文本
            score = h.get("score", 0)
            # 各桶用最有辨识度的字段做标识
            if bucket == "text":
                ident = f"{h.get('entity_type')} | {h.get('entity_name')}"
            elif bucket == "table":
                ident = f"{h.get('entity_name')} ({h.get('table_img_path', '')[:50]}...)"
            elif bucket == "table_row":
                ident = f"row {h.get('row_index')} | {h.get('parent_entity_name')}"
            elif bucket == "image":
                ident = f"{h.get('entity_name')} | {len(h.get('img_paths', []))} 张图"
            else:
                ident = "?"
            lines.append(f"  [{i}] score={score:.3f} | {ident}")
            lines.append(f"      content: {content}")
    return "\n".join(lines)