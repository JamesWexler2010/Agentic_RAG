# services/text_entity_service.py
import re
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
from collections import OrderedDict
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from services.index_service import load_embeddings, index_dir
import re

# ———─ 文本清洗 ─────────────────────────────────────────────

def _clean_page_markers(md_text: str) -> str:
    return re.sub(r'<!--\s*page:\s*\d*\s*-->', '', md_text).strip()


# ─── 数据结构 ────────────────────────────────────────────

@dataclass
class TextEntity:
    entity_id: str          # section_path 做唯一键
    entity_name: str        # 当前层级名
    entity_type: str        # "section" | "chunk"
    depth: int
    section_path: str
    content: str = ""       # chunk 实体存原文，纯父节点为空


# ─── 工具函数 ────────────────────────────────────────────

def _normalize_path(path: str) -> str:
    return re.sub(r'\s*[（(]及后续\d+节[）)]\s*', '', path).strip()


def get_parent_id(entity_id: str) -> Optional[str]:
    parts = entity_id.split(" > ")
    return " > ".join(parts[:-1]) if len(parts) > 1 else None


# ─── 路径 ────────────────────────────────────────────────

def _chunks_json_path(file_id: str) -> Path:
    return Path("data") / file_id / "chunks.json"


def cleaned_chunks_json_path(file_id: str) -> Path:
    return Path("data") / file_id / "multi_cleaned_chunks.json"


def _text_entities_path(file_id: str) -> Path:
    return Path("data") / file_id / "text_entities.json"


def _text_index_dir(file_id: str) -> str:
    return str(index_dir(file_id))


# ─── MarkdownHeaderTextSplitter 切分 ─────────────────────

# 定义要识别的 Markdown 标题层级
HEADERS_TO_SPLIT_ON = [
    ("#",    "Header 1"),
    ("##",   "Header 2"),
    ("###",  "Header 3"),
    ("####", "Header 4"),
    ("#####","Header 5"),
    ("######","Header 6"),
]

# metadata key 按深度排序，用于构建 section_path
HEADER_KEYS_ORDERED = ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5", "Header 6"]


def split_markdown_by_headers(md_text: str) -> list[dict]:
    """
    使用 MarkdownHeaderTextSplitter 切分 Markdown，返回 chunk 列表。
    每个 chunk 包含：
      - chunk_id: int
      - headers: dict          # 原始 metadata，如 {"Header 1": "...", "Header 2": "..."}
      - section_path: str      # 由 headers 拼接而成，如 "第1章 船体专业 > 第2节 船体装配工艺规程 > 3 一般要求"
      - content: str           # chunk 的文本内容
      - length: int
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,       # 保留标题文本在 content 中，可按需改为 True
    )
    docs = splitter.split_text(md_text)

    chunks = []
    for i, doc in enumerate(docs):
        # 按层级顺序拼出 section_path
        path_parts = []
        for key in HEADER_KEYS_ORDERED:
            if key in doc.metadata:
                path_parts.append(doc.metadata[key].strip())

        section_path = " > ".join(path_parts) if path_parts else f"未分类段落_{i}"

        chunks.append({
            "chunk_id":     i + 1,
            "headers":      dict(doc.metadata),   # 保留原始 header 信息
            "section_path": section_path,
            "content":      doc.page_content,
            "length":       len(doc.page_content),
        })

    return chunks


# ─── 提取实体 ────────────────────────────────────────────

def extract_text_entities(chunks: list[dict]) -> list[TextEntity]:
    """
    从 MarkdownHeaderTextSplitter 产出的 chunks 中提取实体：
    1. 利用每个 chunk 的 headers metadata，逐级创建 section 实体（去重）
    2. 为每个 chunk 创建叶子级 chunk 实体
    """
    section_registry: OrderedDict[str, TextEntity] = OrderedDict()
    chunk_entities: list[TextEntity] = []

    for chunk in chunks:
        headers = chunk["headers"]

        # ① 按层级顺序逐级注册 section 实体
        path_parts = []
        for key in HEADER_KEYS_ORDERED:
            if key not in headers:
                break                       # 遇到缺失层级就停止（保证连续层级）
            part = headers[key].strip()
            path_parts.append(part)
            entity_id = " > ".join(path_parts)

            if entity_id not in section_registry:
                section_registry[entity_id] = TextEntity(
                    entity_id=entity_id,
                    entity_name=part,
                    entity_type="section",
                    depth=len(path_parts) - 1,    # depth 从 0 开始
                    section_path=entity_id,
                    content="",
                )

        # ② 创建 chunk 实体
        leaf_path = chunk["section_path"]
        chunk_entities.append(TextEntity(
            entity_id=f"chunk:{chunk['chunk_id']}",
            entity_name=path_parts[-1] if path_parts else f"chunk_{chunk['chunk_id']}",
            entity_type="chunk",
            depth=len(path_parts),
            section_path=leaf_path,
            content=chunk["content"],
        ))

    return list(section_registry.values()) + chunk_entities


# ─── 持久化 ──────────────────────────────────────────────

def save_text_entities(file_id: str, entities: list[TextEntity]) -> Path:
    p = _text_entities_path(file_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(e) for e in entities]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_text_entities(file_id: str) -> list[TextEntity]:
    p = _text_entities_path(file_id)
    if not p.exists():
        raise FileNotFoundError(f"text_entities.json not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return [TextEntity(**item) for item in data]


# ─── FAISS 索引 ──────────────────────────────────────────

def build_text_entity_index(file_id: str) -> dict:
    """
    构建 FAISS 索引：
    - chunk 实体：用 content（原文）作为 page_content
    - section 实体：用 section_path 作为 page_content
    """
    entities = load_text_entities(file_id)
    if not entities:
        return {"ok": False, "error": "NO_TEXT_ENTITIES"}

    docs = []
    for e in entities:
        if e.entity_type == "chunk":
            page_content = e.content
        else:
            page_content = e.section_path.replace(" > ", " ")

        docs.append(Document(
            page_content=page_content,
            metadata={
                "entity_id":    e.entity_id,
                "entity_name":  e.entity_name,
                "entity_type":  e.entity_type,
                "depth":        e.depth,
                "section_path": e.section_path,
            },
        ))

    vs = FAISS.from_documents(docs, embedding=load_embeddings())
    vs.save_local(_text_index_dir(file_id), index_name="text_entities")

    sections = sum(1 for e in entities if e.entity_type == "section")
    chunks_count = sum(1 for e in entities if e.entity_type == "chunk")
    return {"ok": True, "total": len(docs), "sections": sections, "chunks": chunks_count}


def load_text_entity_vs(file_id: str) -> FAISS:
    path = _text_index_dir(file_id)
    idx_file = os.path.join(path, "text_entities.faiss")
    if not os.path.exists(idx_file):
        raise FileNotFoundError(f"Text entity index not found: {path}")
    return FAISS.load_local(
        path,
        load_embeddings(),
        index_name="text_entities",
        allow_dangerous_deserialization=True,
    )


# ─── 入口 ────────────────────────────────────────────────

async def build_all_text_entities(file_id: str) -> dict:
    """build_media_index → 切块 → 提取实体 → 保存 JSON → 构建 FAISS 索引"""
    from services.table_index_service import build_media_index

    cleaned_md_path = Path("data") / file_id / "multi_cleaned_output.md"

    # cleaned MD 不存在时才执行 build_media_index
    if not cleaned_md_path.exists():
        media_result = await build_media_index(file_id)
        if not media_result["ok"]:
            return media_result

    md_text = cleaned_md_path.read_text(encoding="utf-8")

    md_text = _clean_page_markers(md_text) 

    # ── 使用 MarkdownHeaderTextSplitter 切分 ──
    chunks = split_markdown_by_headers(md_text)

    # ── 保存 chunks JSON ──
    chunks_path = cleaned_chunks_json_path(file_id)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 提取实体 & 构建索引 ──
    entities = extract_text_entities(chunks)
    save_text_entities(file_id, entities)
    index_result = build_text_entity_index(file_id)

    return {
        "ok":            True,
        "chunks":        len(chunks),
        "chunks_json":   str(chunks_path),
        "entities_json": str(_text_entities_path(file_id)),
        "index":         index_result,
    }