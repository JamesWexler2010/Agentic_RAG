import json
from pathlib import Path
from services.index_service import index_dir
from services.text_index_service import _clean_page_markers, _text_entities_path, build_text_entity_index, cleaned_chunks_json_path, extract_text_entities, save_text_entities, split_markdown_by_headers
from services.table_index_service import (
    _build_table_faiss, _image_entities_path, _markdown_path,
    _save_image_entities_json, _save_table_entities_json, _table_entities_path,
    build_cleaned_markdown, build_image_index, build_img_cleaned_markdown,
    build_table_sub_entity_index, extract_image_entities, extract_table_entities,
    generate_all_tables_nl, resolve_continued_tables, load_table_entities_json, load_image_entities_json,
)

# ─────────────────────────────────────────────────────────────
#  Stage 1  提取表格实体 → 落盘 JSON → 等待人工确认
# ─────────────────────────────────────────────────────────────
async def build_media_index_stage1(file_id: str, force: bool = False) -> dict:
    table_entities_path = _table_entities_path(file_id)

    # ── 幂等跳过 ──
    if not force and table_entities_path.exists():
        entities = load_table_entities_json(file_id)
        return {
            "ok":                  True,
            "stage":               1,
            "skipped":             True,
            "tables":              len(entities),
            "table_entities_json": str(table_entities_path),
            "next_step":           "已有 table_entities.json，可直接调用 stage2（或传 force=True 重新提取）",
        }

    md_file = _markdown_path(file_id)
    if not md_file.exists():
        return {"ok": False, "error": "MARKDOWN_NOT_FOUND"}

    md_text  = md_file.read_text(encoding="utf-8")
    entities = extract_table_entities(md_text, file_id=file_id)
    entities = resolve_continued_tables(entities, file_id=file_id)
    table_entities_path = _save_table_entities_json(file_id, entities)

    return {
        "ok":                  True,
        "stage":               1,
        "skipped":             False,
        "tables":              len(entities),
        "table_entities_json": str(table_entities_path),
        "next_step":           "检查 table_entities.json，确认无误后调用 stage2",
    }


# ─────────────────────────────────────────────────────────────
#  Stage 2  表格 → 占位符 → 提取图片实体 → 落盘 JSON → 等待人工确认
# ─────────────────────────────────────────────────────────────
async def build_media_index_stage2(file_id: str, force: bool = False) -> dict:
    image_entities_path   = _image_entities_path(file_id)
    table_cleaned_md_path = Path("data") / file_id / "table_cleaned_output.md"

    # ── 幂等跳过 ──
    if not force and image_entities_path.exists() and table_cleaned_md_path.exists():
        img_entities = load_image_entities_json(file_id)
        return {
            "ok":                  True,
            "stage":               2,
            "skipped":             True,
            "images":              len(img_entities),
            "table_cleaned_md":    str(table_cleaned_md_path),
            "image_entities_json": str(image_entities_path),
            "next_step":           "已有 image_entities.json，可直接调用 stage3（或传 force=True 重新提取）",
        }

    table_entities_path = _table_entities_path(file_id)
    if not table_entities_path.exists():
        return {"ok": False, "error": "TABLE_ENTITIES_NOT_FOUND — 请先执行 stage1"}

    md_file  = _markdown_path(file_id)
    md_text  = md_file.read_text(encoding="utf-8")
    entities = load_table_entities_json(file_id)

    table_cleaned_md = build_cleaned_markdown(md_text, entities) if entities else md_text
    table_cleaned_md_path.parent.mkdir(parents=True, exist_ok=True)
    table_cleaned_md_path.write_text(table_cleaned_md, encoding="utf-8")

    img_entities        = extract_image_entities(table_cleaned_md, file_id=file_id)
    image_entities_path = _save_image_entities_json(file_id, img_entities)

    return {
        "ok":                  True,
        "stage":               2,
        "skipped":             False,
        "images":              len(img_entities),
        "table_cleaned_md":    str(table_cleaned_md_path),
        "image_entities_json": str(image_entities_path),
        "next_step":           "检查 image_entities.json，确认无误后调用 stage3",
    }


# ─────────────────────────────────────────────────────────────
#  Stage 3  图片 → 占位符 → 落盘 cleaned MD → 构建索引 → NL 生成 → 等待人工确认
# ─────────────────────────────────────────────────────────────
async def build_media_index_stage3(file_id: str, force: bool = False) -> dict:
    cleaned_md_path = Path("data") / file_id / "multi_cleaned_output.md"
    nl_json_path    = Path("data") / file_id / "table_nl.json"          # generate_all_tables_nl 的落盘路径

    # ── 幂等跳过 ──
    if not force and cleaned_md_path.exists() and nl_json_path.exists():
        return {
            "ok":            True,
            "stage":         3,
            "skipped":       True,
            "next_step":     "已有 cleaned_md 和 NL JSON，可直接调用 stage4（或传 force=True 重新生成）",
        }

    table_cleaned_md_path = Path("data") / file_id / "table_cleaned_output.md"
    table_entities_path   = _table_entities_path(file_id)
    image_entities_path   = _image_entities_path(file_id)

    for p in (table_cleaned_md_path, table_entities_path, image_entities_path):
        if not p.exists():
            return {"ok": False, "error": f"{p.name} 不存在 — 请先执行前序 stage"}

    table_cleaned_md = table_cleaned_md_path.read_text(encoding="utf-8")
    entities = load_table_entities_json(file_id)
    img_entities = load_image_entities_json(file_id)

    cleaned_md = (
        build_img_cleaned_markdown(table_cleaned_md, img_entities)
        if img_entities else table_cleaned_md
    )
    cleaned_md_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_md_path.write_text(cleaned_md, encoding="utf-8")

    if entities:
        _build_table_faiss(file_id, entities)
    if img_entities:
        build_image_index(file_id, img_entities)

    nl_result = await generate_all_tables_nl(file_id, entities) if entities else None

    return {
        "ok":            True,
        "stage":         3,
        "skipped":       False,
        "cleaned_md":    str(cleaned_md_path),
        "nl_generation": nl_result,
        "next_step":     "检查 NL 生成 JSON，确认无误后调用 stage4",
    }


# ─────────────────────────────────────────────────────────────
#  Stage 4  子实体拆分 → 构建 table_sub_entities 索引
# ─────────────────────────────────────────────────────────────
async def build_media_index_stage4(file_id: str, force: bool = False) -> dict:
    sub_entity_index_path = Path("data") / file_id / "index_faiss" / "table_sub_entities.faiss" # FAISS 落盘路径

    # ── 幂等跳过 ──
    if not force and sub_entity_index_path.exists():
        return {
            "ok":               True,
            "stage":            4,
            "skipped":          True,
            "sub_entity_index": str(sub_entity_index_path),
            "next_step":        "已完成，传 force=True 可重新构建索引",
        }

    table_entities_path = _table_entities_path(file_id)
    image_entities_path = _image_entities_path(file_id)

    if not table_entities_path.exists():
        return {"ok": False, "error": "TABLE_ENTITIES_NOT_FOUND — 请先执行前序 stage"}

    entities     = load_table_entities_json(file_id)
    img_entities = (
        load_image_entities_json(file_id)
        if image_entities_path.exists() else []
    )

    sub_entity_result = build_table_sub_entity_index(file_id) if entities else None

    return {
        "ok":               True,
        "stage":            4,
        "skipped":          False,
        "tables":           len(entities),
        "images":           len(img_entities),
        "sub_entity_index": sub_entity_result,
    }

# ─────────────────────────────────────────────────────────────
#  Text Stage 1  切块 → 保存 chunks.json → 等待人工确认
# ─────────────────────────────────────────────────────────────
async def build_text_entities_stage1(file_id: str, force: bool = False) -> dict:
    chunks_path = cleaned_chunks_json_path(file_id)

    # ── 幂等跳过 ──
    if not force and chunks_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        return {
            "ok":          True,
            "skipped":     True,
            "chunks":      len(chunks),
            "chunks_json": str(chunks_path),
            "next_step":   "已有 chunks.json，可直接调用 stage2（或传 force=True 重新切块）",
        }

    cleaned_md_path = Path("data") / file_id / "multi_cleaned_output.md"
    if not cleaned_md_path.exists():
        return {"ok": False, "error": "CLEANED_MD_NOT_FOUND — 请先执行 media index 流程"}

    md_text = cleaned_md_path.read_text(encoding="utf-8")
    md_text = _clean_page_markers(md_text)
    chunks  = split_markdown_by_headers(md_text)

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "ok":          True,
        "skipped":     False,
        "chunks":      len(chunks),
        "chunks_json": str(chunks_path),
        "next_step":   "检查 chunks.json，确认无误后调用 build_text_entities_stage2",
    }


# ─────────────────────────────────────────────────────────────
#  Text Stage 2  提取实体 → 保存 JSON → 构建 FAISS 索引
# ─────────────────────────────────────────────────────────────
async def build_text_entities_stage2(file_id: str, force: bool = False) -> dict:
    index_result_path = _text_entities_path(file_id)

    # ── 幂等跳过 ──
    if not force and index_result_path.exists():
        return {
            "ok":            True,
            "skipped":       True,
            "entities_json": str(index_result_path),
            "next_step":     "已完成，传 force=True 可重新构建索引",
        }

    chunks_path = cleaned_chunks_json_path(file_id)
    if not chunks_path.exists():
        return {"ok": False, "error": "CHUNKS_NOT_FOUND — 请先执行 stage1"}

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    entities = extract_text_entities(chunks)
    save_text_entities(file_id, entities)
    index_result = build_text_entity_index(file_id)

    return {
        "ok":            True,
        "skipped":       False,
        "entities_json": str(_text_entities_path(file_id)),
        "index":         index_result,
    }