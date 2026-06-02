# services/table_index_service.py
import re
import os
import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from services.index_service import load_embeddings, index_dir
from langchain.chat_models import init_chat_model


# ─── LLM Prompt ──────────────────────────────────────────

TABLE_NL_PROMPT = """你是一名工程文档撰写专家。请将以下表格按行转换为自然语言描述。

## 核心规则

### 1. 合并单元格还原
- rowspan 合并的单元格：该值属于其覆盖的每一行，转换时每行都必须包含该内容
- colspan 合并的单元格：视为该行在这几列上共用同一内容
- 若某格为空或"-"，描述时略去
- 数值、范围、公式原样保留
- 图示列（仅含图片或图示的列）写"详见图示"，不描述图片内容

### 2. 逐行描述
- 每个物理行（最细粒度的数据行）单独输出一段
- 该行所有有效字段都必须出现在描述中，不得遗漏
- 每行格式：**[分组路径]**：完整描述。

表格标题：{table_caption}
表格内容：{table_body}"""


# ═══════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class TableEntity:
    entity_name: str                       # 表名，如 "表1 结构装配精度"
    entity_type: str                       # 固定 "table"
    table_path: str                        # "{file_id}/{entity_name}"，用作全局唯一键
    table_body: str                        # <table>...</table> HTML 原文
    table_img_path: Optional[str] = None   # 紧跟表格的图片路径（如有）
    parent_table:   Optional[str] = None   # 续表指向主表的 entity_name；主表为 None


@dataclass
class TableSubEntity:
    parent_table_path: str                 # 所属表格的 table_path
    parent_entity_name: str                # 所属表格的 entity_name
    row_index: int                         # 在 NL 描述中的行号
    row_text: str                          # 该行的自然语言文本
    table_img_path: str = ""


@dataclass
class ImageEntity:
    entity_name: str                       # 图名，如 "图2 局部间隙超标处理示意图"
    entity_type: str                       # 固定 "image"
    img_paths:   list[str]                 # 同一图的所有图片路径（多视角/续图）


# ═══════════════════════════════════════════════════════════
#  路径工具
# ═══════════════════════════════════════════════════════════

def _markdown_path(file_id: str) -> Path:
    return Path("data") / file_id / "output.md"

def _table_index_dir(file_id: str) -> str:
    return str(index_dir(file_id))

def _nl_output_path(file_id: str) -> Path:
    return Path("data") / file_id / "table_nl.json"

def _checkpoint_path(file_id: str) -> Path:
    return Path("data") / file_id / "table_nl.checkpoint"

def _table_entities_path(file_id: str) -> Path:
    return Path("data") / file_id / "table_entities.json"

def _image_entities_path(file_id: str) -> Path:
    return Path("data") / file_id / "image_entities.json"

def _sub_entities_path(file_id: str) -> Path:
    return Path("data") / file_id / "table_sub_entities.json"


# ═══════════════════════════════════════════════════════════
#  1. 表格实体提取
# ═══════════════════════════════════════════════════════════

# ── 1.1 从 MD 中提取原始 TableEntity ──

_TABLE_EXTRACT_RE = re.compile(
    r'\*\*(.+?)\*\*\s*\n'                       # group(1): 加粗表名
    r'(?:(?!\*\*\s*表\s*[\dA-Za-z]).+?\n)*?'    # 中间行：放行注解粗体，阻止下一个 **表N 表标题
    r'(<table>[\s\S]+?</table>)'                 # group(2): 表格 HTML
    r'(?:\s*\n!\[.*?\]\((.*?)\))?',              # group(3): 可选的紧跟图片路径
    re.MULTILINE
)


def extract_table_entities(markdown_text: str, file_id: str) -> list[TableEntity]:
    """
    用正则从 output.md 中找出所有 **表名** + <table>...</table> 块，
    每个匹配创建一个 TableEntity。
    """
    entities = []
    for match in _TABLE_EXTRACT_RE.finditer(markdown_text):
        name = match.group(1).strip()
        entities.append(TableEntity(
            entity_name=name,
            entity_type="table",
            table_path=f"{file_id}/{name}",
            table_body=match.group(2).strip(),
            table_img_path=match.group(3).strip() if match.group(3) else None,
        ))
    return entities


# ── 1.2 续表解析：将 "表 N（续）" 归并到对应的主表 ──

_续表_RE = re.compile(r'^表\s*[\w.\-]+\s*[（(]\s*续\s*\d*\s*[）)]')   # ← [修复] \d+ → [\w.\-]+


def resolve_continued_tables(entities: list[TableEntity], file_id: str) -> list[TableEntity]:
    """
    按顺序遍历，遇到续表就指向最近的主表：
      - parent_table 设为主表 entity_name
      - entity_name 改为 "主表名（续）" / "主表名（续2）"
    """
    last_main: Optional[str] = None
    sub_count: int = 0
    for e in entities:
        if _续表_RE.match(e.entity_name):
            if last_main:
                sub_count += 1
                suffix = "（续）" if sub_count == 1 else f"（续{sub_count}）"
                e.parent_table = last_main
                e.entity_name  = f"{last_main}{suffix}"
                e.table_path   = f"{file_id}/{e.entity_name}"
        else:
            last_main = e.entity_name
            sub_count = 0
    return entities


# ═══════════════════════════════════════════════════════════
#  2. Markdown 清洗 —— 表格 → 占位符
# ═══════════════════════════════════════════════════════════

_TABLE_BLOCK_RE = re.compile(
    r'\*\*(.+?)\*\*\s*\n'
    r'(?:(?!\*\*\s*表\s*[\dA-Za-z]).*?\n)*?'    # ← 放行注解粗体，阻止 **表N 表标题
    r'(<table>[\s\S]+?</table>)'
    r'(?:\s*\n!\[.*?\]\([^)]*\))?',
    re.MULTILINE
)


def build_cleaned_markdown(
    markdown_text: str,
    entities: list[TableEntity],
) -> str:
    """
    把 MD 中每个 **表名** + <table>...</table>（含可能跟随的图片行）
    替换为 [[表格：entity_name]]。
    用 (table_body, 出现序号) 双重查找，避免极端情况下 body 相同导致覆盖。
    """
    # 构建 body → [name1, name2, ...] 有序列表，pop(0) 按顺序消费    ← [修复]
    body_to_names: dict[str, list[str]] = {}
    for e in entities:
        body_to_names.setdefault(e.table_body, []).append(e.entity_name)

    def replace(m: re.Match) -> str:
        table_body = m.group(2).strip()
        names = body_to_names.get(table_body)
        if names:
            name = names.pop(0)
        else:
            name = m.group(1).strip()
        return f"[[表格：{name}]]"

    return _TABLE_BLOCK_RE.sub(replace, markdown_text)


# ═══════════════════════════════════════════════════════════
#  3. 图片实体提取
# ═══════════════════════════════════════════════════════════

_IMG_LINE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_FIG_NAME_RE = re.compile(r'图\s*[\w.\-]+[^\n）)]*')        # ← [修复] \d+ → [\w.\-]+ 支持 图A.5
_SUBFIG_RE   = re.compile(r'^[（(]?[a-zA-Z][）)]\s*')
_IMG_续图_RE = re.compile(r'[（(]\s*续\s*\d*\s*[）)]?\s*$')


def _strip_续图_suffix(alt: str) -> str:
    """去掉续图后缀。如 '图A.5（续）' → '图A.5'"""
    return re.sub(r'\s*[（(]\s*续\s*\d*\s*[）)]?\s*$', '', alt).strip()


def _extract_fig_name(alt_texts: list[str]) -> str:
    """从一组 alt 文本里提取主图名（含 "图N" 的那个），找不到则用最后一个 alt。"""
    for alt in reversed(alt_texts):
        m = _FIG_NAME_RE.search(alt)
        if m:
            return m.group(0).strip()
    return alt_texts[-1].strip() if alt_texts else "未命名图片"


def _parse_subfig_alt(alt: str) -> tuple[str, str]:
    """
    从 alt 文本里分离子图名和主图名。
    例："b)分中退焊法 图8 退焊焊接" → ("b)分中退焊法", "图8 退焊焊接")
    例："a)施工条件允许时"          → ("a)施工条件允许时", "")
    """
    m = _FIG_NAME_RE.search(alt)
    if m:
        return alt[:m.start()].strip(), m.group(0).strip()
    return alt.strip(), ""


def extract_image_entities(cleaned_md: str, file_id: str) -> list[ImageEntity]:
    """
    提取规则：
    1. 按 \\n\\n 切段，收集含图片的段落
    2. 子图段落（alt 全含 a)/b) 前缀）向后收集直到找到含 图N 的段落
    3. 组内每张图构建 entity_name = 主图名 + 子图名，同名合并 img_paths
    4. 续图段落（alt 含（续）后缀）并入最近注册的主图 img_paths
    5. 同名图片的合并策略（按段落 idx 间距判断）：
       - 情况 A（连续碎图）：两次出现之间没有其他正文/表格段落（idx 相邻，
         间距 ≤ _ADJACENT_IDX_GAP），认为是同一张图的多个切片，合并到同一
         img_paths 列表。
       - 情况 B（跨章节同名）：中间隔了大量正文/表格（idx 间距超过阈值），
         认为是不同章节独立的图，保持原名、新建独立实体分开输出（由调用方
         用队列按序消费同名实体）。
    """
    # 两个含图段落的 para-idx 差值 ≤ 此阈值时，视为连续碎图（情况 A）
    _ADJACENT_IDX_GAP = 2

    # name → list of (entity, last_para_idx)，支持同名图跨章节多次出现
    name_to_occurrences: dict[str, list[tuple[ImageEntity, int]]] = {}
    # 保序输出用
    ordered_entities: list[ImageEntity] = []

    def register(name: str, paths: list[str], para_idx: int) -> None:
        """
        注册一组路径到 name 对应的实体。
        - 若该名字从未出现过：新建实体。
        - 若已出现过，且最后一次出现的段落 idx 与当前 para_idx 相邻（情况 A）：
          追加到已有实体的 img_paths。
        - 若已出现过，但间距过大（情况 B）：保持原名、新建独立实体分开输出。
        """
        if name not in name_to_occurrences:
            entity = ImageEntity(entity_name=name, entity_type="image", img_paths=list(paths))
            name_to_occurrences[name] = [(entity, para_idx)]
            ordered_entities.append(entity)
            return

        occurrences = name_to_occurrences[name]
        last_entity, last_idx = occurrences[-1]

        if para_idx - last_idx <= _ADJACENT_IDX_GAP:
            # 情况 A：连续碎图，合并进最后一次出现的实体
            last_entity.img_paths.extend(paths)
            # 更新最后一次的 para_idx（支持三张以上碎图连续追加）
            occurrences[-1] = (last_entity, para_idx)
        else:
            # 情况 B：跨章节同名，保持原名新建独立实体
            new_entity = ImageEntity(entity_name=name, entity_type="image", img_paths=list(paths))
            occurrences.append((new_entity, para_idx))
            ordered_entities.append(new_entity)

    # ── 收集含图片的段落 ──
    paras     = cleaned_md.split('\n\n')
    img_paras = []
    for idx, para in enumerate(paras):
        lines     = [l.strip() for l in para.splitlines() if l.strip()]
        img_lines = [l for l in lines if _IMG_LINE_RE.match(l)]
        if not img_lines:
            continue
        img_paras.append({
            'idx':   idx,
            'alts':  [_IMG_LINE_RE.match(l).group(1) for l in img_lines],
            'paths': [_IMG_LINE_RE.match(l).group(2) for l in img_lines],
        })

    # ── 逐段落处理 ──
    i = 0
    while i < len(img_paras):
        para = img_paras[i]
        alts, paths, para_idx = para['alts'], para['paths'], para['idx']

        is_sub = all(_SUBFIG_RE.match(a) for a in alts)

        if is_sub:
            # 子图段落：向后收集直到组内某个 alt 含 "图N"
            group = [para]
            j     = i + 1
            found_main = any(_FIG_NAME_RE.search(a) for a in alts)
            while not found_main and j < len(img_paras):
                group.append(img_paras[j])
                found_main = any(
                    _FIG_NAME_RE.search(a)
                    for gp in group for a in gp['alts']
                )
                j += 1

            # 提取主图名
            main_name = ''
            for gp in group:
                for a in gp['alts']:
                    _, fn = _parse_subfig_alt(a)
                    if fn:
                        main_name = fn
                        break
                if main_name:
                    break

            # 为组内每张图注册实体（子图组视为同一段落，使用组首 idx）
            group_idx = group[0]['idx']
            for gp in group:
                for alt, path in zip(gp['alts'], gp['paths']):
                    sub_name, _ = _parse_subfig_alt(alt)
                    full_name   = f"{main_name} {sub_name}".strip() if main_name else sub_name
                    register(full_name, [path], group_idx)

            i = j

        else:
            # 判断是否是续图段落
            is_续图 = any(_IMG_续图_RE.search(a) for a in alts)

            if is_续图:
                for alt, path in zip(alts, paths):
                    if _IMG_续图_RE.search(alt):
                        # 续图并入最近注册的实体（直接追加，不走 register 的间距判断）
                        if ordered_entities:
                            ordered_entities[-1].img_paths.append(path)
                            # 同步更新 occurrences 中的 last_idx
                            base_name = ordered_entities[-1].entity_name
                            if base_name in name_to_occurrences:
                                occs = name_to_occurrences[base_name]
                                last_ent, _ = occs[-1]
                                if last_ent is ordered_entities[-1]:
                                    occs[-1] = (last_ent, para_idx)
                        else:
                            main_name = _strip_续图_suffix(alt) or _extract_fig_name([alt])
                            register(main_name, [path], para_idx)
                    else:
                        register(_extract_fig_name([alt]), [path], para_idx)
            else:
                # 普通图片段落
                name = _extract_fig_name(alts)
                register(name, paths, para_idx)

            i += 1

    return ordered_entities


# ═══════════════════════════════════════════════════════════
#  4. Markdown 清洗 —— 图片 → 占位符
# ═══════════════════════════════════════════════════════════

def build_img_cleaned_markdown(cleaned_md: str, entities: list[ImageEntity]) -> str:
    """将 cleaned_md 中所有图片段落替换为 [[图片：entity_name]]。"""
    path_to_name = {
        path: e.entity_name
        for e in entities
        for path in e.img_paths
    }

    def replace_para(para: str) -> str:
        lines     = [l.strip() for l in para.splitlines() if l.strip()]
        img_lines = [l for l in lines if _IMG_LINE_RE.match(l)]
        if not img_lines:
            return para
        for l in img_lines:
            path = _IMG_LINE_RE.match(l).group(2)
            if path in path_to_name:
                return f"[[图片：{path_to_name[path]}]]"
        return para

    return '\n\n'.join(replace_para(p) for p in cleaned_md.split('\n\n'))


# ═══════════════════════════════════════════════════════════
#  5. 检查点 & JSON 持久化
# ═══════════════════════════════════════════════════════════

def _load_checkpoint(file_id: str) -> set[str]:
    p = _checkpoint_path(file_id)
    if not p.exists():
        return set()
    return set(p.read_text(encoding="utf-8").splitlines())


def _save_checkpoint(file_id: str, entity_name: str) -> None:
    with _checkpoint_path(file_id).open("a", encoding="utf-8") as f:
        f.write(entity_name + "\n")


def _load_nl_json(file_id: str) -> dict:
    p = _nl_output_path(file_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_nl_json(file_id: str, data: dict) -> None:
    p = _nl_output_path(file_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_table_entities_json(file_id: str, entities: list[TableEntity]) -> Path:
    p = _table_entities_path(file_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([{
            "entity_name":    e.entity_name,
            "entity_type":    e.entity_type,
            "table_path":     e.table_path,
            "table_body":     e.table_body,
            "table_img_path": e.table_img_path or "",
            "parent_table":   e.parent_table or "",
        } for e in entities], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return p


def _save_image_entities_json(file_id: str, entities: list[ImageEntity]) -> Path:
    p = _image_entities_path(file_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([{
            "entity_name": e.entity_name,
            "entity_type": e.entity_type,
            "img_paths":   e.img_paths,
        } for e in entities], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return p

def _save_sub_entities_json(file_id: str, subs: list[TableSubEntity]) -> Path:
    p = _sub_entities_path(file_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps([{
            "parent_table_path":  s.parent_table_path,
            "parent_entity_name": s.parent_entity_name,
            "row_index":          s.row_index,
            "row_text":           s.row_text,
            "table_img_path":     s.table_img_path,
        } for s in subs], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return p

# ═══════════════════════════════════════════════════════════
#  6. LLM 生成表格自然语言描述
# ═══════════════════════════════════════════════════════════

async def _generate_single_nl(entity: TableEntity, llm) -> str:  # ← [修复] llm 由外部传入
    prompt = TABLE_NL_PROMPT.format(
        table_caption=entity.entity_name,
        table_body=entity.table_body,
    )
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    return response.content or ""


async def generate_all_tables_nl(file_id: str, entities: list[TableEntity]) -> dict:
    # 1. 加载断点记录
    done = _load_checkpoint(file_id)
    
    # 2. 筛选待处理队列：
    # 条件 A: e.table_img_path 必须有值 (不为 None 或空字符串)
    # 条件 B: e.table_img_path 不在已完成列表 (done) 中
    pending = [
        e for e in entities 
        if e.table_img_path and str(e.table_img_path) not in done
    ]

    # 计算因为缺少图片而被跳过的数量（可选，用于日志）
    missing_img_count = len([e for e in entities if not e.table_img_path])
    if missing_img_count > 0:
        print(f"提示：发现了 {missing_img_count} 个表格没有图片路径，已自动跳过。")

    if not pending:
        return {"ok": True, "skipped": len(done), "generated": 0}

    llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
    nl_data = _load_nl_json(file_id)

    for entity in pending:
        # 这里能确保 entity.table_img_path 绝对不为 None
        img_key = str(entity.table_img_path)
        
        nl_text = await _generate_single_nl(entity, llm)
        
        # 写入结果，Key 仍然可以使用 table_path 或 img_key
        # 如果 table_path 不唯一，建议结果字典也考虑使用 img_key 作为键
        nl_data[img_key] = {
            "entity_name":    entity.entity_name,
            "table_path":     entity.table_path,
            "table_img_path": img_key,
            "nl_description": nl_text,
        }
        
        _save_nl_json(file_id, nl_data)
        
        # 3. 保存断点：此时传入的 img_key 绝对是字符串，不会再报 TypeError
        _save_checkpoint(file_id, img_key)

    return {"ok": True, "skipped": len(done), "generated": len(pending)}

# ═══════════════════════════════════════════════════════════
#  7. 主入口：构建所有媒体索引
# ═══════════════════════════════════════════════════════════

async def build_media_index(file_id: str) -> dict:
    """
    完整流程：
      ① 读取 output.md
      ② 提取表格实体 + 续表解析
      ③ 表格 → 占位符，得到 table_cleaned_md
      ④ 提取图片实体
      ⑤ 图片 → 占位符，得到 multi_cleaned_output.md
      ⑥ 落盘所有 JSON
      ⑦ 构建 FAISS 索引（tables / images）
      ⑧ LLM 生成表格 NL 描述 → 子实体拆分 → 构建 table_sub_entities 索引
    """
    md_file = _markdown_path(file_id)
    if not md_file.exists():
        return {"ok": False, "error": "MARKDOWN_NOT_FOUND"}

    md_text = md_file.read_text(encoding="utf-8")

    # ── 表格 ──
    entities = extract_table_entities(md_text, file_id=file_id)
    entities = resolve_continued_tables(entities, file_id=file_id)

    # ── 清洗 MD：表格 → 占位符 ──
    if entities:
        table_cleaned_md = build_cleaned_markdown(md_text, entities)
    else:
        table_cleaned_md = md_text

    # ── 图片 ──
    img_entities = extract_image_entities(table_cleaned_md, file_id=file_id)

    # ── 清洗 MD：图片 → 占位符 ──
    if img_entities:
        cleaned_md = build_img_cleaned_markdown(table_cleaned_md, img_entities)
    else:
        cleaned_md = table_cleaned_md

    # ── 落盘 cleaned markdown ──
    cleaned_md_path = Path("data") / file_id / "multi_cleaned_output.md"
    cleaned_md_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_md_path.write_text(cleaned_md, encoding="utf-8")

    # ── 落盘 JSON ──
    table_entities_path = _save_table_entities_json(file_id, entities)
    image_entities_path = _save_image_entities_json(file_id, img_entities)

    # ── 构建 FAISS 索引：表格 & 图片 ──
    if entities:
        _build_table_faiss(file_id, entities)

    if img_entities:
        build_image_index(file_id, img_entities)

    # ── LLM 生成表格 NL 描述 + 子实体拆分与索引 ──
    nl_result = None
    sub_entity_result = None
    if entities:
        nl_result = await generate_all_tables_nl(file_id, entities)
        sub_entity_result = build_table_sub_entity_index(file_id)

    return {
        "ok":                  True,
        "tables":              len(entities),
        "images":              len(img_entities),
        "entities":            entities,
        "img_entities":        img_entities,
        "cleaned_md":          str(cleaned_md_path),
        "table_entities_json": str(table_entities_path),
        "image_entities_json": str(image_entities_path),
        "nl_generation":       nl_result,
        "sub_entity_index":    sub_entity_result,
    }


# ═══════════════════════════════════════════════════════════
#  8. 表格索引（构建 + 加载）
# ═══════════════════════════════════════════════════════════

def _build_table_faiss(file_id: str, entities: list[TableEntity]) -> None:
    """为表格实体构建 FAISS 索引。"""
    docs = [
        Document(
            page_content=entity.entity_name,
            metadata={
                "entity_name":    entity.entity_name,
                "entity_type":    entity.entity_type,
                "table_path":     entity.table_path,
                "table_body":     entity.table_body,
                "table_img_path": entity.table_img_path or "",
                "parent_table":   entity.parent_table or "",
            }
        )
        for entity in entities
    ]
    vs = FAISS.from_documents(docs, embedding=load_embeddings())
    vs.save_local(_table_index_dir(file_id), index_name="tables")


def load_table_vs(file_id: str) -> FAISS:
    path = _table_index_dir(file_id)
    if not os.path.exists(os.path.join(path, "tables.faiss")):
        raise FileNotFoundError(f"Table index not found: {path}")
    return FAISS.load_local(
        path, load_embeddings(),
        index_name="tables",
        allow_dangerous_deserialization=True,
    )


def build_table_sub_entity_index(file_id: str) -> dict:
    """为子实体构建 FAISS 索引。"""
    subs = split_nl_to_sub_entities(file_id)
    if not subs:
        return {"ok": False, "error": "NO_SUB_ENTITIES"}
    
    # ── 落盘 JSON ──
    sub_entities_path = _save_sub_entities_json(file_id, subs)

    docs = [
        Document(
            page_content=s.row_text,
            metadata={
                "parent_table_path":  s.parent_table_path,
                "parent_entity_name": s.parent_entity_name,
                "row_index":          s.row_index,
                "table_img_path":     s.table_img_path,
            },
        )
        for s in subs
    ]

    vs = FAISS.from_documents(docs, embedding=load_embeddings())
    vs.save_local(_table_index_dir(file_id), index_name="table_sub_entities")
    return {"ok": True, "sub_entities": len(docs), "sub_entities_json": str(sub_entities_path)}


def load_table_sub_entity_vs(file_id: str) -> FAISS:
    path = _table_index_dir(file_id)
    if not os.path.exists(os.path.join(path, "table_sub_entities.faiss")):
        raise FileNotFoundError(f"Table sub-entity index not found: {path}")
    return FAISS.load_local(
        path, load_embeddings(),
        index_name="table_sub_entities",
        allow_dangerous_deserialization=True,
    )


# ═══════════════════════════════════════════════════════════
#  9. 图片索引（构建 + 加载）
# ═══════════════════════════════════════════════════════════

def build_image_index(file_id: str, img_entities: list[ImageEntity]) -> dict:
    """为图片实体构建 FAISS 索引，以 entity_name 作为检索内容。"""
    if not img_entities:
        return {"ok": False, "error": "NO_IMAGE_ENTITIES"}

    docs = [
        Document(
            page_content=e.entity_name,
            metadata={
                "entity_name": e.entity_name,
                "entity_type": e.entity_type,
                "img_paths":   json.dumps(e.img_paths, ensure_ascii=False),
            }
        )
        for e in img_entities
    ]

    vs = FAISS.from_documents(docs, embedding=load_embeddings())
    vs.save_local(_table_index_dir(file_id), index_name="images")
    return {"ok": True, "images": len(docs)}


def load_image_vs(file_id: str) -> FAISS:
    path = _table_index_dir(file_id)
    if not os.path.exists(os.path.join(path, "images.faiss")):
        raise FileNotFoundError(f"Image index not found: {path}")
    return FAISS.load_local(
        path, load_embeddings(),
        index_name="images",
        allow_dangerous_deserialization=True,
    )


# ═══════════════════════════════════════════════════════════
#  10. 子实体拆分（依赖 table_nl.json）
# ═══════════════════════════════════════════════════════════

def split_nl_to_sub_entities(file_id: str) -> list[TableSubEntity]:
    """将 table_nl.json 中每个表格的 nl_description 按行拆为子实体。"""
    nl_data = _load_nl_json(file_id)
    sub_entities = []
    for table_path, info in nl_data.items():
        lines = [l.strip() for l in info["nl_description"].split("\n") if l.strip()]
        for i, line in enumerate(lines):
            sub_entities.append(TableSubEntity(
                parent_table_path=table_path,
                parent_entity_name=info["entity_name"],
                row_index=i,
                row_text=line,
                table_img_path=info.get("table_img_path", ""),
            ))
    return sub_entities
# ═══════════════════════════════════════════════════════════
#  11.反序列化 JSON → 实体对象列表
# ═══════════════════════════════════════════════════════════
def load_table_entities_json(file_id: str) -> list[TableEntity]:
    """从 table_entities.json 反序列化为 TableEntity 列表"""
    p = _table_entities_path(file_id)
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        TableEntity(
            entity_name    = d["entity_name"],
            entity_type    = d["entity_type"],
            table_path     = d["table_path"],
            table_body     = d["table_body"],
            table_img_path = d["table_img_path"] or None,
            parent_table   = d["parent_table"] or None,
        )
        for d in data
    ]


def load_image_entities_json(file_id: str) -> list[ImageEntity]:
    """从 image_entities.json 反序列化为 ImageEntity 列表"""
    p = _image_entities_path(file_id)
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        ImageEntity(
            entity_name = d["entity_name"],
            entity_type = d["entity_type"],
            img_paths   = d["img_paths"],
        )
        for d in data
    ]