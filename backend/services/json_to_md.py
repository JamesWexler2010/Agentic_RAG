"""
json_to_md.py
将 content_list.json 转换为 Markdown 文件（带页码标记版本）。

【页码标记策略】
  - 仅在相邻两块的 page_idx 发生变化时，在两块之间插入独立一行：
        <!-- page: N -->
    表示"从此行起进入第 N 页"。
  - 块内部不再重复标注页码，保持正文干净。

【Chunking 阶段使用方式】
  - 扫描 chunk 文本，提取其中所有 <!-- page: N --> 标记：
      * 无标记        → metadata.pages = [当前已知页]
      * 含一个标记    → metadata.pages = [上一页, N]（chunk 跨页）
      * 含多个标记    → metadata.pages = [所有跨越的页]
  - 通过 pages 列表可精准溯源到原文档对应页，提高引用置信度。

支持的类型：
  text (text_level=1/2/3 → #/##/### 标题, 普通 text → 段落)
  list      → Markdown 无序列表
  table     → HTML <table>（保留原始结构）+ caption / footnote
              caption 中的 "单位为毫米" / "单位：毫米" 等说明会被自动剔除
  image     → 图片占位 + caption / footnote
  equation  → LaTeX 数学公式块 $$ ... $$
  header / footer / page_number → 跳过（页眉页脚非正文）
"""

import json
import sys
import re
from pathlib import Path


# ---------- caption 清理 ----------

# 匹配 "单位为毫米" / "单位:毫米" / "单位：mm" 等说明，
# 允许外层包裹中文括号（）或英文括号()，也允许无括号、前后带分隔符。
# 单位词支持：毫米、mm、ｍｍ（全角）；连接词支持：为 / 是 / : / ：
_UNIT_NOTE_PATTERN = re.compile(
    r"[\(（]?\s*单位\s*[:：为是]\s*(?:毫米|mm|ｍｍ)\s*[\)）]?",
    flags=re.IGNORECASE,
)


def _clean_caption(text: str) -> str:
    """从 caption 文本中剔除 "单位为毫米" 类说明，并归一化空白。"""
    if not text:
        return ""
    cleaned = _UNIT_NOTE_PATTERN.sub("", text)
    # 清理因删除而残留的多余空格、连续标点
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([，,。；;：:])", r"\1", cleaned)
    return cleaned.strip()


# ---------- 各类型的渲染函数（不嵌入 page 注释）----------

def render_text(item):
    text = item.get("text", "").strip()
    if not text:
        return ""
    level = item.get("text_level")
    if level == 1:
        return f"# {text}\n"
    elif level == 2:
        return f"## {text}\n"
    elif level == 3:
        return f"### {text}\n"
    else:
        return f"{text}\n"


def render_list(item):
    list_items = item.get("list_items", [])
    if not list_items:
        return ""
    lines = "\n".join(f"- {li.strip()}" for li in list_items if li.strip())
    return f"{lines}\n"


def render_table(item):
    captions  = item.get("table_caption", [])
    footnotes = item.get("table_footnote", [])
    body      = item.get("table_body", "").strip()
    img_path  = item.get("img_path", "")

    parts = []
    for cap in captions:
        cap = _clean_caption(cap)
        if cap:
            parts.append(f"**{cap}**")
    if body:
        parts.append(body)
    for fn in footnotes:
        fn = fn.strip()
        if fn:
            parts.append(f"*{fn}*")
    if img_path:
        # alt 文本同样使用清理后的 caption
        caption_text = " ".join(
            _clean_caption(c) for c in captions if _clean_caption(c)
        )
        alt = caption_text if caption_text else "表格"
        parts.append(f"![{alt}]({img_path})")
    return "\n".join(parts) + "\n" if parts else ""


def render_image(item, caption_override=None):
    """渲染单张图片。

    Args:
        item: 原始 JSON 条目。
        caption_override: 外部指定的 caption 文本，优先级高于 item 自身的 caption。
    """
    captions  = item.get("image_caption", [])
    footnotes = item.get("image_footnote", [])
    img_path  = item.get("img_path", "")

    if caption_override is not None:
        caption_text = caption_override
    else:
        caption_text = " ".join(c.strip() for c in captions if c.strip())

    parts = []
    if img_path:
        alt = caption_text if caption_text else "图片"
        parts.append(f"![{alt}]({img_path})")
    for fn in footnotes:
        fn = fn.strip()
        if fn:
            parts.append(f"*{fn}*")
    return "\n".join(parts) + "\n" if parts else ""


def render_equation(item):
    text = item.get("text", "").strip()
    inner = re.sub(r"^\$\$\s*|\s*\$\$$", "", text, flags=re.DOTALL).strip()
    return f"$$\n{inner}\n$$\n"


# ---------- 辅助：提取图片的 caption 文本 ----------

def _get_image_caption_text(item):
    """返回图片条目的 caption 文本（空字符串表示无 caption）。"""
    captions = item.get("image_caption", [])
    return " ".join(c.strip() for c in captions if c.strip())


# ---------- 辅助：将缓冲的图片块写入 output_lines ----------

def _flush_pending_images(pending_images, caption_text, output_lines):
    """将挂起的无 caption 图片全部输出，使用 caption_text 作为 alt 文本。

    缓冲图片与最终带 caption 的图片本质上是同一张图的切片，
    因此它们之间不插入页码标记，页码统一跟随带 caption 的图片。
    """
    count = 0
    for img_item, _page in pending_images:
        rendered = render_image(img_item, caption_override=caption_text)
        if rendered:
            output_lines.append(rendered)
            count += 1
    return count


# ---------- 主转换逻辑 ----------

SKIP_TYPES = {"header", "footer", "page_number"}

RENDERERS = {
    "text":     render_text,
    "list":     render_list,
    "table":    render_table,
    "equation": render_equation,
    # image 不在 RENDERERS 中，由主循环单独处理
}

PAGE_MARKER = "<!-- page: {page} -->"


def convert_json_to_markdown(json_path, output_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    items = [item for item in data if item.get("type") not in SKIP_TYPES]

    output_lines = []
    page_state = {"current_page": None}
    block_count = 0

    pending_images = []

    for item in items:
        t = item.get("type", "")

        # ---- 图片特殊处理：缓冲无 caption 的图片 ----
        if t == "image":
            caption_text = _get_image_caption_text(item)
            page = item.get("page_idx")

            if not caption_text:
                pending_images.append((item, page))
                continue
            else:
                if page is not None and page != page_state["current_page"]:
                    if page_state["current_page"] is not None or page != 0:
                        output_lines.append(PAGE_MARKER.format(page=page + 1))
                    output_lines.append("")
                    page_state["current_page"] = page

                if pending_images:
                    block_count += _flush_pending_images(
                        pending_images, caption_text, output_lines
                    )
                    pending_images.clear()

                rendered = render_image(item)
                if rendered:
                    output_lines.append(rendered)
                    block_count += 1
                continue

        # ---- 非图片类型 ----
        if pending_images:
            block_count += _flush_pending_images(
                pending_images, "", output_lines
            )
            pending_images.clear()

        renderer = RENDERERS.get(t)
        if renderer is None:
            print(f"[WARN] 未知类型 '{t}'，已跳过。", file=sys.stderr)
            continue

        rendered = renderer(item)
        if not rendered:
            continue

        page = item.get("page_idx")

        if page is not None and page != page_state["current_page"]:
            if page_state["current_page"] is not None or page != 0:
                output_lines.append(PAGE_MARKER.format(page=page + 1))
            output_lines.append("")
            page_state["current_page"] = page

        output_lines.append(rendered)
        block_count += 1

    if pending_images:
        block_count += _flush_pending_images(
            pending_images, "", output_lines
        )
        pending_images.clear()

    md_content = "\n".join(output_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    marker_count = sum(1 for line in output_lines if line.startswith("<!-- page:"))
    print(
        f"[OK] 转换完成：{output_path}\n"
        f"     正文块：{block_count} 个 | 页面分隔标记：{marker_count} 处"
    )


# ---------- 入口 ----------

if __name__ == "__main__":
    input_file  = "C:\\Users\\Lenovo\\Desktop\\project_2_3.29\\project_2\\f_55l2wt09\\minerU\\153c58a3-9a4b-41df-80d7-82eae3088d10_content_list.json"
    output_file = "C:\\Users\\Lenovo\\Desktop\\project_2_3.29\\project_2\\backend\\data\\f_55l2wt09\\output.md"

    if not Path(input_file).exists():
        print(f"[ERROR] 找不到输入文件：{input_file}", file=sys.stderr)
        sys.exit(1)

    convert_json_to_markdown(input_file, output_file)