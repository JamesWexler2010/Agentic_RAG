# services/pdf_service.py
from __future__ import annotations
import os, io, math, json
from pathlib import Path
from typing import Dict, Any, List
import fitz
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # 服务器无头
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#+—— windows不使用unstructured ——
# from langchain_unstructured import UnstructuredLoader
# from unstructured.partition.pdf import partition_pdf
from html2text import html2text

# 统一的根目录：每个 fileId 一个子目录
DATA_ROOT = Path("data")

def workdir(file_id: str) -> Path:
    d = DATA_ROOT / file_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def dir_original_pages(file_id: str) -> Path:
    p = workdir(file_id) / "pages" / "original"
    p.mkdir(parents=True, exist_ok=True); return p

def dir_parsed_pages(file_id: str) -> Path:
    p = workdir(file_id) / "pages" / "parsed"
    p.mkdir(parents=True, exist_ok=True); return p

def original_pdf_path(file_id: str) -> Path:
    return workdir(file_id) / "original.pdf"

def markdown_output(file_id: str) -> Path:
    return workdir(file_id) / "output.md"

def images_dir(file_id: str) -> Path:
    p = workdir(file_id) / "images"
    p.mkdir(parents=True, exist_ok=True); return p

def save_upload(file_id: str, upload_bytes: bytes, filename: str) -> Dict[str, Any]:
    """保存上传的 PDF，并返回页数"""
    pdf_path = original_pdf_path(file_id)
    pdf_path.write_bytes(upload_bytes)
    with fitz.open(pdf_path) as doc:
        pages = doc.page_count
    return {"fileId": file_id, "name": filename, "pages": pages}


def render_original_pages(file_id: str, dpi: int = 144):
    """把原始 PDF 渲染为 PNG，存到 pages/original/"""
    pdf_path = original_pdf_path(file_id)
    out_dir = dir_original_pages(file_id)
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            (out_dir / f"page-{idx:04d}.png").write_bytes(pix.tobytes("png"))

def _plot_boxes_to_ax(ax, pix, segments):
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    categories = set()
    for seg in segments:
        points = seg["coordinates"]["points"]
        lw = seg["coordinates"]["layout_width"]
        lh = seg["coordinates"]["layout_height"]
        scaled = [(x * pix.width / lw, y * pix.height / lh) for x, y in points]
        color = category_to_color.get(seg.get("category"), "deepskyblue")
        categories.add(seg.get("category", "Text"))
        poly = patches.Polygon(scaled, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(poly)

    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for cat, color in category_to_color.items():
        if cat in categories:
            legend_handles.append(patches.Patch(color=color, label=cat))
    ax.legend(handles=legend_handles, loc="upper right")
'''
def render_parsed_pages_with_boxes(file_id: str, docs_local: List[Dict[str, Any]], dpi: int = 144):
    """
    根据 UnstructuredLoader 的 metadata（含坐标）在原图上叠框，输出到 pages/parsed/
    """
    pdf_path = original_pdf_path(file_id)
    out_dir = dir_parsed_pages(file_id)
    with fitz.open(pdf_path) as doc:
        # 预聚合：按 page_number 分组 segments
        segments_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for d in docs_local:
            meta = d.metadata if hasattr(d, "metadata") else d["metadata"]
            pno = meta.get("page_number")
            if pno is None: continue
            segments_by_page.setdefault(pno, []).append(meta)

        for page_number in range(1, doc.page_count + 1):
            page = doc.load_page(page_number - 1)
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(pil)
            ax.axis("off")
            _plot_boxes_to_ax(ax, pix, segments_by_page.get(page_number, []))
            fig.tight_layout()
            fig.savefig(out_dir / f"page-{page_number:04d}.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
'''
def render_parsed_pages_with_boxes(file_id: str, dpi: int = 144):
    """用 pymupdf 提取文字块坐标，在页面图上叠框"""
    pdf_path = original_pdf_path(file_id)
    out_dir = dir_parsed_pages(file_id)
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            scale_x = pix.width / page.rect.width
            scale_y = pix.height / page.rect.height

            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(pil)
            ax.axis("off")

            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                x0, y0, x1, y1 = block["bbox"]
                rect = plt.Rectangle(
                    (x0 * scale_x, y0 * scale_y),
                    (x1 - x0) * scale_x,
                    (y1 - y0) * scale_y,
                    linewidth=1, edgecolor="deepskyblue", facecolor="none"
                )
                ax.add_patch(rect)

            fig.tight_layout()
            fig.savefig(out_dir / f"page-{page_number:04d}.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)

'''
def unstructured_segments(file_id: str) -> List[Any]:
    """用 UnstructuredLoader 产生高分辨率布局段"""
    pdf_path = str(original_pdf_path(file_id))
    loader = UnstructuredLoader(
        file_path=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        ocr_languages="chi_sim+eng",
        ocr_engine="paddleocr",  # 如果装不上可换成 'auto' 或注释掉
    )
    out = []
    for d in loader.lazy_load():
        out.append(d)
    return out

def pdf_to_markdown(file_id: str):
    pdf_path = str(original_pdf_path(file_id))
    out_md = markdown_output(file_id)
    img_dir = images_dir(file_id)

    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy="hi_res",
        ocr_languages="chi_sim+eng",
        ocr_engine="paddleocr"  # 同上
    )

    # 提取图片
    image_map = {}
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            image_map[page_num] = []
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                img_path = img_dir / f"page{page_num}_img{img_index}.png"
                if pix.n < 5:
                    pix.save(str(img_path))
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(str(img_path))
                image_map[page_num].append(img_path.name)  # 只保存文件名

    md_lines: List[str] = []
    inserted_images = set()
    for el in elements:
        cat = getattr(el, "category", None)
        text = (getattr(el, "text", "") or "").strip()
        meta = getattr(el, "metadata", None)
        page_num = getattr(meta, "page_number", None) if meta else None

        if not text and cat != "Image":
            continue

        if cat == "Title" and text.startswith("- "):
            md_lines.append(text + "\n")
        elif cat == "Title":
            md_lines.append(f"# {text}\n")
        elif cat in ["Header", "Subheader"]:
            md_lines.append(f"## {text}\n")
        elif cat == "Table":
            html = getattr(meta, "text_as_html", None) if meta else None
            if html:
                md_lines.append(html2text(html) + "\n")
            else:
                md_lines.append((text or "") + "\n")
        elif cat == "Image" and page_num:
            for name in image_map.get(page_num, []):
                if (page_num, name) not in inserted_images:
                    md_lines.append(f"![Image](./images/{name})\n")
                    inserted_images.add((page_num, name))
        else:
            md_lines.append(text + "\n")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return {"markdown": out_md.name, "images_dir": "images"}
'''

'''
def run_full_parse_pipeline(file_id: str) -> Dict[str, Any]:
    """
    完整流程：原始页图渲染 → Unstructured 布局段 → 叠框图 → 输出 Markdown
    返回用于 /status 的统计或元信息
    """
    render_original_pages(file_id)
    docs = unstructured_segments(file_id)
    render_parsed_pages_with_boxes(file_id, docs)
    md_info = pdf_to_markdown(file_id)
    return {"md": md_info["markdown"]}
'''
def save_metadata(file_id: str, metadata: Dict[str, Any]):
    """将元数据持久化到文件目录下的 meta.json"""
    meta_path = workdir(file_id) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[*] Metadata saved for {file_id}")
'''
def render_mineru_layout_pages(file_id, layout_json_path):
    """
    根据 MinerU 返回的 layout.json，在原始 PDF 上画框并生成前端展示图片
    """
    print(f"[*] 正在读取 JSON 布局数据: {layout_json_path}")
    
    # 1. 加载 layout.json 数据
    with open(layout_json_path, 'r', encoding='utf-8') as f:
        layout_data = json.load(f)

    # 2. 打开你本地的原始 PDF
    from services.pdf_service import original_pdf_path, dir_parsed_pages
    pdf_path = original_pdf_path(file_id)
    if not pdf_path.exists():
        print(f"[!] 画框失败：找不到原始 PDF 文件 {pdf_path}")
        return

    doc = fitz.open(str(pdf_path))
    
    # 3. 准备保存图片的输出目录
    output_dir = dir_parsed_pages(file_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 遍历 JSON 里的每一页进行画框
    pdf_info = layout_data.get("pdf_info", [])
    
    for page_data in pdf_info:
        page_idx = page_data.get("page_idx", 0)
        
        # 安全校验，防止 JSON 页码超出实际 PDF 页数
        if page_idx >= len(doc):
            continue
            
        page = doc[page_idx]
        
        # 提取正文块 (文本、标题、表格、图片等)
        blocks = page_data.get("para_blocks", [])
        
        # 如果你想把页眉、页码也框出来，取消下面这行的注释：
        # blocks.extend(page_data.get("discarded_blocks", []))

        for block in blocks:
            bbox = block.get("bbox")
            block_type = block.get("type", "unknown")
            
            if bbox and len(bbox) == 4:
                # 转换坐标为 PyMuPDF 支持的 Rect 格式
                rect = fitz.Rect(bbox)
                
                # 为了美观，根据不同类型设置不同框的颜色 (RGB)
                if block_type in ["title", "header"]:
                    color = (0, 0, 1)  # 标题用蓝色
                elif block_type in ["table", "image"]:
                    color = (0, 0.8, 0)  # 表格和图片用绿色
                else:
                    color = (1, 0, 0)  # 普通文本用红色
                
                # 在页面上画矩形框 (width=1 代表线条粗细)
                page.draw_rect(rect, color=color, width=1.5)
                
                # 在框的左上角写上类型的文字标签（可选，如果不想要文字可以删掉这一段）
                page.insert_text(
                    (bbox[0], max(0, bbox[1] - 3)), # Y坐标往上抬一点，防止挡住文字
                    str(block_type), 
                    fontsize=8, 
                    color=color
                )

        # 5. 将画完框的页面渲染成图片并保存 (2倍缩放保证清晰度)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_path = output_dir / f"page-{page_idx + 1:04d}.png"
        pix.save(str(img_path))
        print(f"[+] 渲染成功: 第 {page_idx + 1} 页已生成带框图片 -> {img_path.name}")
    '''