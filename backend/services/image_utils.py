# services/image_utils.py
"""
图片 URL 工具:把后端磁盘上的相对图片路径(images/xxx.png)
重写为前端可访问的 API URL,并从 Markdown 文本中提取图片列表。

被 rag_service 和 graph_service 共用。

后端图片实际存储位置(由 pdf_service.images_dir 决定):
    data/<file_id>/images/<filename>

前端访问端点(由 app.py 的 /api/v1/pdf/images 路由提供):
    GET /api/v1/pdf/images?fileId=<file_id>&imagePath=<filename>
"""
from __future__ import annotations
import os
import re
from typing import List, Dict, Any


# 匹配 Markdown 图片:![alt](images/xxx.png) / ![alt](./images/xxx.png) / ![alt](/images/xxx.png)
_IMG_RE = re.compile(r'!\[([^\]]*)\]\(\.?/?images/([^)]+)\)')


def get_backend_base() -> str:
    """
    动态获取后端基地址(每次调用时读取,适配 ngrok URL 随时变化的场景)。

    判断优先级:
      1. BACKEND_BASE_URL  —— 显式指定,生产环境推荐设置此项
      2. NGROK_URL          —— 存在即视为本地 ngrok 开发环境
      3. 空串               —— 同源部署(前后端同域),不需要前缀
    """
    return (
        os.getenv("BACKEND_BASE_URL")
        or os.getenv("NGROK_URL")
        or ""
    ).rstrip("/")


def _build_image_api_url(file_id: str, image_filename: str) -> str:
    """统一构造 /api/v1/pdf/images 的访问 URL。"""
    base = get_backend_base()
    return f"{base}/api/v1/pdf/images?fileId={file_id}&imagePath={image_filename}"


def rewrite_image_urls(text: str, file_id: str) -> str:
    """
    重写 Markdown 文本中的相对图片路径为后端 API URL。

    输入:  ![alt](images/foo.png)
    输出:  ![alt](http://host/api/v1/pdf/images?fileId=xxx&imagePath=foo.png)
    """
    if not text or not file_id:
        return text
    return _IMG_RE.sub(
        lambda m: f'![{m.group(1)}]({_build_image_api_url(file_id, m.group(2))})',
        text,
    )


def extract_image_urls(text: str, file_id: str) -> List[Dict[str, str]]:
    """
    从 Markdown 文本中提取所有图片,返回结构化列表。

    每项为 {"alt": "图片描述", "url": "http://backend/api/v1/pdf/images?..."}。
    前端可直接遍历渲染 <img>。
    """
    if not text or not file_id:
        return []
    images = []
    for m in _IMG_RE.finditer(text):
        alt = m.group(1) or "图片"
        filename = m.group(2)
        images.append({"alt": alt, "url": _build_image_api_url(file_id, filename)})
    return images


def normalize_table_img_paths(raw_paths: Any, file_id: str) -> List[Dict[str, str]]:
    """
    把图谱 citation 的 table_img_path 字段(可能是 str / list[str] / None / 其他)
    统一适配为 [{"alt", "url"}] 列表,与 extract_image_urls 产出一致。

    table_img_path 里存的是相对路径(如 "images/table_5.png" 或裸文件名 "table_5.png"
    或绝对路径),这里统一抽出文件名加上 API 前缀,前端拿到就能直接显示。
    """
    if not raw_paths or not file_id:
        return []

    # 统一成 list
    if isinstance(raw_paths, str):
        paths = [raw_paths]
    elif isinstance(raw_paths, list):
        paths = [p for p in raw_paths if p]
    else:
        return []

    images = []
    for p in paths:
        # 兼容 "images/foo.png" / "foo.png" / "/abs/path/foo.png" / Windows 路径,只取文件名
        filename = str(p).replace("\\", "/").split("/")[-1]
        if not filename:
            continue
        images.append({
            "alt": filename,
            "url": _build_image_api_url(file_id, filename),
        })
    return images