# services/mineru_service.py
# ============================================================
#  MinerU PDF 解析服务
#  职责：上传 PDF → 提交云端解析 → 轮询结果 → 下载解压 → 渲染带框页面图
# ============================================================

# ── 标准库 ──────────────────────────────────────────────────
import os
import io
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, Any

# ── 第三方库 ────────────────────────────────────────────────
import fitz          # PyMuPDF：PDF 渲染 / 画框
import requests      # HTTP 客户端
from requests.exceptions import RequestException, Timeout

# ── 本项目内部模块 ───────────────────────────────────────────
from services.pdf_service import (
    workdir,
    dir_parsed_pages,
    markdown_output,
    images_dir,
    original_pdf_path,   # 放在顶层，避免函数内循环导入
)


# ============================================================
#  自定义异常
#  用具体异常类替代裸 Exception，方便上层按类型捕获
# ============================================================
class MinerUError(Exception):
    """MinerU 解析流程中发生的业务异常"""


class MinerUTimeoutError(MinerUError):
    """MinerU 轮询超时"""


# ============================================================
#  第一段：上传文件到 Catbox，获取公网直链
#  输入：本地文件路径
#  输出：成功返回 https://files.catbox.moe/xxx 链接，失败返回 None
# ============================================================
def upload_to_catbox(local_path: str) -> str | None:
    """
    将本地 PDF 上传到 Catbox，返回可供 MinerU 云端拉取的公网直链。
    Catbox 文件永久保存，无"阅后即焚"限制。
    """
    CATBOX_API = "https://catbox.moe/user/api.php"
    # Catbox 对单文件大小有限制（200 MB），此处记录文件大小方便排查
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"[*] [Catbox] 准备上传: {os.path.basename(local_path)} ({file_size_mb:.2f} MB)")

    try:
        with open(local_path, "rb") as f:
            # Catbox multipart 格式：reqtype 固定为 fileupload
            files = {
                "reqtype":     (None, "fileupload"),
                "fileToUpload": (os.path.basename(local_path), f),   
            }
            # timeout=(连接超时, 读取超时)，防止大文件上传时进程永久挂起
            response = requests.post(CATBOX_API, files=files, timeout=60)

        # Catbox 正常返回 200 + 文本直链，异常时返回错误描述字符串
        if response.status_code != 200:
            print(f"[!] [Catbox] HTTP 状态异常: {response.status_code}")
            return None

        link = response.text.strip()

        # 严格校验返回域名，防止误把错误提示当成链接使用
        if link.startswith("https://files.catbox.moe/"):
            print(f"[+] [Catbox] 上传成功: {link}")
            return link

        print(f"[!] [Catbox] 返回内容不符合预期: {link[:200]}")
        return None

    except Timeout:
        # 单独捕获超时，给出更明确的提示
        print(f"[!] [Catbox] 上传超时，文件可能过大或网络不稳定")
        return None
    except RequestException as e:
        print(f"[!] [Catbox] 网络请求异常: {e}")
        return None
    except OSError as e:
        # 文件不存在 / 无读取权限等磁盘错误
        print(f"[!] [Catbox] 文件读取失败: {e}")
        return None


# ============================================================
#  第二段：定位 MinerU 解析结果的存放目录
#  规则：data/{file_id}/minerU/
# ============================================================
def minerU_result(file_id: str) -> Path:
    """返回（并确保存在）当前任务的 MinerU 输出目录。"""
    p = workdir(file_id) / "minerU"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
#  第三段：根据 layout.json 在原始 PDF 上绘制元素框，生成前端展示图
#  输入：file_id（定位原始 PDF）、layout.json 路径
#  输出：data/{file_id}/parsed_pages/page-XXXX.png
# ============================================================

# 不同 block 类型对应的 RGB 颜色，集中管理方便后续扩展
_BLOCK_COLORS: Dict[str, tuple] = {
    "title":  (0.0, 0.0, 1.0),   # 蓝色 —— 标题 / 页眉
    "header": (0.0, 0.0, 1.0),
    "table":  (0.0, 0.8, 0.0),   # 绿色 —— 表格 / 图片
    "image":  (0.0, 0.8, 0.0),
}
_DEFAULT_COLOR = (1.0, 0.0, 0.0)  # 红色 —— 普通正文


def _is_valid_bbox(bbox) -> bool:
    """校验 bbox 是否为包含 4 个数值的列表，防止脏数据导致 fitz 崩溃。"""
    return (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
    )


def render_mineru_layout_pages(file_id: str, layout_json_path: Path) -> None:
    """
    读取 MinerU 返回的 layout.json，在原始 PDF 每页上叠加元素检测框，
    并将结果渲染为 PNG 图片供前端展示。
    """
    print(f"[*] 正在读取布局数据: {layout_json_path}")

    # ── 3.1 加载并解析 layout.json ──────────────────────────
    try:
        with open(layout_json_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
    except json.JSONDecodeError as e:
        # JSON 格式损坏时给出明确错误，不能静默继续
        raise MinerUError(f"layout.json 解析失败（文件可能损坏）: {e}") from e
    except OSError as e:
        raise MinerUError(f"无法读取 layout.json: {e}") from e

    # ── 3.2 打开原始 PDF ────────────────────────────────────
    pdf_path = original_pdf_path(file_id)
    if not pdf_path.exists():
        raise MinerUError(f"找不到原始 PDF: {pdf_path}")

    output_dir = dir_parsed_pages(file_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_info = layout_data.get("pdf_info", [])
    if not pdf_info:
        print("[!] layout.json 中 pdf_info 为空，跳过渲染")
        return

    # with 语句确保 PDF 文件句柄在任何情况下都被正确关闭，防止资源泄漏
    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)

        for page_data in pdf_info:
            page_idx = page_data.get("page_idx", 0)

            # ── 3.3 页码越界保护 ─────────────────────────────
            if page_idx >= total_pages:
                # 打印警告而非静默跳过，便于排查 JSON 与 PDF 不一致的问题
                print(f"[!] 警告: layout.json 页码 {page_idx} 超出 PDF 总页数 {total_pages}，已跳过")
                continue

            page = doc[page_idx]

            # ── 3.4 遍历当前页所有检测块，绘制标注框 ──────────
            for block in page_data.get("para_blocks", []):
                bbox       = block.get("bbox")
                block_type = block.get("type", "unknown")

                # 跳过坐标数据缺失或格式异常的块
                if not _is_valid_bbox(bbox):
                    continue

                rect  = fitz.Rect(bbox)
                color = _BLOCK_COLORS.get(block_type, _DEFAULT_COLOR)

                # 在页面上绘制矩形标注框
                page.draw_rect(rect, color=color, width=1.5)

                # 在框左上角插入类型标签，Y 轴稍微上移避免遮挡正文
                page.insert_text(
                    (bbox[0], max(0, bbox[1] - 3)),
                    block_type,
                    fontsize=8,
                    color=color,
                )

            # ── 3.5 渲染当前页为 PNG（2× 缩放保证清晰度）────────
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_path = output_dir / f"page-{page_idx + 1:04d}.png"
            pix.save(str(img_path))
            print(f"[+] 第 {page_idx + 1}/{total_pages} 页渲染完成 → {img_path.name}")


# ============================================================
#  第四段：完整的 MinerU 云端解析主流程
#  步骤：提交任务 → 轮询状态 → 下载 ZIP → 解压 → 整理文件结构
#  输入：公网可访问的 PDF URL、file_id、MinerU API Token
#  输出：{"status": "success", "md": "<output.md 的本地路径>"}
# ============================================================

# MinerU API 基础地址，集中定义方便切换测试/生产环境
_MINERU_BASE = "https://mineru.net/api/v4"
# 单次轮询最长等待时间（秒）
_POLL_TIMEOUT_SEC = 600
# 连续请求失败多少次后放弃轮询
_MAX_POLL_FAILURES = 5


def process_pdf_with_mineru(file_url: str, file_id: str, token: str) -> Dict[str, Any]:
    """
    将 PDF 提交到 MinerU 云端，等待解析完成后下载结果并整理到本地标准目录。
    成功返回 {"status": "success", "md": "output.md 路径"}。
    失败抛出 MinerUError 或 MinerUTimeoutError。
    """
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {token}",
    }

    # ── 4.1 提交解析任务 ─────────────────────────────────────
    print(f"[*] [MinerU] 提交解析任务 (file_id={file_id})")
    print(f"[*] [MinerU] PDF URL 前缀: {file_url[:60]}...")

    try:
        res = requests.post(
            f"{_MINERU_BASE}/extract/task",
            headers=headers,
            json={"url": file_url, "model_version": "vlm"},
            timeout=(10, 30),   # (连接超时 10s, 等待响应 30s)
        )
        res.raise_for_status()  # 4xx / 5xx 时立即抛出异常
        res_json = res.json()
    except Timeout:
        raise MinerUError("提交任务超时，请检查网络或 MinerU 服务状态")
    except RequestException as e:
        raise MinerUError(f"提交任务网络异常: {e}") from e
    except ValueError as e:
        raise MinerUError(f"提交任务响应非 JSON: {e}") from e

    if res_json.get("code") != 0:
        raise MinerUError(f"MinerU 拒绝任务: {res_json.get('msg', '未知错误')}")

    task_id = res_json["data"]["task_id"]
    print(f"[+] [MinerU] 任务提交成功，TaskID: {task_id}")

    # ── 4.2 轮询任务状态 ─────────────────────────────────────
    full_zip_url = ""
    start_time     = time.time()
    failure_count  = 0   # 记录连续请求失败次数，避免因偶发网络抖动直接中止

    while True:
        elapsed = int(time.time() - start_time)

        # 总时长保护，防止任务卡死时进程永久阻塞
        if elapsed > _POLL_TIMEOUT_SEC:
            raise MinerUTimeoutError(f"MinerU 解析超时（>{_POLL_TIMEOUT_SEC}s），TaskID: {task_id}")

        try:
            poll_res = requests.get(
                f"{_MINERU_BASE}/extract/task/{task_id}",
                headers=headers,
                timeout=(10, 30),
            )
            poll_res.raise_for_status()
            poll_json = poll_res.json()
            failure_count = 0  # 成功后重置失败计数
        except (RequestException, ValueError) as e:
            failure_count += 1
            print(f"[!] [MinerU] 轮询请求失败 ({failure_count}/{_MAX_POLL_FAILURES}): {e}")
            if failure_count >= _MAX_POLL_FAILURES:
                raise MinerUError(f"连续 {_MAX_POLL_FAILURES} 次轮询失败，放弃任务") from e
            time.sleep(5)
            continue

        # 用 .get() 链式访问，防止服务端返回格式异常时 KeyError 崩溃
        data  = poll_json.get("data") or {}
        state = data.get("state", "unknown")
        print(f"[*] [MinerU] 状态: {state} (已耗时 {elapsed}s)")

        if state == "done":
            full_zip_url = data.get("full_zip_url", "")
            if not full_zip_url:
                raise MinerUError("任务完成但 full_zip_url 为空")
            print(f"[+] [MinerU] 解析完成，ZIP 链接已获取")
            break
        elif state == "failed":
            raise MinerUError(f"MinerU 云端解析失败: {data.get('err_msg', '未知原因')}")

        time.sleep(5)

    # ── 4.3 下载结果 ZIP ─────────────────────────────────────
    print(f"[*] [MinerU] 正在下载结果 ZIP...")
    try:
        # stream=False 适合 <100MB 的结果包；更大文件建议改为 stream=True 分块写盘
        zip_resp = requests.get(full_zip_url, timeout=(10, 120))
        zip_resp.raise_for_status()
    except Timeout:
        raise MinerUError("下载结果 ZIP 超时，文件可能过大")
    except RequestException as e:
        raise MinerUError(f"下载结果 ZIP 失败: {e}") from e

    zip_bytes = zip_resp.content
    print(f"[*] [MinerU] 下载完成，大小: {len(zip_bytes)/1024:.2f} KB")

    # ── 4.4 安全解压（防止 zip slip 路径穿越攻击）────────────
    target_dir = minerU_result(file_id)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # zip slip：恶意压缩包可能包含 ../../../etc/passwd 等路径，必须过滤
        for member in z.namelist():
            member_path = Path(member)
            # 检查路径中是否含有 ".." 分量或以 "/" 开头的绝对路径
            if ".." in member_path.parts or member_path.is_absolute():
                raise MinerUError(f"检测到不安全的压缩包路径，拒绝解压: {member}")
        z.extractall(target_dir)

    print(f"[*] [MinerU] 解压到: {target_dir}")

    # ── 4.5 整理文件：渲染带框页面图 ────────────────────────
    layout_jsons = list(target_dir.rglob("*layout.json"))
    if layout_jsons:
        print(f"[+] [MinerU] 找到布局文件: {layout_jsons[0].name}，开始渲染...")
        render_mineru_layout_pages(file_id, layout_jsons[0])
    else:
        # 找不到布局文件不应致命，只记录警告，后续步骤继续执行
        print("[!] [MinerU] 警告: 未找到 layout.json，跳过带框图渲染")

    # ── 4.6 整理文件：复制主 Markdown 到标准路径，不使用page来定位当前页 ─────────────
    # md_files = list(target_dir.rglob("*.md"))
    # if md_files:
    #     # 取第一个 .md（MinerU 通常只产出一个），复制到 data/{file_id}/output.md
    #     shutil.copy(md_files[0], markdown_output(file_id))
    #     print(f"[+] [MinerU] Markdown 已复制 → {markdown_output(file_id)}")
    # else:
    #     print("[!] [MinerU] 警告: 未找到任何 .md 文件")
    
    #+ —— v2 4.6.1 使用json_to_md.py转换Markdown，确保页面分隔标记正确插入，供前端按页展示 ——
    # 精确匹配 _content_list.json 后缀，唯一定位目标文件
    json_files = list(target_dir.glob("*_content_list.json"))

    if json_files:
        json_md_input = json_files[0]
        from services.json_to_md import convert_json_to_markdown
        convert_json_to_markdown(json_md_input, markdown_output(file_id))
        print(f"[+] [MinerU] JSON 已转换为 Markdown → {markdown_output(file_id)}")
    else:
        print("[?] [MinerU] 未找到 _content_list.json，跳过步骤")
    
    #+ —— v2 4.6.2 对整理的文件进行自定义的数据清洗，保证md文件的结构清晰，便于构建基于md结构的chunks，还有删除一些边边角角的冗余元数据 ——
    from services.md_cleaning import MarkdownHeadingCleaner
    cleaner = MarkdownHeadingCleaner()
    # 直接在原文件上进行清洗，覆盖写入清洗后的内容
    cleaner.process_file(str(markdown_output(file_id)), str(markdown_output(file_id)))
    print(f"[+] [MinerU] Markdown 已清洗 → {markdown_output(file_id)}")
          

    # ── 4.7 整理文件：合并图片目录到标准路径 ────────────────
    img_dirs = [d for d in target_dir.rglob("images") if d.is_dir()]
    if img_dirs:
        # dirs_exist_ok=True：目标目录已存在时合并而非报错
        shutil.copytree(img_dirs[0], images_dir(file_id), dirs_exist_ok=True)
        print(f"[+] [MinerU] 图片已合并 → {images_dir(file_id)}")
    else:
        print("[?] [MinerU] 未找到图片目录，跳过图片同步")

    print(" [MinerU] 全部流程完成！")
    return {"status": "success", "md": str(markdown_output(file_id))}