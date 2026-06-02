from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio, time, os, random, string
from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
#+——— 新增:引入相关依赖 ———
import shutil 
import json
from pathlib import Path
from fastapi import BackgroundTasks
from services.pdf_service import (
    save_upload,  render_original_pages, 
    original_pdf_path, dir_original_pages, dir_parsed_pages, markdown_output, save_metadata
)  

from services.index_service import build_faiss_index, search_faiss
from fastapi.responses import StreamingResponse, JSONResponse
from services.rag_service import retrieve, answer_stream, clear_history

# ✨ 新增:引入图谱服务(与 rag_service 对称,提供图谱模式的检索 + 流式生成)
from services.graph_service import retrieve_graph, answer_stream_graph, invalidate_agent_cache

#+——— 引入必要的服务依赖 MinerU ———
from services.mineru_service import process_pdf_with_mineru, upload_to_catbox
from services.pdf_service import original_pdf_path
#+————————————————————————————————



app = FastAPI(
    title="船舶制造RAG系统API",
    version="1.0.0",
    description="船舶制造RAG系统后端API。"
)

# 允许前端本地联调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"

# ---------------- 内存态存储(教学Mock) ----------------
current_pdf: Dict[str, Any] = {
    "fileId": None,
    "name": None,
    "pages": 0,
    "status": "idle",      # idle | parsing | ready | error
    "progress": 0
}
citations: Dict[str, Dict[str, Any]] = {}   # citationId -> { fileId, page, snippet, bbox, previewUrl }

# ---------------- 工具函数 ----------------
def rid(prefix: str) -> str:
    return f"{prefix}_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def now_ts() -> int:
    return int(time.time())

def err(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}, "requestId": rid("req"), "ts": now_ts()}

# ---------------- Pydantic 模型(契约) ----------------
# ✨ 合并:之前 ChatRequest 被定义了两次,这里合并为唯一定义,带 mode 字段
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    pdfFileId: Optional[str] = None
    mode: Optional[str] = "pdf"  # pdf | graph,默认 pdf

# ---------------- Health ----------------
@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health():
    return {"ok": True, "version": "1.0.0"}

# ---------------- Chat(SSE,POST 返回 event-stream) ----------------

@app.post(f"{API_PREFIX}/chat", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """
    SSE 事件:token | citation | done | error
    根据 req.mode 分流:
      - mode == "pdf":   走 rag_service(FAISS 向量检索 + 流式 LLM)
      - mode == "graph": 走 graph_service(GraphRAG Agent + 伪流式)
    """
    async def gen():
        try:
            question = (req.message or "").strip()
            session_id = (req.sessionId or "default").strip()
            file_id = (req.pdfFileId or "").strip()
            mode = (req.mode or "pdf").strip()

            # =======================================================
            # 图谱模式:调用 graph_service,与 PDF 分支使用同一套 SSE 转发逻辑
            # =======================================================
            if mode == "graph":
                graph_citations, graph_ctx = [], ""
                branch = "no_context"
                try:
                    graph_citations, graph_ctx = await retrieve_graph(question, file_id)
                    # retrieve_graph 返回非空 context_text("graph_mode") 即代表已检索成功
                    branch = "with_context" if graph_ctx else "no_context"
                except Exception as e:
                    print(f"[chat:graph] retrieve_graph 失败: {e}")
                    branch = "no_context"

                # 缓存图谱 citations 到全局字典(供前端 /pdf/chunk 接口后续查询)
                # 注意:citation_id 形如 "{fileId}-g{idx}",与 PDF 的 "-c{idx}" 区分,不会互相覆盖
                if branch == "with_context" and graph_citations:
                    for c in graph_citations:
                        citations[c["citation_id"]] = {
                            "id":          c["citation_id"],
                            "fileId":      c["fileId"],
                            "page":        c.get("page", 1),
                            "pages":       c.get("pages", [1]),
                            "snippet":     c.get("snippet", ""),
                            "bbox":        [],
                            "previewUrl":  c.get("previewUrl", ""),
                            "previewUrls": [],
                            "images":      c.get("images", []),
                            # ---- 图谱专属字段(前端可选用于差异化展示)----
                            "source":         c.get("source", "graph"),
                            "type":           c.get("type", ""),
                            "title":          c.get("title", ""),
                            "markdown":       c.get("markdown", ""),
                            "summary":        c.get("summary", ""),
                            "section_name":   c.get("section_name", ""),
                            "section_path":   c.get("section_path", ""),
                            "structure_info": c.get("structure_info", ""),
                            "level":          c.get("level"),
                            "table_name":     c.get("table_name", ""),
                            "table_img_path": c.get("table_img_path", []),
                        }

                # 流式推送:answer_stream_graph 多传一个 file_id 用于取暂存答案
                async for evt in answer_stream_graph(
                    question=question,
                    citations=graph_citations,
                    context_text=graph_ctx,
                    branch=branch,
                    session_id=session_id,
                    file_id=file_id,
                ):
                    if evt["type"] == "token":
                        yield "event: token\n"
                        text = evt["data"].replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
                        yield f'data: {{"text":"{text}"}}\n\n'
                    elif evt["type"] == "citation":
                        yield "event: citation\n"
                        yield f"data: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
                    elif evt["type"] == "done":
                        used = "true" if evt["data"].get("used_retrieval") else "false"
                        yield "event: done\n"
                        yield f'data: {{"used_retrieval": {used}}}\n\n'
                    elif evt["type"] == "graph_query":          # ✨ 新加这一段
                        yield "event: graph_query\n"
                        yield f"data: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"                        
                    elif evt["type"] == "error":
                        yield "event: error\n"
                        msg = evt["data"].get("message", "未知错误")
                        esc = msg.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
                        yield f'data: {{"message":"{esc}"}}\n\n'
                return

            # =======================================================
            # 下面是原有的 PDF 处理逻辑 (当 mode == "pdf" 时执行)
            # =======================================================
            retrieved_citations, context_text = [], ""
            branch = "no_context"
            if file_id:
                try:
                    retrieved_citations, context_text = await retrieve(question, file_id)
                    branch = "with_context" if context_text else "no_context"
                except FileNotFoundError:
                    branch = "no_context"

            # 写入 citations 字典
            if branch == "with_context" and retrieved_citations:
                for c in retrieved_citations:
                    citations[c["citation_id"]] = {
                        "id":          c["citation_id"],
                        "fileId":      c["fileId"],
                        "page":        c["page"],
                        "pages":       c["pages"],
                        "snippet":     c["snippet"],
                        "bbox":        [],
                        "previewUrl":  c["previewUrl"],
                        "previewUrls": [
                            f"/api/v1/pdf/page?fileId={c['fileId']}&page={p}&type=original"
                            for p in c["pages"]
                        ],
                        "images":      c.get("images", []),
                    }

            # 推送 token 流
            async for evt in answer_stream(
                question=question,
                citations=retrieved_citations,
                context_text=context_text,
                branch=branch,
                session_id=session_id
            ):
                if evt["type"] == "token":
                    yield "event: token\n"
                    text = evt["data"].replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
                    yield f'data: {{"text":"{text}"}}\n\n'
                elif evt["type"] == "citation":
                    yield "event: citation\n"
                    yield f"data: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
                elif evt["type"] == "done":
                    used = "true" if evt["data"].get("used_retrieval") else "false"
                    yield "event: done\n"
                    yield f'data: {{"used_retrieval": {used}}}\n\n'
                elif evt["type"] == "error":
                    yield "event: error\n"
                    msg = evt["data"].get("message", "未知错误")
                    esc = msg.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
                    yield f'data: {{"message":"{esc}"}}\n\n'

        except Exception as e:
            yield "event: error\n"
            esc = str(e).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"message":"{esc}"}}\n\n'

    headers = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ---------------- Chat: 清除对话 ----------------
class ClearChatRequest(BaseModel):
    sessionId: Optional[str] = None

@app.post(f"{API_PREFIX}/chat/clear", tags=["Chat"])
async def chat_clear(req: ClearChatRequest):
    sid = (req.sessionId or "default").strip()
    clear_history(sid)
    return {"ok": True, "sessionId": sid, "cleared": True}


# ---------------- PDF: 上传(仅单文件,直接替换) ----------------

current_pdf = {"fileId": None, "name": None, "pages": 0, "status": "idle", "progress": 0}

@app.post(f"{API_PREFIX}/pdf/upload", tags=["PDF"])
async def pdf_upload(file: UploadFile = File(...), replace: Optional[bool] = True):
    if not file:
        return JSONResponse(err("NO_FILE", "缺少文件"), status_code=400)
    # 生成新的 fileId(替换策略:上传即替换)
    fid = rid("f")
    saved = save_upload(fid, await file.read(), file.filename)
    current_pdf.update({**saved, "status": "idle", "progress": 0})
    citations.clear()
    return saved

#+ ---------------- 暴漏本地文件的下载接口 ----------------
@app.get(f"{API_PREFIX}/pdf/download", tags=["PDF"])
async def pdf_download(fileId: str = Query(...)):
    """供外部服务(如 MinerU)下载原文件"""
    pdf_path = original_pdf_path(fileId)
    if not pdf_path.exists():
        return JSONResponse(err("FILE_NOT_FOUND", "未找到该PDF文件"), status_code=404)
    return FileResponse(str(pdf_path), media_type="application/pdf")

# ---------------- PDF: 触发解析 ----------------
"""
@app.post(f"{API_PREFIX}/pdf/parse", tags=["PDF"])
async def pdf_parse(payload: Dict[str, Any] = Body(...), bg: BackgroundTasks = None):
    file_id = payload.get("fileId")
    if not current_pdf["fileId"] or current_pdf["fileId"] != file_id:
        return JSONResponse(err("FILE_NOT_FOUND", "未找到该文件"), status_code=400)

    current_pdf["status"] = "parsing"
    current_pdf["progress"] = 5

    def _job():
        try:
            # 20 → 60 → 100 三阶段进度示意
            current_pdf["progress"] = 20
            run_full_parse_pipeline(file_id)   # 真解析
            current_pdf["progress"] = 100
            current_pdf["status"] = "ready"
        except Exception as e:
            current_pdf["status"] = "error"
            current_pdf["progress"] = 0
            print("Parse error:", e)

    if bg is not None:
        bg.add_task(_job)
    else:
        _job()

    return {"jobId": rid("j")}
"""

#+ ---------------- PDF: 触发解析v2 ----------------

@app.post(f"{API_PREFIX}/pdf/parse", tags=["PDF"])
async def pdf_parse(payload: Dict[str, Any] = Body(...), bg: BackgroundTasks = None):
    file_id = payload.get("fileId")
    mineru_token = os.getenv("MINERU_TOKEN")

    if not current_pdf["fileId"] or current_pdf["fileId"] != file_id:
        return JSONResponse(err("FILE_NOT_FOUND", "未找到该文件"), status_code=400)
    
    if not mineru_token:
        return JSONResponse(err("MISSING_TOKEN", "缺少 MinerU API Token"), status_code=400)
    
    current_pdf["status"] = "parsing"
    current_pdf["progress"] = 5

    def _job():
        try:
            #+——— 再次增加调用的 parsed 布局图:先将原始 PDF 渲染为图片,供前端展示 original 视图 ———
            current_pdf["progress"] = 10 # 稍微细化一下进度
            render_original_pages(file_id)
            #+—————————————————————————————————————————————————————————————————
            current_pdf["progress"] = 15
            ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
            if not ngrok_url:
                raise ValueError("请设置环境变量 NGROK_URL,例如:https://abc123.ngrok-free.app")
            public_url = f"{ngrok_url}/api/v1/pdf/download?fileId={file_id}"
            print(f"[parse] >>> MinerU 拉取地址: {public_url}", flush=True)
            current_pdf["progress"] = 20
            
            # final_file_url = file_url
            # if not final_file_url:
            #     raise ValueError("由于使用了 MinerU API,必须在 payload 中传入公网可访问的 fileUrl")

            # 核心替换点:调用 MinerU 处理管线替代本地的 unstructured 解析
            process_pdf_with_mineru(
                file_url=public_url, 
                file_id=file_id, 
                token=mineru_token
            )
            
            # 解析成功,更新全局状态
            current_pdf["progress"] = 100
            current_pdf["status"] = "ready"
        except Exception as e:
            # 解析失败,捕获异常并更新错误状态
            current_pdf["status"] = "error"
            current_pdf["progress"] = 0
            print("Parse error:", e)

    # 提交给 FastAPI 的后台任务执行,不阻塞当前 HTTP 响应
    if bg is not None:
        bg.add_task(_job)
    else:
        _job()

    return {"jobId": rid("j")}


# ---------------- PDF: 状态 ----------------
@app.get(f"{API_PREFIX}/pdf/status", tags=["PDF"])
async def pdf_status(fileId: str = Query(...)):
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return {"status": "idle", "progress": 0}
    resp = {"status": current_pdf["status"], "progress": current_pdf["progress"]}
    if current_pdf["status"] == "error":
        resp["errorMsg"] = "解析失败"
    return resp

# ---------------- PDF: 页面图 ----------------
@app.get(f"{API_PREFIX}/pdf/page", tags=["PDF"])
async def pdf_page(
    fileId: str = Query(...),
    page: int = Query(..., ge=1),
    type: str = Query(..., pattern="^(original|parsed)$")
):
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return JSONResponse(status_code=404, content=None)

    if current_pdf["status"] != "ready" and type == "parsed":
        # 未解析就请求 parsed 页,按你的契约可以给 400/403;这里保持 204 更温和
        return JSONResponse(status_code=204, content=None)

    base = dir_original_pages(fileId) if type == "original" else dir_parsed_pages(fileId)
    # 优先查找 JPEG(新格式),兼容旧的 PNG 文件
    img = base / f"page-{page:04d}.jpg"
    media = "image/jpeg"
    if not img.exists():
        img = base / f"page-{page:04d}.png"
        media = "image/png"
    if not img.exists():
        return JSONResponse(err("PAGE_NOT_FOUND", "页面不存在或未渲染"), status_code=404)
    # 页面图不会变化,缓存 1 小时减少重复请求
    return FileResponse(str(img), media_type=media,
                        headers={"Cache-Control": "public, max-age=3600"})

# ---------------- PDF: 图片文件 ----------------
@app.get(f"{API_PREFIX}/pdf/images", tags=["PDF"])
async def pdf_images(
    fileId: str = Query(...),
    imagePath: str = Query(...)
):
    """获取PDF解析后的图片文件"""
    if not current_pdf["fileId"] or current_pdf["fileId"] != fileId:
        return JSONResponse(status_code=404, content=None)

    # 构建图片文件的完整路径
    from services.pdf_service import images_dir
    image_file = images_dir(fileId) / imagePath
    
    if not image_file.exists():
        return JSONResponse(err("IMAGE_NOT_FOUND", "图片文件不存在"), status_code=404)
    
    # 检查文件是否在images目录内(安全考虑)
    try:
        image_file.resolve().relative_to(images_dir(fileId).resolve())
    except ValueError:
        return JSONResponse(err("INVALID_PATH", "无效的图片路径"), status_code=400)
    
    return FileResponse(str(image_file), media_type="image/png")

# ---------------- PDF: 引用片段 ----------------
@app.get(f"{API_PREFIX}/pdf/chunk", tags=["PDF"])
async def pdf_chunk(citationId: str = Query(...)):
    ref = citations.get(citationId)
    if not ref:
        return JSONResponse(err("NOT_FOUND", "无该引用"), status_code=404)
    return ref

class BuildIndexRequest(BaseModel):
    fileId: str

class SearchRequest(BaseModel):
    fileId: str
    query: str
    k: Optional[int] = 5

@app.post(f"{API_PREFIX}/index/build", tags=["Index"])
async def index_build(req: BuildIndexRequest):
    # 可校验:current_pdf["status"] 应为 ready
    if not current_pdf["fileId"] or current_pdf["fileId"] != req.fileId:
        raise HTTPException(status_code=400, detail="FILE_NOT_FOUND_OR_NOT_CURRENT")
    if current_pdf["status"] != "ready":
        raise HTTPException(status_code=409, detail="NEED_PARSE_FIRST")

    out = build_faiss_index(req.fileId)
    if not out.get("ok"):
        return JSONResponse(err(out.get("error", "INDEX_BUILD_ERROR"), "索引构建失败"), status_code=500)
    
    current_pdf["status"] = "ready"      # 标记为完全就绪
    current_pdf["progress"] = 100
    # ✨ 索引构建成功后,才写入 meta.json,文件才会出现在历史库中
    save_metadata(req.fileId, {
        "fileId":   req.fileId,
        "name":     current_pdf.get("name", "Unknown.pdf"),
        "pages":    current_pdf.get("pages", 0),
        "status":   "ready",
        "progress": 100,
    })
    
    return {"ok": True, "chunks": out["chunks"]}

@app.post(f"{API_PREFIX}/index/search", tags=["Index"])
async def index_search(req: SearchRequest):
    out = search_faiss(req.fileId, req.query, req.k or 5)
    if not out.get("ok"):
        code = out.get("error", "INDEX_NOT_FOUND")
        return JSONResponse(err(code, "请先构建索引"), status_code=400)
    return out

# =================================================================
# 历史文件库相关接口 (加载历史列表 & 切换文件)
class SelectFileRequest(BaseModel):
    fileId: str

@app.get(f"{API_PREFIX}/pdf/list", tags=["PDF"])
async def pdf_list():
    """扫描 data 目录,返回所有存有 meta.json 的文件记录"""
    files = []
    data_dir = Path("data")
    
    if not data_dir.exists():
        return {"files": []}
    
    # 遍历 data 目录下的每一个子文件夹
    for folder in data_dir.iterdir():
        if folder.is_dir():
            meta_path = folder / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        
                        # 检查索引文件是否存在 (用于前端显示绿色的勾)
                        # 注意:根据你的 index_service.py,索引文件夹叫 "index_faiss"
                        has_index = (folder / "index_faiss" / "index.faiss").exists()
                        meta["hasIndex"] = has_index
                        
                        files.append(meta)
                except Exception as e:
                    print(f"读取 {folder.name} 的 meta.json 失败: {e}")
                    
    # 按创建时间或降序排列(可选)
    return {"files": files}

@app.post(f"{API_PREFIX}/pdf/select", tags=["PDF"])
async def pdf_select(req: SelectFileRequest):
    """手动切换当前系统关注的 PDF 文件"""
    file_id = req.fileId
    meta_path = Path("data") / file_id / "meta.json"
    
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="本地找不到该文件的记录")
    
    # ✨ 新增:检查 FAISS 索引是否存在
    index_file = Path("data") / file_id / "index_faiss" / "index.faiss"
    if not index_file.exists():
        raise HTTPException(status_code=409, detail="该文件尚未构建索引,无法切换")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        # ✨ 最关键的一步:更新内存中的全局变量 current_pdf
        current_pdf.update({
            "fileId": meta.get("fileId", file_id),
            "name": meta.get("name", "Unknown.pdf"),
            "pages": meta.get("pages", 0),
            "status": "ready",
            "progress": 100
        })
        
        # 切换文件时,清空之前文件的引用缓存,防止 RAG 错乱
        citations.clear()
        clear_history("default")  # ← 新增:清除旧文件的聊天历史
        
        return {"ok": True, "current": current_pdf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换文件失败: {str(e)}")

class DeleteFileRequest(BaseModel):
    fileId: str

@app.post(f"{API_PREFIX}/pdf/delete", tags=["PDF"])
async def pdf_delete(req: DeleteFileRequest):
    """删除历史文件:清除磁盘数据 + 内存状态 + 图谱 Agent 缓存"""
    file_id = req.fileId
    file_dir = Path("data") / file_id

    if not file_dir.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    # 1. 删除整个文件目录(PDF、页面图、索引、meta.json 全部清除)
    shutil.rmtree(file_dir)

    # 2. 如果删的是当前激活文件,重置内存状态
    if current_pdf.get("fileId") == file_id:
        current_pdf.update({
            "fileId": None, "name": None, "pages": 0,
            "status": "idle", "progress": 0
        })
        citations.clear()
        clear_history("default")

    # 3. ✨ 清除图谱 Agent 缓存(避免删了文件后内存里还残留旧 agent)
    try:
        invalidate_agent_cache(file_id)
    except Exception as e:
        print(f"[pdf_delete] 清除图谱 agent 缓存失败(可忽略): {e}")

    return {"ok": True, "deleted": file_id}
# =================================================================