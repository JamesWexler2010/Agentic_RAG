# services/rag_service.py
from __future__ import annotations
import os, asyncio, textwrap
from typing import List, Dict, Any, Tuple, AsyncGenerator
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import defaultdict
#+—— 本地模型直接复用 index_service 的单例,不再重复加载 ———
from services.index_service import load_local_embeddings

# ✨ 改动:图片 URL 工具抽到公共模块 image_utils.py,与 graph_service 共享
from services.image_utils import rewrite_image_urls, extract_image_urls

# 存储结构:sessions[session_id] = [{"role":"user|assistant","content":"..."}...]
_sessions: dict[str, list[dict]] = defaultdict(list)

def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])

def append_history(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})

def clear_history(session_id: str) -> None:
    _sessions.pop(session_id, None)

# ---------------- 配置 ----------------
# MODEL_NAME = "deepseek-chat"
# MODEL_PROVIDER = "deepseek"
# TEMPERATURE = 0
MODEL_NAME = "gpt-4o"
MODEL_PROVIDER = "openai"
TEMPERATURE = 0

EMBED_MODEL = "text-embedding-3-large"
K = 3
# FAISS L2:越小越相似;数值可以灵活调整
SCORE_TAU_TOP1 = 0.45
SCORE_TAU_MEAN3 = 0.60

SYSTEM_INSTRUCTION = (
    "你是模态 PDF 检索 RAG 机器人助手,可以围绕多模态文档进行解析、检索和问答。\n"
    "请优先使用当前上传并已解析/索引的文档来回答问题;若未检索到相关内容,则基于通识知识作答,"
    "并**明确说明未找到匹配的文档片段**。\n"
    "当检索到的上下文中包含与答案直接相关的图片时,请在回答中一并给出这些图片的 Markdown 引用,"
    "例如:`![参考图1](图片URL)`。如果没有合适的图片,也就是如果没有检索到图片,绝不伪造图片或路径。"
)

GRADE_PROMPT = (
    "你是一个判定器,评估检索到的上下文是否有助于回答用户问题。\n"
    "上下文片段:\n{context}\n\n问题:{question}\n"
    "如果上下文对回答该问题有帮助,返回 'yes';否则返回 'no'。"
)

ANSWER_WITH_CONTEXT = (
    "请使用提供的上下文回答用户的问题。\n\n"
    "问题:\n{question}\n\n上下文:\n{context}\n\n"
    "要求:使用 Markdown;表达简洁但完整;如需给出代码,请使用三引号代码块(```)。\n"
    "若上下文包含与答案直接相关的图片,请在相关段落后内联给出图片(Markdown 语法)。\n"
)

ANSWER_NO_CONTEXT = (
    "当前未找到与文档直接相关的片段,将基于通识知识作答。\n"
    "问题:\n{question}"
)


# ---------------- 模型/向量函数 ----------------
def _get_llm():
    return init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=TEMPERATURE)

def _get_grader():
    return init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=0)

#获取在线大模型的向量
def _get_embeddings():
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL") or "https://ai.devtool.tech/proxy/v1",
        model=EMBED_MODEL,
    )

#+—— 获取本地模型的向量,且使用index_service过程加载的模型,避免重复加载模型耗时 ———
def _get_local_embeddings():
    return load_local_embeddings()

def _vs_dir(file_id: str) -> str:
    return os.path.join("data", file_id, "index_faiss")

def _load_vs(file_id: str) -> FAISS:
    vs_path = _vs_dir(file_id)
    idx_file = os.path.join(vs_path, "index.faiss")
    if not os.path.exists(idx_file):
        raise FileNotFoundError(f"FAISS index not found at {vs_path}; build index first.")
    #在线大模型构建的索引使用在线大模型的向量
    return FAISS.load_local(vs_path, _get_embeddings(), allow_dangerous_deserialization=True)
    #本地模型构建的索引使用本地模型的向量
    # return FAISS.load_local(vs_path, _get_local_embeddings(), allow_dangerous_deserialization=True)

def _score_ok(scores: List[float]) -> bool:
    if not scores:
        return False
    top1 = scores[0]
    mean3 = sum(scores[:3]) / min(3, len(scores))
    return (top1 <= SCORE_TAU_TOP1) or (mean3 <= SCORE_TAU_MEAN3)


# ---------------- 主流程:检索 + 判定 + 生成 ----------------
async def retrieve(question: str, file_id: str) -> tuple[list[dict], str]:
    """
    返回 (citations, context_text)
    citations: [{citation_id, fileId, rank, page, pages, snippet, score, previewUrl, images}]
    context_text: 供 LLM 使用的拼接上下文
    """
    vs = _load_vs(file_id)
    hits = vs.similarity_search_with_score(question, k=K)
    citations = []
    ctx_snippets = []
    scores = []
    for i, (doc, score) in enumerate(hits, start=1):
        snippet_short = (doc.page_content or "").strip()
        # ✨ 从原始 chunk 中提取图片列表(在重写之前,用原始相对路径匹配)
        chunk_images = extract_image_urls(snippet_short, file_id)
        # ✨ 将 chunk 内的相对图片路径重写为前端可访问的 API URL
        snippet_short = rewrite_image_urls(snippet_short, file_id)

        # ✨ 修改:明确兜底为第1页,比 or [None] 更清晰
        pages_list = doc.metadata.get("pages") or []
        # 兜底:确保所有页码 >= 1,防止旧索引中残留的 0-indexed 值
        pages_list = [max(p, 1) for p in pages_list]
        page = pages_list[0] if pages_list else 1

        citations.append({
            "citation_id": f"{file_id}-c{i}",
            "fileId": file_id,
            "rank": i,
            "page":  page,
            "pages": pages_list,
            "snippet": snippet_short,
            "score": float(score),
            "previewUrl": f"/api/v1/pdf/page?fileId={file_id}&page={page}&type=original",
            "images": chunk_images,   # ✨ 新增:该 chunk 内的图片列表,前端可直接渲染
        })
        ctx_snippets.append(f"[{i}] {snippet_short}")
        scores.append(float(score))
    context_text = "\n\n".join(ctx_snippets) if ctx_snippets else "(no hits)"

    # 规则 + LLM 复核
    ok_by_score = _score_ok(scores)
    if not ok_by_score:
        grader = _get_grader()
        grade_prompt = GRADE_PROMPT.format(context=context_text, question=question)
        decision = await grader.ainvoke([{"role": "user", "content": grade_prompt}])
        ok_by_llm = "yes" in (decision.content or "").lower()
    else:
        ok_by_llm = True

    branch = "with_context" if ok_by_llm else "no_context"
    return citations, context_text if branch == "with_context" else ""


async def answer_stream(
    question: str,
    citations: list[dict],
    context_text: str,
    branch: str,
    session_id: str | None = None
) -> AsyncGenerator[dict, None]:
    """
    以增量事件的形式产出:
      {"type":"citation", "data": {...}}
      {"type":"token", "data": "text chunk"}
      {"type":"done", "data": {"used_retrieval": bool}}
    同时:如果提供了 session_id,会把本轮问答写入内存历史。
    """
    # 先把 citations 全部发给前端(便于角标立刻出现)
    if branch == "with_context" and citations:
        for c in citations:
            yield {"type": "citation", "data": c}

    # 组装"历史 + 本轮提示"
    llm = _get_llm()
    history_msgs = get_history(session_id) if session_id else []

    if branch == "with_context" and context_text:
        user_prompt = ANSWER_WITH_CONTEXT.format(question=question, context=context_text)
    else:
        user_prompt = ANSWER_NO_CONTEXT.format(question=question)

    # 完整消息序列:system + 历史多轮 + 当前用户
    msgs = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    # 将历史逐条附加(保持 role: "user"/"assistant")
    msgs.extend(history_msgs)
    # 当前用户问题
    msgs.append({"role": "user", "content": user_prompt})

    # 把最终生成的文本拼接出来用于写历史
    final_text_parts: list[str] = []

    # 优先使用流式
    try:
        async for chunk in llm.astream(msgs):
            delta = getattr(chunk, "content", None)
            if delta:
                final_text_parts.append(delta)
                yield {"type": "token", "data": delta}
    except Exception as stream_err:
        # 回退:非流式整段生成
        try:
            resp = await llm.ainvoke(msgs)
            text = resp.content or ""
            final_text_parts.append(text)
            for i in range(0, len(text), 20):
                yield {"type": "token", "data": text[i:i+20]}
                await asyncio.sleep(0.005)
        except Exception as invoke_err:
            # 两种方式都失败,将错误传递给前端而不是静默吞掉
            import traceback
            traceback.print_exc()
            yield {"type": "error", "data": {"message": f"LLM 调用失败: {invoke_err}"}}
            return

    # +—— 前端已经有该逻辑了,这里不需要后端再发一次了,避免重复展示图片 ——
    # if branch == "with_context" and citations:
    #     imgs = []
    #     # 取前 2 张,避免过多(可按需改成 3)
    #     for c in citations[:2]:
    #         url = c.get("previewUrl")
    #         if url:
    #             # 生成 Markdown 图片行
    #             imgs.append(f"![参考页 {c.get('rank', '')}]({url})")
    #     if imgs:
    #         tail = "\n\n---\n**相关页面预览**\n\n" + "\n\n".join(imgs)
    #         # 作为一个额外 token 块发给前端
    #         yield {"type": "token", "data": tail}

    # 将本轮问答写入历史(仅在提供 session_id 时)
    if session_id:
        append_history(session_id, "user", question)
        append_history(session_id, "assistant", "".join(final_text_parts))

    yield {"type": "done", "data": {"used_retrieval": branch == "with_context"}}