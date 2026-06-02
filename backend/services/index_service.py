# services/index_service.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import json
from services.chunking import split_markdown
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv(override=True)

# 复用你已有的数据目录结构
DATA_ROOT = Path("data")

def workdir(file_id: str) -> Path:
    p = DATA_ROOT / file_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def markdown_path(file_id: str) -> Path:
    return workdir(file_id) / "output.md"  #+如果要做md数据清洗的话可能就需要改改了

def index_dir(file_id: str) -> Path:
    p = workdir(file_id) / "index_faiss"
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_chunks(file_id: str, docs: List[Document]) -> Path:
    """将 split_markdown 的结果存为 chunks.json，格式与调试脚本一致"""
    chunks = [
        {
            "chunk_id": i + 1,
            "summary":  doc.metadata.get("summary", ""),
            "pages":    sorted(doc.metadata.get("pages", [])),
            "length":   len(doc.page_content),
            "content":  doc.page_content,
        }
        for i, doc in enumerate(docs)
    ]

    out_path = workdir(file_id) / "chunks.json"
    out_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return out_path

#使用在线大模型的向量
def load_embeddings() -> OpenAIEmbeddings:
    # 读取环境变量；支持你的代理 base_url
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_EMBEDDING_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(model="text-embedding-3-large", **kwargs)

#全局变量，避免重复加载模型
global_local_embeddings = None
#使用本地模型的向量
def load_local_embeddings() -> HuggingFaceEmbeddings:
    global global_local_embeddings
    if global_local_embeddings is None:
        global_local_embeddings = HuggingFaceEmbeddings(
            model_name="C:\\Users\\Lenovo\\Desktop\\project_2_3.29\\project_2\\qwen3_4b_merged",  # ← 改成你的实际路径
            model_kwargs={"device": "cpu"},          # 没 GPU 改成 "cpu"
            encode_kwargs={"normalize_embeddings": True},
        )
    return global_local_embeddings
    

'''
def split_markdown(md_text: str) -> List[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        # 需要更细可以加 ("###", "Header 3")
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)
    # 可加一点清洗
    cleaned: List[Document] = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        # 限制太长的段落，避免向量化出错
        if len(txt) > 8000:
            txt = txt[:8000]
        cleaned.append(Document(page_content=txt, metadata=d.metadata))
    return cleaned
'''
def build_faiss_index(file_id: str) -> Dict[str, Any]:
    md_file = markdown_path(file_id)
    if not md_file.exists():
        return {"ok": False, "error": "MARKDOWN_NOT_FOUND"}
    md_text = md_file.read_text(encoding="utf-8")

    docs = split_markdown(md_text)
    if not docs:
        return {"ok": False, "error": "EMPTY_MD"}
     # ✨ 新增：保存切片结果
    save_chunks(file_id, docs)
    #在线大模型构建索引
    embeddings = load_embeddings()
    #本地模型构建索引
    # embeddings = load_local_embeddings()
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(str(index_dir(file_id)))
    return {"ok": True, "chunks": len(docs)}

def search_faiss(file_id: str, query: str, k: int = 5) -> Dict[str, Any]:
    idx = index_dir(file_id)
    if not (idx / "index.faiss").exists():
        return {"ok": False, "error": "INDEX_NOT_FOUND"}
    #在线大模型构建的索引使用在线大模型的向量
    embeddings = load_embeddings()
    #本地模型构建的索引使用本地模型的向量
    # embeddings = load_local_embeddings()
    vs = FAISS.load_local(str(idx), embeddings, allow_dangerous_deserialization=True)
    hits = vs.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in hits:
        results.append({
            "text": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata,
        })
    return {"ok": True, "results": results}
