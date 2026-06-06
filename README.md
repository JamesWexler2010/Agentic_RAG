# 基于多智能体协同的船舶建造知识问答系统

## 功能

### 1. PDF 文档处理
用户上传 PDF，系统接入 MinerU 完成 OCR、版面解析、Markdown 转换，将多模态文档内容转化为高质量的结构化数据；提供文档解析状态查询，并在右侧 Panel 预览文档版面解析结果。

### 2.1 索引构建（普通向量检索）
解析得到的 Markdown 文档进行切分（chunking）：
- 基于 Markdown 的文档结构分块，是平台默认的分块策略：
  - 首先设定文本块的最小、最大分割长度；
  - 然后自动识别章节（`#`、`##`、`###`），对已识别章节的字数进行计数，在恰好位于 > 最小分割长度 同时 < 最大分割长度的前提下进行分段；
  - 当遇到超长段落（超出最大分割长度）时，执行递归分段算法，确保语义完整性；
  - 对于 HTML 格式的长表格不切分，不破坏结构。

使用 Embedding 模型（OpenAI Embeddings）将片段向量化，保存至 FAISS 向量数据库，用于后续检索。

**分支：自训练 Embedding 模型**
可选择切换至自训练的 Embedding 模型，辅助对 MinerU 输出的 HTML 格式表格内容的检索。使用 EasyDataset 构建训练集，生成的问答对中的回答为 RAG 所需的原文档片段，模型选用 qwen3_Embedding_4B，可达到性能与显存开销的平衡。

### 2.2 索引构建（"向量 + 图谱"检索）
直接使用 LangChain 框架的 `MarkdownHeaderTextSplitter` 对 Markdown 文档进行切分，利用文章的章节层级关系进行知识图谱三元组构建，实体类型分别为：章节标题、文档文本内容、图表名称、表格逐行内容。

使用 Embedding 模型（OpenAI Embeddings）将构建的实体内容向量化，保存至 FAISS 向量数据库，用于后续图谱检索实体命中。

**检索增强：** 使用 LLM 对 HTML 表格内容进行逐行自然语义处理，保留完整无改动的表格行内容，构建表格行实体挂载在表格下，提升表格内容检索的召回率和准确率。

### 4. 知识问答
- 答案中附带引用（Citations），方便追溯来源；
- 对于图表，直接给出截图，不对其进行语义理解；
- **普通向量检索路径：** 回答附带文本块所在的原文档页面链接；
- 支持流式输出（SSE）；
- 基于 InMemorySaver 的历史对话记忆，页面重新加载时清空；
- **"向量 + 图谱"检索路径：** 在右侧 Panel 展示此轮问答中调用的实体节点图。

---

## 技术栈

### 后端

| 组件 | 版本 | 用途 |
|---|---|---|
| FastAPI | 0.135.1 | Web 框架 + Swagger UI |
| Uvicorn | 0.42.0 | ASGI 服务器 |
| pydantic | 2.12.5 | 请求/响应体校验 |
| python-multipart | 0.0.22 | 文件上传 |
| python-dotenv | 1.2.2 | 环境变量 |

### PDF 解析

| 组件 | 版本 | 用途 |
|---|---|---|
| MinerU | — | OCR、版面解析、导出 Markdown |
| PyMuPDF (fitz) | 1.27.2 | 页面渲染为 PNG（版面预览） |
| Pillow | 12.1.1 | 图片后处理 |

### 索引与检索

| 组件 | 版本 | 用途 |
|---|---|---|
| faiss-cpu | 1.13.2 | 向量存储与 Top-K 检索 |
| langchain-text-splitters (`MarkdownHeaderTextSplitter`) | 1.1.1 | Markdown 结构化切分 |
| langchain-openai | 1.1.11 | OpenAI Embeddings 向量化 |
| Neo4j | 6.1.0 | 知识图谱存储与检索 |

### 对话

| 组件 | 版本 | 用途 |
|---|---|---|
| LangGraph (`InMemorySaver`) | 1.1.3 | 多轮会话状态 + 对话记忆 |
| LangChain | 1.2.12 | RAG 链路编排 |
| openai | 2.29.0 | LLM 调用 |

### 前端

| 技术 | 用途 |
|---|---|
| React / Next.js | SSE 、页面渲染 |
| Figma | UI 原型 |

### 环境

```
Python >= 3.9
Node.js >= 18
```
