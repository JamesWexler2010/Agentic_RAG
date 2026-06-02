import re
import logging
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_PAGE_RE   = re.compile(r'<!--\s*page:\s*(\d+)\s*-->')
_ATOMIC_RE = re.compile(r'<table[\s>]|!\[', re.IGNORECASE)


# ── 1. 逐行状态机切段 ─────────────────────────────────────────
def _split_sections(text: str, split_levels=(1,2,3,4,5,6)) -> List[Dict]:
    """
    对齐 LangChain MarkdownHeaderTextSplitter 核心逻辑：
    - 代码块内的 # 不识别为标题
    - split_levels 控制哪些层级触发切分，其余层级的标题归并到父级正文
      推荐 (1,2,3)：消除深层空标题产生的大量空 section
    每个 section：
      headers : {level: title}  完整祖先链，直接用于摘要
      content : 已清洗正文（无 page marker，连续空行已压缩）
      markers : 本 section 内的页码列表
    """
    HEAD = re.compile(r'^(#{1,6})\s+(.*)')
    cur_headers: Dict[int, str] = {}
    cur_lines:   List[str]      = []
    in_code  = False
    sections: List[Dict] = []

    def flush():
        raw     = '\n'.join(cur_lines)
        markers = [int(p) for p in _PAGE_RE.findall(raw)]
        content = re.sub(r'\n{3,}', '\n\n', _PAGE_RE.sub('', raw)).strip()
        if content:
            sections.append({'headers': dict(cur_headers),
                             'content': content, 'markers': markers})

    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('```'):
            in_code = not in_code
        if in_code:
            cur_lines.append(line)
            continue
        m = HEAD.match(s)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if level in split_levels:
                flush()
                cur_lines = [s]                          # 标题行作为 section 首行
                cur_headers[level] = title
                for k in list(cur_headers):              # 清除子层级
                    if k > level: del cur_headers[k]
                continue
        cur_lines.append(line)

    flush()
    return sections


# ── 2. 路径摘要 ───────────────────────────────────────────────
def _make_summary(section: Dict) -> str:
    h = section['headers']
    return ' > '.join(h[k] for k in sorted(h)) if h else '文档前言'


# ── 3. 超长拆分（section 级 table/image 不可切分）────────────
def _split_oversized(content: str, summary: str,
                     pages: list, max_len: int,
                     min_len: int = 0) -> List[Document]:
    """
    降级策略（仅在 section 本身就超过 max_len 时触发）：
      table/image 段落 → 整段保留（即使超长）
      普通段落         → 按段落 → 句子 → 固定长度 三级切分

    emit 规则：
      force=False：len >= min_len 才输出，否则 cur 保持不变继续积累
      force=True ：忽略 min_len 强制输出（atomic 块 / 无法继续积累时）
    返回值 bool 表示是否实际输出，cur 由 emit 内部管理，调用方不再手动重置。
    """
    pages  = sorted(pages)
    result: List[Document] = []
    cur    = ''
    part   = 1

    def emit(t: str, force: bool = False) -> bool:
        nonlocal cur, part
        t = t.strip()
        if not t:
            return False
        if force or len(t) >= min_len:
            result.append(Document(
                page_content=t,
                metadata={'summary': f"{summary} - Part {part}", 'pages': pages}
            ))
            part += 1
            cur = ''
            return True
        return False   # 不足 min_len，cur 保持不变，由调用方继续积累

    for para in content.split('\n\n'):
        if not para.strip():
            continue

        if _ATOMIC_RE.search(para):
            # ATOMIC BLOCK：table / image 段落不可拆分，必须整段保留。
            # 超过 max_len 时 emit 后 chunk 超长，属已知预期行为。
            if cur:
                if len(cur) >= min_len:
                    emit(cur, force=True)
                    if len(para) >= min_len:
                        emit(para, force=True)   # atomic 够大，直接输出
                    else:
                        cur = para               # atomic 太小，存入 cur 等后续补足 
                else:
                    merged = cur + '\n\n' + para
                    if len(merged) < min_len:
                        cur = merged
                    else:
                        emit(merged, force=True)    
            elif len(para) >= min_len:
                emit(para, force=True)         # atomic 本身达标，直接输出
            else:
                cur = para                     # 不足 min_len，存入 cur 等待后续段落补足
            continue

        if len(para) > max_len:              # 段落超长：按句子切
            for sent in re.split(r'(?<=[。！？.!?])', para):
                if not sent.strip():
                    continue
                if len(sent) > max_len:      # 单句兜底：固定长度硬截
                    for i in range(0, len(sent), max_len):
                        cur = cur + sent[i:i+max_len] if cur else sent[i:i+max_len]
                        if len(cur) >= min_len:
                            emit(cur)
                elif len(cur) + len(sent) > max_len:
                    emit(cur, force=True)    # 无法继续积累，强制输出
                    cur = sent
                    if len(cur) >= min_len:
                        emit(cur)
                else:
                    cur += sent
                    if len(cur) >= min_len:
                        emit(cur)
            continue

        # 普通段落：贪婪积累
        if len(cur) + len(para) + 2 > max_len:
            emit(cur, force=True)            # 无法继续积累，强制输出
            cur = para
        else:
            cur = cur + '\n\n' + para if cur else para
        if len(cur) >= min_len:
            emit(cur)

    if cur:
        emit(cur, force=True)                # 末尾残留：强制输出
    return result


# ── 4. 主入口：贪婪滑动窗口合并 ─────────────────────────────
def split_markdown(text: str,
                   min_len: int = 200,
                   max_len: int = 3000,
                   split_levels: tuple = (1,2,3,4,5,6)) -> List[Document]:
    """
    流程：
      状态机切段 → 贪婪窗口合并到 min_len 后输出

    窗口规则：
      - 每个 section 无条件加入窗口（保证 section 完整性）
      - 加入后窗口 >= min_len → 立即输出，开新窗口
      - 加入前窗口已 >= min_len 且加入后会超 max_len → 先输出再加入
        （避免能分开时不必要地触发超长拆分）
      - 末尾残留 < min_len → 并入前一个 chunk（消除文档末尾孤立小块）
      - 单个 section 本身 > max_len → _split_oversized 拆分
    """
    if not text.strip():
        return []

    docs: List[Document] = []
    cur_page = 1
    win_content = ''
    win_first:  Dict = {}
    win_count   = 0
    win_pages:  set  = set()

    def emit_window(content: str, pages: set) -> None:
        base    = _make_summary(win_first)
        summary = base if win_count == 1 else f"{base}（及后续{win_count-1}节）"
        if len(content) > max_len:
            docs.extend(_split_oversized(content, summary, sorted(pages), max_len, min_len))
        else:
            docs.append(Document(page_content=content,
                                 metadata={'summary': summary, 'pages': sorted(pages)}))

    def reset_window() -> None:
        nonlocal win_content, win_first, win_count, win_pages
        win_content, win_first, win_count, win_pages = '', {}, 0, set()

    for sec in _split_sections(text, split_levels):
        content = sec['content']
        # 修复：只用本 section 实际出现的页码，不带入跨窗口的 cur_page 游标
        if sec['markers']:
            sec_pages = set(sec['markers'])
            cur_page  = sec['markers'][-1]
        else:
            sec_pages = {cur_page}

        # 窗口已够大 且 加入后超长 → 先输出当前窗口
        if win_content and len(win_content) >= min_len \
                and len(win_content) + 2 + len(content) > max_len:
            emit_window(win_content, win_pages)
            reset_window()

        # 加入窗口
        win_content = win_content + '\n\n' + content if win_content else content
        win_pages  |= sec_pages
        win_count  += 1
        if win_count == 1:
            win_first = sec

        # 窗口够大 → 输出
        if len(win_content) >= min_len:
            emit_window(win_content, win_pages)
            reset_window()

    # 末尾残留
    if win_content:
        if docs and len(win_content) < min_len:
            prev         = docs.pop()
            merged       = prev.page_content + '\n\n' + win_content
            merged_pages = sorted(set(prev.metadata['pages']) | win_pages)
            if len(merged) > max_len:
                base = prev.metadata['summary'].split('（及后续')[0]
                docs.extend(_split_oversized(merged, base, merged_pages, max_len, min_len))
            else:
                docs.append(Document(page_content=merged,
                                     metadata={'summary': prev.metadata['summary'],
                                               'pages': merged_pages}))
        else:
            emit_window(win_content, win_pages)

    # ── Warning 检查 ─────────────────────────────────────────────
    # chunk 超过 max_len 的唯一合理原因：该 chunk 内含有 table 或 image 的 HTML 段落，
    # _split_oversized 已保证不在 atomic 块内部截断（会破坏 HTML 结构），
    # 因此这类超长是预期行为，无法通过调整 max_len 消除。
    for i, doc in enumerate(docs):
        if len(doc.page_content) > max_len:
            logger.warning(
                f"[chunk {i+1}] 长度 {len(doc.page_content)} 超过 max_len {max_len}，"
                f"含不可切分的 table/image 块，属预期行为：{doc.metadata['summary']!r}"
            )

    return docs


# ── 调试入口 ──────────────────────────────────────────────────
if __name__ == '__main__':
    import json

    with open('backend/data/f_55l2wt09/output.md', encoding='utf-8') as f:
        docs = split_markdown(f.read(), min_len=200, max_len=3000)

    result = [
        {
            'chunk_id': i + 1,
            'summary':  doc.metadata['summary'],
            'pages':    sorted(list(doc.metadata['pages'])),
            'length':   len(doc.page_content),
            'content':  doc.page_content,
        }
        for i, doc in enumerate(docs)
    ]

    with open('backend/data/f_55l2wt09/chunks_4.11_2.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"共 {len(docs)} 个 chunk，已写入 chunks_4.11_2.json")