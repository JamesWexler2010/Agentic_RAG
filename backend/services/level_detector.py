"""
level_detector.py

总结类问题的目标层级判定器(v3, 与建图阶段的 6 级标题正则对齐)。

层级体系(level ↔ 图谱 depth):
  level=1  章          depth=0  例:"第1章 船体专业"
  level=2  节/附录     depth=1  例:"第4节 分段肋板拉入法"、"附录A"
  level=3  节内一级    depth=2  例:"5 作业程序"  (无正则,只靠 LLM)
  level=4  x.x         depth=3  例:"3.12 平面度要求"
  level=5  x.x.x       depth=4  例:"5.7.1"
  level=6  x.x.x.x     depth=5  例:"5.7.1.1"

设计原则:
  1. 用户明确提到层级标号 → 正则命中候选 → LLM 复核确认 → 返回最终 level
  2. 用户未提到任何层级标号 → 返回 None,告知调用方"按命中节点本身处理"
                              (由 graph_search_section.py 走"不 lift"分支)

LLM 复核(对应 Q2 选 a):
  任何正则命中后都让 LLM 复核——避免:
    - "5.7 章节" 同时命中 L4 和 L2,需要 LLM 判断主问的是哪一层
    - "章节" 被字面命中但用户其实没在问层级
    - LLM 还可以主动选 level=3(节内一级),即使正则不直接产生 3 候选
"""
from __future__ import annotations
import re
from typing import Optional, Any, List, Tuple


# ============================================================
# 关键词 / 正则模式
# ============================================================

# --- Level 1 (章) ---
_L1_KWS: tuple[str, ...] = ('某章', '该章', '一章', '本章', '这一章', '哪一章')
_L1_PATTERNS: tuple[re.Pattern, ...] = (
    # 第N章(N 可以是阿拉伯数字或汉字数字)
    re.compile(r'第\s*[0-9一二三四五六七八九十百零]+\s*章'),
)

# --- Level 2 (节 / 附录) ---
# 注:'章节' 是泛指词,放在 L2 一档(更常见的语义是"某一节")
_L2_KWS: tuple[str, ...] = ('某节', '该节', '一节', '本节', '这一节', '哪一节', '章节')
_L2_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r'第\s*[0-9一二三四五六七八九十百零]+\s*节'),
    re.compile(r'附\s*录\s*[A-Za-z0-9]'),
)

# --- Level 4 (x.x) ---
# 用户问题里出现 "3.12" 这类数字,但要确保不是 x.x.x 或更深
# Python re 中负向先行断言: (?!\.\d) 表示后面不能再跟一个 ".数字"
_L4_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r'\d+\.\d+(?!\.\d)'),
)

# --- Level 5 (x.x.x) ---
_L5_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r'\d+\.\d+\.\d+(?!\.\d)'),
)

# --- Level 6 (x.x.x.x) ---
_L6_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r'\d+\.\d+\.\d+\.\d+'),
)

# 注:Level 3 (节内一级,如"5 作业程序") 没有可靠的正则——
# 单独的 \d+ 会误命中"工序 5"、"5 个"等。
# 因此 Level 3 仅在 LLM 复核阶段产生(LLM 看完整问题语境后决定)。


# ============================================================
# LLM 复核 prompt
# ============================================================

_LLM_CONFIRM_PROMPT = """用户问题中包含了一些层级相关的关键词或数字标号,
正则规则给出了候选的目标查询层级。
请结合问题的真实意图,判断最合理的查询层级(1-6)。

层级语义:
  1 = 整章        (例:"第1章 船体专业")
  2 = 整节 / 附录 (例:"第4节 ...", "附录A")
  3 = 节内一级    (例:"5 作业程序"、"3 工序建造精度要求"——
                  通常在"节"下面、由单个数字+标题组成的一级条目)
  4 = x.x         (例:"3.12 平面度要求"、"5.2 纵骨")
  5 = x.x.x       (例:"5.7.1")
  6 = x.x.x.x     (例:"5.7.1.1")

判断准则:
  - 优先尊重用户的明确标号(如 "5.7" 一般就是 level=5 那行所在层级,即 4)
  - 但要看上下文:如果用户说"5.7 所在的章节"、"5.7 比之前的章节",意图是更上层
  - 若问题里只提了泛指词(如"章节"),倾向 level=2
  - 若问题明显是问某个由单数字开头的条目(如"5 作业程序讲了什么"),取 level=3

用户问题: {question}

正则规则推测的候选层级: {candidates}
(说明:候选可能是多个,因为问题里命中了不同的关键词)

请仅输出一个数字(1、2、3、4、5 或 6),不要任何其他内容:"""


# ============================================================
# 主入口
# ============================================================

def decide_level(
    question: str,
    llm: Optional[Any] = None,
    verbose: bool = True,
) -> Optional[int]:
    """
    判定查询的目标层级。

    Returns:
        - int(1-6):用户明确提到层级,且通过 LLM 复核
        - None:    用户未提到任何层级标号
                   调用方收到 None 后,应让 graph_search_section.py 走
                   "命中节点本身就是 target,不 lift"的分支

    Args:
        question: 用户问题
        llm:      用于复核的 LLM 实例。**强烈建议传入**——纯正则无法处理:
                  - 多候选场景(如"5.7 章节"同时命中 L4 和 L2)
                  - 仅靠 LLM 才能产生的 level=3(节内一级)
        verbose:  打印调试日志
    """
    q = (question or "").strip()
    if not q:
        if verbose:
            print("[decide_level] 空问题 → 返回 None(用命中节点本身)")
        return None

    # ---- 1) 收集所有正则/关键词命中 ----
    # 注:不再"短路返回",而是收集所有候选,交给 LLM 在多候选时辨别
    candidates: List[Tuple[int, str]] = []  # [(level, evidence), ...]

    # L1
    for pat in _L1_PATTERNS:
        m = pat.search(q)
        if m:
            candidates.append((1, f"章级模式 '{m.group()}'"))
    for kw in _L1_KWS:
        if kw in q:
            candidates.append((1, f"章级关键词 '{kw}'"))

    # L2
    for pat in _L2_PATTERNS:
        m = pat.search(q)
        if m:
            candidates.append((2, f"节级/附录模式 '{m.group()}'"))
    for kw in _L2_KWS:
        if kw in q:
            candidates.append((2, f"节级关键词 '{kw}'"))

    # Level 3 没有正则——仅由 LLM 在复核阶段产生

    # L6 必须先于 L5、L4 检查(否则 1.2.3.4 会被 L4/L5 抢匹配)
    for pat in _L6_PATTERNS:
        m = pat.search(q)
        if m:
            candidates.append((6, f"四级标号 '{m.group()}'"))

    # L5 先于 L4
    if not any(lv == 6 for lv, _ in candidates):
        for pat in _L5_PATTERNS:
            m = pat.search(q)
            if m:
                candidates.append((5, f"三级标号 '{m.group()}'"))

    # L4
    if not any(lv in (5, 6) for lv, _ in candidates):
        for pat in _L4_PATTERNS:
            m = pat.search(q)
            if m:
                candidates.append((4, f"二级标号 '{m.group()}'"))

    # ---- 2) 没有任何候选 → 返回 None ----
    if not candidates:
        if verbose:
            print("[decide_level] 未命中任何层级关键词/标号 → 返回 None(用命中节点本身)")
        return None

    # ---- 3) 有候选 → LLM 复核(Q2: 选项 a, 一律走复核) ----
    candidate_levels = sorted({lv for lv, _ in candidates})

    if verbose:
        evidence_str = ", ".join(f"level={lv}({ev})" for lv, ev in candidates)
        print(f"[decide_level] 正则候选: {evidence_str}")

    if llm is None:
        # 没有 llm 实例时降级:候选唯一 → 直接返回;多候选 → 取最深(更精准)
        chosen = max(candidate_levels)
        if verbose:
            print(f"[decide_level] 无 llm 实例,降级规则选 level={chosen}")
        return chosen

    # LLM 复核
    try:
        prompt = _LLM_CONFIRM_PROMPT.format(
            question=q,
            candidates=candidate_levels,
        )
        resp = llm.invoke([{"role": "user", "content": prompt}])
        text = (getattr(resp, "content", "") or "").strip()

        # 解析 LLM 输出(1-6)
        for level_str in ("1", "2", "3", "4", "5", "6"):
            if text.startswith(level_str):
                final = int(level_str)
                if verbose:
                    print(
                        f"[decide_level] LLM 复核 → level={final} "
                        f"(候选={candidate_levels}, 原始输出={text!r})"
                    )
                return final

        # LLM 输出无法解析 → 降级取候选里最深层
        chosen = max(candidate_levels)
        if verbose:
            print(
                f"[decide_level] LLM 输出无法解析 {text!r},"
                f"降级取候选最深 level={chosen}"
            )
        return chosen

    except Exception as e:
        chosen = max(candidate_levels)
        if verbose:
            print(f"[decide_level] LLM 复核异常: {e},降级取候选最深 level={chosen}")
        return chosen