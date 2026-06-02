"""
summarizers.py

基于 LCEL (LangChain Expression Language) 的摘要生成器工厂。
采用闭包工厂模式，支持外部动态注入 LLM 实例。

包含三个核心组件：
1. table_summarizer: 专门提炼表格核心参数与规范边界。
2. chunk_summarizer: 融合正文与表格摘要，支持动态长度控制。
3. section_summarizer: 自底向上宏观汇总，支持动态长度控制。
"""
from typing import List, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

# =====================================================================
# 1. 表格摘要器 (Table Summarizer)
# =====================================================================
def table_summarizer(llm: BaseChatModel) -> Callable[[str, str], str]:
    """
    生成专门用于提取表格核心数据与结论的 summarizer。
    表格通常要求极度精炼，无需动态计算长度。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是船舶工程领域的技术数据提取专家。"),
        ("user", "请阅读以下表格数据，提炼出核心规范、参数边界或关键指导原则。\n"
                 "【要求】\n"
                 "1. 忽略琐碎的排版或无关紧要的极个别数据，关注整体规律或极值（如最大/最小值、核心材料要求、间隙参数等）。\n"
                 "2. 语言必须极其精炼，直接输出结论，不要加“本表格展示了...”等废话。\n\n"
                 "【表格名称】: {table_name}\n"
                 "【表格行数据】:\n{table_rows}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def summarize(table_name: str, table_rows: str) -> str:
        # 如果表格没内容，直接返回空
        if not table_rows.strip():
            return ""
        return chain.invoke({
            "table_name": table_name, 
            "table_rows": table_rows
        })
    
    return summarize


# =====================================================================
# 2. 文本块摘要器 (Chunk Summarizer - 动态长度版)
# =====================================================================
def chunk_summarizer(llm: BaseChatModel) -> Callable[[str], str]:
    """
    生成用于提炼底层 Chunk (含正文与图表摘要) 的 summarizer。
    支持根据输入原文长度，动态调整大模型输出的摘要目标字数。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是船舶工程领域的技术文档摘要专家，擅长提炼规范、参数、流程要点。"),
        ("user", "请为以下技术文档片段生成简洁、准确的摘要。\n"
                 "【要求】\n"
                 "1. 保留所有关键参数（数值、范围、单位、规范代号）。\n"
                 "2. 逻辑性地融合正文与包含的【表格摘要】内容。\n"
                 "3. 摘要长度请严格控制在大约 **{target_length} 字**左右，直接输出摘要内容。\n\n"
                 "【文档片段内容】:\n{chunk_markdown}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def summarize(chunk_markdown: str) -> str:
        if not chunk_markdown.strip():
            return ""
            
        input_length = len(chunk_markdown)
        
        # 【动态计算逻辑】：按原文长度的 25% 压缩，最少不低于 40 字，最多不超过 300 字
        target_length = max(40, min(300, int(input_length * 0.25)))
        
        return chain.invoke({
            "chunk_markdown": chunk_markdown,
            "target_length": target_length
        })
    
    return summarize


# =====================================================================
# 3. 章节摘要器 (Section Summarizer - 动态长度版)
# =====================================================================
def section_summarizer(llm: BaseChatModel) -> Callable[[List[str], str], str]:
    """
    生成用于自底向上汇总章节 (Section) 的 summarizer。
    支持根据子节点摘要总长度，动态调整大模型输出的宏观摘要目标字数。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是船舶工程领域的技术文档摘要专家，擅长整合多个子项摘要为有逻辑结构的章节总览。"),
        ("user", "你正在为章节《{section_name}》生成总览摘要。下面是该章节下所有子项的摘要列表：\n\n"
                 "【子项摘要】\n{children_text}\n\n"
                 "【要求】\n"
                 "1. 整合所有子项核心信息，形成有逻辑结构的整体概述，体现章节主题。\n"
                 "2. 保留关键技术参数和规范要点，合并同类项，绝对禁止机械拼接！\n"
                 "3. 摘要长度请严格控制在大约 **{target_length} 字**左右，直接输出摘要内容。\n")
    ])
    chain = prompt | llm | StrOutputParser()
    
    def summarize(child_summaries: List[str], section_name: str) -> str:
        if not child_summaries:
            return ""
            
        # 将列表拼接成带序号的文本，方便大模型阅读
        children_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(child_summaries)])
        input_length = len(children_text)
        
        # 【动态计算逻辑】：章节汇总属于二次浓缩，按输入长度的 30% 压缩，最少不低于 80 字，最多不超过 500 字
        target_length = max(80, min(500, int(input_length * 0.3)))
        
        return chain.invoke({
            "section_name": section_name, 
            "children_text": children_text,
            "target_length": target_length
        })
    
    return summarize