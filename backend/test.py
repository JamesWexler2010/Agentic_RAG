from typing import Dict, List

from neo4j import GraphDatabase
from langchain.chat_models import init_chat_model
from services.agent import agent_query, build_graphrag_agent,truncate
# from neo4j_extract_entities import retrieve_entities
# from services.graph_search_section import write_summary
# from services.summarizer import section_summarizer

neo4j_uri:  str   = "bolt://localhost:7687"
neo4j_auth: tuple = ("neo4j", "20472036")

driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)

llm = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)

# Step 1: 构造 agent(一次性)
agent, tools = build_graphrag_agent(
    file_id="f_55l2wt09",
    driver=driver,
    llm=llm,
    top_k=3,
)

# Step 2: 调用 agent_query(每次问答)
result = agent_query(agent, tools, "第2节船体装配工艺规程的概要")

# Step 3: 使用结果
print("=" * 60)
print(f"Q: {result['question']}")
print("=" * 60)

print(f"\n Answer:")
print(result["answer"])

print(f"\n Evidence (工具调用记录):")
for i, ev in enumerate(result["evidence"], 1):
    if ev["type"] == "call":
        print(f"  [{i}] → 调用 {ev['tool']}({ev['args']})")
    else:
        print(f"  [{i}] ← {ev['tool']} 返回 ({len(ev['content'])} 字符):")
        print(f"        {ev['content']}")

print(f"\n Citations ({len(result['citations'])} 条):")
for c in result["citations"][:3]:
    title = c.get("title") or c.get("section_name", "(无标题)")
    print(f"  • [{c['type']}] {title}")