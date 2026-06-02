from neo4j import GraphDatabase
from typing import Dict, List, Any

def get_full_tables_from_hits(driver, hits: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    根据向量检索返回的 table_row 命中结果，回溯并获取这些表格的完整内容。
    
    :param driver: Neo4j 驱动实例 (neo4j.GraphDatabase.driver)
    :param hits: 向量检索返回的字典，需包含 'table_row' 键
    :return: 包含完整表格信息的列表
    """
    
    # 1. 提取命中的 row_id 列表
    table_row_hits = hits.get("table_row", [])
    if not table_row_hits:
        return []
        
    row_ids = [row.get("row_id") for row in table_row_hits if row.get("row_id")]
    
    if not row_ids:
        return []

    # 2. 定义 Cypher 语句
    # 说明：使用 DISTINCT t 确保即便命中了同表的多个行，也只返回一次该表
    cypher_query = """
    UNWIND $row_ids AS rid
    MATCH (tr:TableRow {row_id: rid})
    MATCH (t:Table)-[:HAS_ROW]->(tr)
    WITH DISTINCT t
    MATCH (t)-[:HAS_ROW]->(all_rows:TableRow)
    WITH t, all_rows 
    ORDER BY all_rows.row_index ASC
    RETURN 
        t.entity_name AS table_name,
        t.table_img_path AS table_img,
        t.table_body AS html_content,
        t.table_path AS table_path,
        collect({
            index: all_rows.row_index, 
            text: all_rows.row_text,
            row_id: all_rows.row_id
        }) AS full_rows
    """

    # 3. 执行查询
    final_results = []
    with driver.session() as session:
        result = session.run(cypher_query, row_ids=row_ids)
        
        for record in result:
            final_results.append({
                "table_name": record["table_name"],
                "table_path": record["table_path"],
                "table_img": record["table_img"],
                "html_content": record["html_content"],
                "full_rows": record["full_rows"],
                # 方便后续拼成文本交给 LLM
                "full_text_content": "\n".join([r["text"] for r in record["full_rows"]])
            })
            
    return final_results

def get_table_context_with_chunks(driver, hits: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    table_hits = hits.get("table", [])
    if not table_hits:
        return []
        
    table_names = [t.get("entity_name") for t in table_hits if t.get("entity_name")]
    if not table_names:
        return []

    # 修复后的 Cypher：通过两步 WITH 分离聚合操作
    cypher_query = """
    UNWIND $table_names AS t_name
    MATCH (targetTable:Table {entity_name: t_name})
    MATCH (c:Chunk)-[:CONTAINS_TABLE]->(targetTable)
    
    // 第一步：先找出该 Chunk 下所有的表及其行
    WITH DISTINCT c
    MATCH (c)-[:CONTAINS_TABLE]->(allTables:Table)
    MATCH (allTables)-[:HAS_ROW]->(rows:TableRow)
    
    // 第二步：【关键修复】先按表格分组，聚合行 (rows)
    // 这里的 WITH c, allTables 相当于 SQL 的 GROUP BY
    WITH c, allTables, rows ORDER BY rows.row_index ASC
    WITH c, allTables, collect({idx: rows.row_index, txt: rows.row_text}) AS table_rows
    
    // 第三步：再按 Chunk 分组，聚合表格 (tables)
    // 此时 table_rows 已经是一个列表，不再是聚合函数，可以被安全地放入 collect
    WITH c, collect({
        table_name: allTables.entity_name,
        table_html: allTables.table_body,
        rows: table_rows
    }) AS tables_data
    
    RETURN 
        c.entity_name AS chunk_title,
        c.content AS chunk_text,
        c.section_path AS hierarchy,
        tables_data
    """

    context_results = []
    with driver.session() as session:
        result = session.run(cypher_query, table_names=table_names)
        
        for record in result:
            # 编排逻辑
            formatted_context = f"**章节层级**: {record['hierarchy']}\n"
            formatted_context += f"**文本说明**: \n{record['chunk_text']}\n\n"
            
            for tbl in record['tables_data']:
                formatted_context += f"**相关表格**: {tbl['table_name']}\n"
                # 注意：这里的 tbl['rows'] 已经是排好序的列表了
                row_texts = [r['txt'] for r in tbl['rows']]
                formatted_context += "\n".join(row_texts) + "\n\n"
            
            context_results.append({
                "chunk_title": record["chunk_title"],
                "full_formatted_text": formatted_context,
                "raw_data": record
            })
            
    return context_results