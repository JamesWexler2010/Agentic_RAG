from services.rag_pipeline import retrieve_entities, format_hits_summary

def main():
    file_id = "f_55l2wt09"  # 替换成你的测试文档 ID

    test_queries = [
        "间隙a在该区间内的5 mm<a≤16 mm焊接工艺要求",
    ]

    for q in test_queries:
        print(f"\n{'='*70}\nQuery: {q}\n{'='*70}")
        hits = retrieve_entities(file_id, q)
        print(hits)

if __name__ == "__main__":
    main()