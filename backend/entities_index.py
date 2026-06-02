import asyncio
from services.table_index_serivce_split import (
    build_media_index_stage1,
    build_media_index_stage2,
    build_media_index_stage3,
    build_media_index_stage4,
    build_text_entities_stage1,
    build_text_entities_stage2,
)

FILE_ID = "f_55l2wt09"

async def main():
    # ── Media Stage 1 ──
    result = await build_media_index_stage1(FILE_ID)
    print(result)
    if not result["ok"]:
        return
    input("检查 table_entities.json，确认无误后按 Enter 继续...")

    # ── Media Stage 2 ──
    result = await build_media_index_stage2(FILE_ID)
    print(result)
    if not result["ok"]:
        return
    input("检查 image_entities.json，确认无误后按 Enter 继续...")

    # ── Media Stage 3 ──
    result = await build_media_index_stage3(FILE_ID, force=True)
    print(result)
    if not result["ok"]:
        return
    input("检查 NL 生成 JSON，确认无误后按 Enter 继续...")

    # ── Media Stage 4 ──
    result = await build_media_index_stage4(FILE_ID, force=True)
    print(result)
    if not result["ok"]:
        return
    input("确认 Stage 4 完成，按 Enter 继续...")

    # ── Text Stage 1 ──
    result = await build_text_entities_stage1(FILE_ID, force=True)
    print(result)
    if not result["ok"]:
        return
    input("检查 multi_cleaned_chunks.json，确认无误后按 Enter 继续...")

    # ── Text Stage 2 ──
    result = await build_text_entities_stage2(FILE_ID, force=True)
    print(result)
    if not result["ok"]:
        return

asyncio.run(main())