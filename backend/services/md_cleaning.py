import re
import os


class MarkdownHeadingCleaner:
    def __init__(self):
        # 匹配任意层级的 Markdown 标题
        self.any_heading_pattern = re.compile(r'^(#{1,6})\s+(.*)')

        # 第1级: 第x章
        self.l1_pattern = re.compile(r'^第\s*[0-9一二三四五六七八九十百]+\s*章')
        # 第2级: 第x节
        self.l2_pattern = re.compile(r'^第\s*[0-9一二三四五六七八九十百]+\s*节')
        # 第6级: x.x.x.x  (新增) ── 必须排在 L5 前面判定
        self.l6_pattern = re.compile(r'^\d+\.\d+\.\d+\.\d+(?:[\s\u3000、]|$)')
        # 第5级: x.x.x
        self.l5_pattern = re.compile(r'^\d+\.\d+\.\d+(?:[\s\u3000、]|$)')
        # 第4级: x.x
        self.l4_pattern = re.compile(r'^\d+\.\d+(?:[\s\u3000、]|$)')
        # 第3级: x 或 附录
        self.l3_pattern = re.compile(r'^\d+(?:[\s\u3000、.]|$)')
        self.l3_appendix_pattern = re.compile(r'^附\s*录\s*[A-Z0-9]')

        # 企业标准编号，例如 Q/GSIJ 0404021—2019
        self.std_code_pattern = re.compile(r'^Q/[a-zA-Z0-9\s]+[—\-]\d{4}$')

        # 页面分隔标记──直接透传，不做任何处理
        self.page_marker_pattern = re.compile(r'^<!--\s*page:\s*\d+\s*-->$')

    def clean_line(self, line: str):
        """
        清洗单行文本。
        返回清洗后的字符串；如果该行应被删除，则返回 None。
        """
        stripped_line = line.strip()

        # page marker 显式保护
        if self.page_marker_pattern.match(stripped_line):
            return line

        # --- 规则1：删除 "单位为毫米" 单独成行 ---
        if stripped_line.strip('。.') == "单位为毫米":
            return None

        # --- 规则2：删除标准编号单独成行 ---
        if self.std_code_pattern.match(stripped_line):
            return None

        # 如果当前行不是标题格式，直接返回原貌
        match = self.any_heading_pattern.match(line)
        if not match:
            return line

        title_text = match.group(2).strip()
        if not title_text:
            return ""

        # --- 核心判定逻辑（优先判定深层数字，避免短路误判）---
        # 顺序：L1 → L2 → L6 → L5 → L4 → L3
        # L6 必须排在 L5 前面，否则 X.X.X.X 会被 L5 拒绝后落到 L3 误判
        if self.l1_pattern.match(title_text):
            return f"# {title_text}"
        elif self.l2_pattern.match(title_text):
            return f"## {title_text}"
        elif self.l6_pattern.match(title_text):
            return f"###### {title_text}"
        elif self.l5_pattern.match(title_text):
            return f"##### {title_text}"
        elif self.l4_pattern.match(title_text):
            return f"#### {title_text}"
        elif self.l3_pattern.match(title_text) or self.l3_appendix_pattern.match(title_text):
            return f"### {title_text}"
        else:
            # 不符合任何规则的标题 → 降级为普通文本（去掉 '#'）
            return title_text

    def process_file(self, input_path: str, output_path: str):
        """处理整个文件并输出清洗后的文件"""
        if not os.path.exists(input_path):
            print(f"错误: 找不到文件 {input_path}")
            return

        print(f"开始读取文件: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            cleaned_line = self.clean_line(line.rstrip('\n'))
            if cleaned_line is not None:
                cleaned_lines.append(cleaned_line)

        content = '\n'.join(cleaned_lines)

        # 删除行后会产生连续空行，压缩为最多 2 个换行（1 个空行）
        content = re.sub(r'\n{3,}', '\n\n', content)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content.strip() + '\n')

        print(f"清洗完成！已保存至: {output_path}")


# === 使用示例 ===
if __name__ == "__main__":
    cleaner = MarkdownHeadingCleaner()
    input_file  = "C:\\Users\\Lenovo\\Desktop\\project_2_3.29\\project_2\\backend\\data\\f_55l2wt09\\output.md"
    output_file = "C:\\Users\\Lenovo\\Desktop\\project_2_3.29\\project_2\\backend\\data\\f_55l2wt09\\output.md"
    cleaner.process_file(input_file, output_file)