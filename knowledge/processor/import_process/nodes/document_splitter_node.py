import json
import os
import re
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState


class DocumentSplitterNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 进行转义字符格式化
        md_content, file_title = self._format_md_content(state)
        # 2. 根据标题进行切分
        chunks = self._split_by_title(md_content, file_title)
        config = get_config()
        # 3. 再次切分或者合并chunks
        chunks = self._split_and_merge(chunks, config.max_content_length, config.min_content_length)
        state["chunks"] = self.collect_data(chunks)

        # 4. 其他：输出操作日志
        self.output_log(md_content, chunks, config.max_content_length)

        return state

    def _format_md_content(self, state: ImportGraphState):
        md_content = state.get("md_content", "")
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        file_title = state.get("file_title", "未命名文档")
        return md_content, file_title

    def _split_by_title(self, md_content, file_title):
        tittle_pattern = re.compile(r"^\s*(#{1,6})\s+")
        lines = md_content.split("\n")

        temp = []
        res = []
        title = ""
        is_code = False
        level_tittle = [""] * 7
        current_level = 0

        for line in lines:
            if line.strip() in ("~~~", "```"):
                is_code = not is_code

            if not is_code and tittle_pattern.match(line):
                data = "\n".join(temp)
                if title or data:
                    parent_title = ""
                    for level in range(current_level - 1, 0, -1):
                        if level_tittle[level]:
                            parent_title = level_tittle[level]
                            break
                    if not parent_title:
                        parent_title = title if title else file_title

                    res.append({
                        "title": title,
                        "body": data,
                        "file_title": file_title,
                        "parent_title": parent_title,
                    })

                match_obj = tittle_pattern.match(line)
                if match_obj:
                    level = len(match_obj.group(1))
                    current_level = level
                    level_tittle[level] = line
                    for level in range(level + 1, 7):
                        level_tittle[level] = ""

                # 【修复】这里之前拼写成了 tittle = line，导致标题全部为空
                title = line
                temp = []
            else:
                temp.append(line)

        # 最后一段处理
        data = "\n".join(temp)
        if title or data:
            parent_title = ""
            for lv in range(current_level - 1, 0, -1):
                if level_tittle[lv]:
                    parent_title = level_tittle[lv]
                    break
            if not parent_title:
                parent_title = title if title else file_title

            res.append({
                "title": title,
                "body": data,
                "file_title": file_title,
                "parent_title": parent_title,
            })

        return res

    def _split_and_merge(self, chunks, max_len, min_len):
        current_parts = []
        for chunk in chunks:
            current_parts.extend(self._split_again(chunk, max_len))

        final_chunks = self._merge_chunk(current_parts, min_len)
        return final_chunks

    def _split_again(self, chunk, max_len):
        title = chunk.get("title", "")
        body = chunk.get("body", "")
        file_title = chunk.get("file_title")
        parent_title = chunk.get("parent_title")

        if len(title) > 50:
            title = title[:50]
        tittle_prefix = f"{title}\n\n"
        total_len = len(tittle_prefix) + len(body)

        if total_len <= max_len:
            return [chunk]

        body_size = max_len - total_len
        if body_size <= 0:
            body_size = max_len // 2  # 兜底机制，防止极端情况抛错

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
            keep_separator=False,
        )
        texts = text_splitter.split_text(body)
        final_chunks = []
        for index, text in enumerate(texts):
            final_chunks.append({
                "title": f"{title}-{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}",
            })
        return final_chunks

    def _merge_chunk(self, current_parts, min_len):
        # 【修复】关键保护逻辑：如果传进来的是空列表，直接返回空列表
        if not current_parts:
            return []

        current_part = current_parts[0]
        final_parts = []

        for next_part in current_parts[1:]:
            same_parent = next_part.get("parent_title") == current_part.get("parent_title")
            if same_parent and len(current_part.get("body", "")) < min_len:
                current_part['body'] = current_part.get('body', '').rstrip() + "\n\n" + next_part.get('body',
                                                                                                      '').rstrip()
                current_part['title'] = current_part.get("parent_title")
                current_part['part'] = 0
            else:
                final_parts.append(current_part)
                current_part = next_part

        final_parts.append(current_part)

        part_counter = {}
        result = []
        for parts in final_parts:
            if "part" in parts:
                parent_title = parts.get("parent_title")
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                parts["part"] = part_counter[parent_title]
            result.append(parts)

        return result

    def copy_chunks(self, state):
        file_dir = state.get('file_dir')
        if not file_dir:
            return
        output_file_dir = os.path.join(file_dir, 'chunks.json')
        with open(output_file_dir, 'w', encoding='utf-8') as f:
            json.dump(state.get("chunks", []), f, ensure_ascii=False, indent=4)

    def output_log(self, md_content, chunks, max_content_length):
        line_count = md_content.count("\n") + 1
        self.logger.info(f"md文档共有{line_count}行")
        self.logger.info(f"最终切分章节数:{len(chunks)}")
        self.logger.info(f"最大切片长度:{max_content_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"{i + 1}.{title}")
            if len(chunks) > 5:
                self.logger.info(f"还有{len(chunks) - 5}章节")

    def collect_data(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for chunk in final_chunks:
            title = chunk.get('title', '')
            body = chunk.get('body', '')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')

            content = f"{title}\n\n{body}".strip()

            data = {
                "title": title,
                "content": content,
                "file_title": file_title,
                "parent_title": parent_title,
            }
            if "part" in chunk:
                data['part'] = chunk.get('part')
            chunks.append(data)
        return chunks


if __name__ == '__main__':
    doc_split_node = DocumentSplitterNode()
    file_path = r"E:\project\初始化项目\交通指南\成都交通指南.md"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        state = {
            "file_title": "test",
            "md_content": file_content,
            "file_dir": r"E:\project\初始化项目\交通指南"
        }

        print(doc_split_node.process(state))
    else:
        print(f"找不到测试文件: {file_path}")