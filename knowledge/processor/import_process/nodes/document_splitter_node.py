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
        # 进行文本切分-按照标题切分-切大了继续切分-切小了合并
        # 1. 进行转义字符格式化
        md_content, file_title = self._format_md_content(state)
        # 2. 根据标题进行切分
        chunks = self._split_by_title(md_content, file_title)
        config = get_config()
        # 3. 再次切分或者合并chunks
        chunks = self._split_and_merge(chunks, config.max_content_length,
                                                config.min_content_length)
        state["chunks"] = self.collect_data(chunks)
        #5 其他：输出操作日志 和 备份json格式
        self.output_log(md_content,chunks,config.max_content_length)

        return state

    # 1. 格式化md_content
    def _format_md_content(self, state: ImportGraphState):
        md_content = state.get("md_content")
        if md_content:
            md_content = (md_content.replace("\r\n", "\n")
                          .replace("\r", "\n"))
        file_tittle = state.get("file_title")
        return md_content, file_tittle

    # 2.根据标题切分内容
    def _split_by_title(self, md_content, file_title):
        tittle_pattern = re.compile(r"^\s*(#{1,6})\s+")
        # md_content转成lines
        lines = md_content.split("\n")

        temp = []  # 临时储存结果
        res = []  # 储存最后结果
        title = ""  # 储存当前tittle
        is_code = False  # 是否是代码块
        level_tittle = [""] * 7  # 存储多级标题
        current_level = 0

        # 逐行匹配
        for line in lines:
            # 判断是否是代码块
            if line.strip() in ("~~~", "```"):
                is_code = not is_code  # 如果是代码块则为True
            # 如果不是代码块并且匹配到标题
            if not is_code and tittle_pattern.match(line):
                data = "\n".join(temp)  # 把temp中的内容作为chunk的body加入res之中
                if title or data:
                    parent_title = ""  # 初始化父标题
                    for level in range(current_level - 1, 0, -1):
                        if level_tittle[level]:
                            parent_title = level_tittle[level]
                            break
                    if not parent_title:  # 如果没有父标题，附一个默认值
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
                    # 清空后面的标题
                    for level in range(level + 1, 7):
                        level_tittle[level] = ""

                tittle = line
                temp = []  # 清空temp
            else:
                temp.append(line)
        # 最后一段没有遇到新的标题，需要单独处理
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

    # 3.大的切小的合并
    def _split_and_merge(self, chunks, max_len, min_len):
        current_parts = []
        for chunk in chunks:
            current_parts.extend(self._split_again(chunk, max_len))

        final_chunks = self._merge_chunk(current_parts, min_len)

        return final_chunks

    # 大的切
    def _split_again(self, chunk, max_len):
        title = chunk.get("title")
        body = chunk.get("body")
        file_tittle = chunk.get("file_title")
        parent_title = chunk.get("parent_title")

        # 判断是否大于max_len
        # 处理标题
        if len(title) > 50:
            title = title[:50]
        tittle_prefix = f"{title}\n\n"
        total_len = len(tittle_prefix) + len(body)

        if total_len <= max_len:
            return [chunk]

        body_size = max_len - total_len
        # 比最大值大，递归切分
        text_splitter = RecursiveCharacterTextSplitter(
            # 切分每段大小是多大
            chunk_size=body_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
            keep_separator=False,
        )
        texts = text_splitter.split_text(body)
        final_chunks = []
        for index, text in enumerate(texts):
            final_chunks.append({
                "title": title + "-" + f"{index + 1}",
                "body": text,
                "file_title": file_tittle,
                "parent_title": parent_title,
                "part": f"{index + 1}",
            })
        return final_chunks

    # 小的合并
    def _merge_chunk(self, current_parts, min_len):
        # 贪婪策略匹配，以第一段为基准
        current_part = current_parts[0]
        final_parts = []

        for next_part in current_parts[1:]:
            # 从第二段开始匹配
            same_parent = next_part["parent_title"] == current_part["parent_title"]
            if same_parent and len(current_part.get("body")) < min_len:
                current_part['body'] = current_part.get('body').rstrip() + "\n\n" + next_part.get('body').rstrip()
                current_part['title'] = current_part["parent_title"]
                current_part['part'] = 0
            else:
                final_parts.append(current_part)
                current_part = next_part
        # 处理最后一段
        final_parts.append(current_part)
        # 专门处理每段内容part字段（根据业务也可以不处理）
        part_counter = {}
        result = []
        for parts in final_parts:
            if "part" in parts:
                parent_title = parts["parent_title"]

                part_counter[parent_title] = (
                        part_counter.get(parent_title, 0) + 1)

                new_part = part_counter[parent_title]

                parts["part"] = new_part
            result.append(parts)
        return result

    # 把拆分合并之后，最终组装的数据备份
    # 把数据生成json格式文件，保存到本地目录
    def copy_chunks(self, state):
        file_dir = state.get('file_dir')
        # "file_dir": r"D:\dev\test\a.json"
        output_file_dir = os.path.join(file_dir, 'chunks.json')
        with open(output_file_dir, 'w', encoding='utf-8') as f:
            json.dump(state["chunks"], f, ensure_ascii=False, indent=4)

    # 输出日志
    def output_log(self, md_content, chunks, max_content_length):
        # 统计md文档行
        line_count = md_content.count("\n") + 1
        self.logger.info(f"md文档共有{line_count}行")
        self.logger.info(f"最终切分章节数:{len(chunks)}")
        self.logger.info(f"最大切片长度:{max_content_length}")

        if chunks:
            self.logger.info("章节预览:")  # 只前5个章节，做简单预览；i为下标，sec为单个章节字典
            for i, sec in enumerate(chunks[:5]):
                # 取出标题，并且截断为前30个字符
                title = sec.get("title", "")[:30]
                self.logger.info(f"{i + 1}.{title}")
            if len(chunks) > 5:
                self.logger.info(f"还有{len(chunks) - 5}章节")

    # 对切分和合并数据重新组装
    def collect_data(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        chunks = []
        for chunk in final_chunks:
            # {
            #     "title": title + "-" + f"{index + 1}",
            #     "body": text,
            #     "file_title": file_title,
            #     "parent_title": parent_title,
            #     "part": f"{index + 1}",
            # }
            title = chunk.get('title')
            body = chunk.get('body')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')

            # 最终内容 ：content 包含 title + body
            content = f"{title}\n\n{body}"

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
    # 构建md_content
    file_path = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto\hak180产品安全手册.md"
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    # 构建数据
    state = {
        "file_title": "test",
        "md_content": file_content,
        "file_dir": r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto"
    }

    print(doc_split_node.process(state))
