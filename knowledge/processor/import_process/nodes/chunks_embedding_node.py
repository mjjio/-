import json
from pathlib import Path
from typing import Any, Dict, List

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bgem3_client_util import get_bgem3_client

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bgem3_client_util import get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client

# 引入局部切片提取所需的提示词
from knowledge.prompts.upload.import_prompt import (
    LOCAL_CHUNK_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE
)


class ChunkEmbeddingNode(BaseNode):
    """
    切片处理与嵌入节点：
    1. 调用 LLM 提取每个切片的局部实体信息，将其扁平化地并入切片字典（与 content 平级）。
    2. 调用 BGE-M3 模型，基于切片原文与提取出的标签，生成稠密与稀疏双路向量。
    """

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.参数验证
        chunks = self._validate_params(state)
        file_title = state.get("file_title", "")

        # 2.遍历chunks，分批次处理以提高效率
        batch_size = 3
        len_chunks = len(chunks)
        final_chunks = []

        self.logger.info(f"开始执行切片信息提取与向量嵌入，共 {len_chunks} 个切片...")

        for i in range(0, len_chunks, batch_size):
            batch = chunks[i:i + batch_size]

            # 2.1 先调用大模型提取切片实体，并将结果拍平到每个切片的字典中
            batch = self._extract_entities_for_chunks(batch)

            # 2.2 随后基于这些信息进行嵌入操作
            batch = self._embedding_chunk(batch, file_title)

            # 2.3 汇入最终结果
            final_chunks.extend(batch)
            self.logger.info(f"已完成 {min(i + batch_size, len_chunks)} / {len_chunks} 个切片的处理")

        state['chunks'] = final_chunks
        return state

    def _validate_params(self, state):
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("切片内容缺失")
        return chunks

    def _extract_entities_for_chunks(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        调用大语言模型，提取实体信息并扁平化地保存到 chunk 字典中
        """
        llm_client = get_llm_client()

        for chunk in batch:
            content = chunk.get("content", "")

            # 1. 组装提示词并调用大语言模型
            prompt = USER_PROMPT_TEMPLATE.format(context=content)
            message = [
                SystemMessage(content=LOCAL_CHUNK_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            try:
                response = llm_client.invoke(message)
                result_text = response.content if response else ""

                # 清理 LLM 输出的 markdown 格式标识符
                # 清洗 LLM 返回结果，兼容 markdown 的 json 语法
                content = re.sub(r"^```json\s*", "", result_text, flags=re.IGNORECASE)
                content = re.sub(r"\s*```$", "", content)

                extracted_info = json.loads(content)

            except Exception as e:
                self.logger.error(f"切片实体提取或解析失败: {e}")
                extracted_info = {}

            # 2. 将提取到的信息扁平化写入（和 content 同级），使用 or [] 进行空值兜底
            chunk["attraction_names"] = extracted_info.get("attraction_names") or []
            chunk["route_names"] = extracted_info.get("route_names") or []
            chunk["hotel_names"] = extracted_info.get("hotel_names") or []
            chunk["restaurant_names"] = extracted_info.get("restaurant_names") or []

        return batch

    def _embedding_chunk(self, batch: List[Dict[str, Any]], file_title: str) -> List[Dict[str, Any]]:
        """
        对富化后的切片生成向量并存回字典
        """
        bge_m3_client = get_bgem3_client()
        content_data = []

        for chunk in batch:
            content = chunk.get("content", "")

            # 构造标签字符串，用来辅助文本做向量化，提升检索效果
            tags_desc = []
            if chunk.get("attraction_names"): tags_desc.append("景点: " + ",".join(chunk["attraction_names"]))
            if chunk.get("route_names"): tags_desc.append("线路: " + ",".join(chunk["route_names"]))
            if chunk.get("hotel_names"): tags_desc.append("酒店: " + ",".join(chunk["hotel_names"]))
            if chunk.get("restaurant_names"): tags_desc.append("餐厅: " + ",".join(chunk["restaurant_names"]))

            tags_str = f"[{' | '.join(tags_desc)}]" if tags_desc else ""

            # 将文档标题、提取出的特征标签与原文组合，生成高度语义化的文本用于嵌入
            embed_text = f"来源文档: {file_title}\n{tags_str}\n{content}" if file_title else f"{tags_str}\n{content}"
            content_data.append(embed_text.strip())

        # 批量进行向量嵌入
        response = bge_m3_client.encode_documents(content_data)

        # 取出稠密和稀疏向量，写回当前切片字典
        for index, chunk in enumerate(batch):
            chunk["dense_vector"] = response["dense"][index].tolist()

            csr_result = response["sparse"]
            start_index = csr_result.indptr[index]
            end_index = csr_result.indptr[index + 1]

            # 解析稀疏向量的 ID 和权重 Data
            token_id = csr_result.indices[start_index:end_index].tolist()
            data = csr_result.data[start_index:end_index].tolist()

            # 将向量数据直接保存在切片字典中
            chunk["sparse_vector"] = dict(zip(token_id, data))

        return batch


if __name__ == '__main__':

    base_temp_dir = Path(
        r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto")

    input_path = base_temp_dir / "chunks_item_name.json"
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f" 找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态 (Graph State)
    state = {
        "chunks": content
    }

    # 3. 触发节点执行
    node_bge_embedding = ChunkEmbeddingNode()
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f" 向量生成测试完成！结果已成功备份至:\n{output_path}")
