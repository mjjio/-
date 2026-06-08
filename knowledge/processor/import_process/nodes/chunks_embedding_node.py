import json
from pathlib import Path
from typing import Any, Dict, List

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bgem3_client_util import get_bgem3_client


class ChunkEmbeddingNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.参数验证
        chunks = self._validate_params(state)
        # 2.遍历chunks，调用模型生成稠密系数向量
        # 为了提高效率，分批次做嵌入操作
        batch_size = 3
        len_chunks = len(chunks)
        final_chunks = []

        for i in range(0, len_chunks, batch_size):
            # 2.1 分批次进行向量嵌入，最后一批不够会正常返回List[dict]
            chunk = chunks[i:i+batch_size]
            # 2.2 进行嵌入操作
            batch = self._embedding_chunk(chunk)
            # 2.3 接入最终结果
            final_chunks.extend(batch)
        state['chunks'] = final_chunks
        return state

    def _validate_params(self, state):
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("切片内容缺失")
        return chunks

    def _embedding_chunk(self, batch:List[Dict[str,Any]]) -> List[Dict[str,Any]]:

        # 获得bge-m3模型客户端
        bge_m3_client = get_bgem3_client()

        # 包装content_data
        content_data = []
        for _, chunk in enumerate(batch):
            content = chunk.get("content")
            item_name = chunk.get("item_name")

            # 拼接最终结果
            data = f"{item_name} - {content}"

            content_data.append(data)

        # 做向量嵌入
        response = bge_m3_client.encode_documents(content_data)

        # 取出稠密和稀疏向量
        for index, part in enumerate(batch):
            dense_vector = response["dense"][index].tolist()

            csr_result = response["sparse"]

            start_index = csr_result.indptr[index]
            end_index = csr_result.indptr[index+1]

            # 获得token_id
            token_id = csr_result.indices[start_index:end_index].tolist()
            # 获得data
            data = csr_result.data[start_index:end_index].tolist()

            # 稀疏向量
            sparse_vector = dict(zip(token_id, data))

            # 存到字典里返回
            part["dense_vector"] = dense_vector
            part["sparse_vector"] = sparse_vector

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
