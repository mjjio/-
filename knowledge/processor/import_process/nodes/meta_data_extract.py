import json
import re
from typing import Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bgem3_client_util import get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client

from knowledge.prompts.upload.import_prompt import (
    GLOBAL_METADATA_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE
)


class LocationIntentRecognitionNode(BaseNode):
    """
    文档意图与目的地识别节点 (全局信息提取)
    """

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        file_title, chunks = self._validate_param(state)

        # 2. 根据提示词模板构建提示词上下文 (仅取前几个切片作为摘要)
        prompt_data = self._create_prompt(chunks)

        # 3. 调用大语言模型返回识别出的文档意图和目的地
        file_class, destination = self._call_llm(prompt_data)

        # 4.1 将意图和目的地进行文本拼接，并做向量嵌入
        dense_vectors, sparse_vectors = self._embedding_to_vectors(file_class, destination, prompt_data)

        # 4.2 存到向量数据库 (仅存全局文档表)
        self._save_to_milvus(file_title, file_class, destination, dense_vectors, sparse_vectors)

        # 5. 更新 state，供后续的切片嵌入节点和图谱节点使用
        self._update_state(state, file_class, destination, dense_vectors, sparse_vectors)

        return state

    def _validate_param(self, state: ImportGraphState):
        title, chunks = state.get('file_title'), state.get('chunks')
        if not title:
            raise ValueError("文件标题为空")
        if not chunks:
            raise ValueError("切片内容为空")
        return title, chunks

    def _create_prompt(self, chunks: list) -> str:
        # 获得前几个切片拼接起来作为字符串返回
        res = []
        total = 0
        config = get_config()
        chunk_limit = getattr(config, 'global_metadata_chunk_k', 5)

        for index, chunk in enumerate(chunks[:chunk_limit]):
            content = chunk.get("content") if isinstance(chunk, dict) else getattr(chunk, 'page_content', str(chunk))
            chunk_data = f"切片-{index + 1}-{content}"
            total += len(chunk_data)
            if total > getattr(config, 'max_content_length', 2000):
                break
            res.append(chunk_data)

        return "\n".join(res)

    def _call_llm(self, prompt_data: str) -> Tuple[str, str]:
        llm_client = get_llm_client()

        prompt = USER_PROMPT_TEMPLATE.format(context=prompt_data)
        message = [
            SystemMessage(content=GLOBAL_METADATA_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = llm_client.invoke(message)

        # 判断是否有结果，如果没有则返回默认值
        if not response or not response.content:
            return "旅游综合", "未知目的地"

        # 清洗 LLM 返回结果，兼容 markdown 的 json 语法
        content = re.sub(r"^```json\s*", "", response.content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)

        try:
            metadata_dict = json.loads(content)
            # 为了适配可能不同的 key 命名做兼容
            file_class = str(metadata_dict.get("file_class", metadata_dict.get("content_type", "旅游综合")))
            destination = str(metadata_dict.get("destination", metadata_dict.get("region_name", "未知目的地")))
            return file_class, destination
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 解析失败: {e}\n原文: {content}")
            return "旅游综合", "未知目的地"

    def _embedding_to_vectors(self, file_class: str, destination: str, prompt_data: str):
        # 拼接成具有明确语义的句子，提升后续检索命中率
        semantic_text = f"文档意图: {file_class} \n 目的地: {destination} \n 摘要: {prompt_data[:300]}"

        client = get_bgem3_client()
        embedding_result = client.encode_documents([semantic_text])

        dense = embedding_result['dense'][0].tolist()

        sparse_result = embedding_result['sparse']
        sparse = dict(zip(sparse_result.indices.tolist(), sparse_result.data.tolist()))

        return dense, sparse

    def _save_to_milvus(self, file_title: str, file_class: str, destination: str, dense_vectors: list,
                        sparse_vectors: dict):
        # 1. 获取 milvus 客户端和配置
        client = get_milvus_client()
        config = get_config()
        collection_name = getattr(config, 'global_collection', 'tourism_global_docs_v1')

        # 2. 幂等性，不存在就创建
        if not client.has_collection(collection_name=collection_name):
            self._create_collection(client, collection_name)

        # 3. 添加数据到 collection 之中
        data = [{
            "file_title": file_title,
            "file_class": file_class,
            "destination": destination,
            "dense_vector": dense_vectors,
            "sparse_vector": sparse_vectors
        }]

        result = client.insert(
            collection_name=collection_name,
            data=data
        )
        self.logger.info(f"[{file_title}] 的全局意图向量数据库添加结果: {result}")

    def _create_collection(self, milvus_client, collection_name: str):
        # 1. 创建约束
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name="id", datatype=DataType.VARCHAR, auto_id=True, is_primary=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name="file_class", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="destination", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 添加混合索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 3. 创建 collection
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )

    def _update_state(self, state: ImportGraphState, file_class: str, destination: str, dense_vectors: list,
                      sparse_vectors: dict):
        state['file_class'] = file_class
        state['destination'] = destination
        state['global_dense_vector'] = dense_vectors
        state['global_sparse_vector'] = sparse_vectors
        state['status'] = 'intent_extracted'


if __name__ == '__main__':
    # 测试代码保持不变
    pass