import json

from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_client_util import get_milvus_client


class _MilvusSchemaBuilder:
    @staticmethod
    def build_milvus_schema(milvus_client):
        # 创建约束，开启动态字段以兼容切片中可能存在的其他扩展字段（如 part）
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 1. 主键约束
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )

        # 2. 向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=1024,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 3. 基础标量字段
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)

        # 4. 【核心修改】旅游场景的扁平化实体数组字段 (替代原 item_name)
        schema.add_field(field_name="attraction_names", datatype=DataType.ARRAY, element_type=DataType.VARCHAR,
                         max_capacity=50, max_length=500)
        schema.add_field(field_name="route_names", datatype=DataType.ARRAY, element_type=DataType.VARCHAR,
                         max_capacity=50, max_length=500)
        schema.add_field(field_name="hotel_names", datatype=DataType.ARRAY, element_type=DataType.VARCHAR,
                         max_capacity=50, max_length=500)
        schema.add_field(field_name="restaurant_names", datatype=DataType.ARRAY, element_type=DataType.VARCHAR,
                         max_capacity=50, max_length=500)

        return schema


class _MilvusIndexBuilder:
    @staticmethod
    def build_milvus_index(milvus_client):
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
        return index_params


class _MilvusInsertBuilder:
    def __init__(self, milvus_client, collection_name):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert(self, chunks):
        # 插入切片结果，Milvus 会自动根据 chunks 字典中的 key 映射到 schema
        insert_result = self._milvus_client.insert(
            collection_name=self._collection_name,
            data=chunks,
        )
        ids = insert_result["ids"]
        # 把返回的自增主键写入到原始字典列表中
        self._fill_chunk_ids(chunks, ids)
        return chunks

    def _fill_chunk_ids(self, chunks, ids):
        for chunk, id in zip(chunks, ids):
            chunk["chunk_id"] = id


# 调用节点
class ImportMilvusNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        chunks = self._validate_params(state)

        # 2. 上传向量数据库
        milvus_client = get_milvus_client()
        config = get_config()
        # 读取配置文件中的 chunks_collection (切片库名称)
        collection_name = getattr(config, 'chunks_collection', 'tourism_local_chunks_v1')

        # 3. 判断 collection 是否存在，不存在会自动建表和索引
        self._is_has_collection(milvus_client, collection_name)

        # 4. 上传数据并更新状态里的 chunks (回填了 chunk_id)
        milvus_insert_obj = _MilvusInsertBuilder(milvus_client, collection_name)
        state["chunks"] = milvus_insert_obj.insert(chunks)

        state["status"] = "completed"
        return state

    def _validate_params(self, state):
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("chunks为空")
        return chunks

    def _is_has_collection(self, milvus_client, collection_name):
        # 幂等性，如果不存在则创建
        if not milvus_client.has_collection(collection_name):
            schema = _MilvusSchemaBuilder.build_milvus_schema(milvus_client)
            index_params = _MilvusIndexBuilder.build_milvus_index(milvus_client)

            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )


if __name__ == '__main__':
    input_path = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto\chunks_vector.json"

    import os

    if os.path.exists(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            file_content = json.load(f)

        # 模拟 State
        state: ImportGraphState = {
            "chunks": file_content
        }

        # 调用
        import_milvus = ImportMilvusNode()
        result = import_milvus.process(state)

        # 调用返回结果写入到新 json 文件里面 (包含 Milvus 回填的 chunk_id)
        output_path = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto\chunks_embedding_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print("入库成功，并生成了包含 chunk_id 的新文件。")
    else:
        print(f"找不到测试文件: {input_path}")