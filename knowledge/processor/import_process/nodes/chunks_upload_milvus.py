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
        # 创建约束
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 加约束
        schema.add_field(
            field_name = "chunk_id",
            datatype = DataType.INT64,
            is_primary = True,
            auto_id = True,
        )
        # 向量字段
        schema.add_field(
            field_name = "dense_vector",
            datatype = DataType.FLOAT_VECTOR,
            dim = 1024,
        )
        schema.add_field(
            field_name = "sparse_vector",
            datatype = DataType.SPARSE_FLOAT_VECTOR,
        )
        # 标量字段
        schema.add_field(
            field_name = "content",
            datatype = DataType.VARCHAR,
            max_length = 65535,
        )
        schema.add_field(field_name="title",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="parent_title",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="file_title",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        schema.add_field(field_name="item_name",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
        return schema

class _MilvusIndexBuilder:
    @staticmethod
    def build_milvus_index(milvus_client):
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name = "dense_vector",
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
        # 插入切片结果，把主键更新到chunks之中
        insert_result = self._milvus_client.insert(
            collection_name=self._collection_name,
            data=chunks,
        )
        ids = insert_result["ids"]
        self._fill_chunk_ids(chunks, ids)
        return chunks

    def _fill_chunk_ids(self, chunks, ids):
        for chunk,id in zip(chunks, ids):
            chunk["chunk_id"] = id

# 调用节点    
class ImportMilvusNode(BaseNode):
    def process(self, state):
        # 1. 参数校验
        chunks = self._validate_params(state)
        # 2. 上传向量数据库
        milvus_client = get_milvus_client()
        config = get_config()
        collection_name = config.chunks_collection

        # 判断collection是否存在
        self._is_has_collection(milvus_client, collection_name)

        # 上传向量数据库更新状态
        milvus_insert_obj = _MilvusInsertBuilder(milvus_client, collection_name)
        state["chunks"] = milvus_insert_obj.insert(chunks)
        return state

    def _validate_params(self, state):
        chunks = state["chunks"]
        if not chunks:
            raise ValidationError("chunks为空")
        return chunks

    def _is_has_collection(self, milvus_client, collection_name):
        # 幂等性，如果不存在则创建
        if not milvus_client.has_collection(collection_name):
            schema = _MilvusSchemaBuilder.build_milvus_schema(milvus_client)
            index_params = _MilvusIndexBuilder.build_milvus_index(milvus_client)

            milvus_client.create_collection(
                collection_name = collection_name,
                schema=schema,
                index_params=index_params
            )

if __name__ == '__main__':
    input_path = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto\chunks_vector.json"
    with open(input_path,"r",encoding="utf-8") as f:
        file_content = json.load(f)
    state:ImportGraphState = {
        "chunks": file_content.get("chunks")
    }

    # 调用
    import_milvus = ImportMilvusNode()
    result = import_milvus.process(state)

    # 调用返回结果写入到新json文件里面
    output_path = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto\chunks_embedding_chunks.json"
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=4)