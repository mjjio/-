import json

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.prompts.upload.import_prompt import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.bgem3_client_util import get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client


class ItemNameRecognitionNode(BaseNode):

    def process(self, state:ImportGraphState) -> ImportGraphState:

        # 1.参数校验
        file_title, chunks = self._validate_param(state)

        # 2.根据提示词模板构建提示词
        prompt_data = self._create_prompt(chunks)

        # 3.调用大语言模型返回item_name
        item_name = self._call_llm(file_title, prompt_data)
        # 4.1 商品名嵌入
        dense_vectors, sparse_vectors = self._embedding_to_vectors(item_name)

        # 4.2 存到向量数据库
        self._save_to_milvus(file_title, item_name, dense_vectors, sparse_vectors)

        # 5 更新state
        self._update_state(state, item_name, chunks)

        return state


    def _validate_param(self, state):
        title , chunks = state['file_title'], state['chunks']
        if not title:
            raise ValueError("文件标题为空")
        if not chunks:
            raise ValueError("切片内容为空")
        return title, chunks

    def _create_prompt(self, chunks:list[dict]) -> str:
        # 获得前五个切片拼接起来作为字符串返回
        res = []
        total = 0
        config = get_config()

        for index,chunk in enumerate(chunks[:5]):
            content = chunk.get("content")
            # 拼接chunk内容
            chunk_data = f"切片-{index+1}-{content}"
            total += len(chunk_data)
            if total > config.max_content_length:
                break
            res.append(chunk_data)
        return "\n".join(res)

    def _call_llm(self, file_tittle, prompt_data):
        llm_client = get_llm_client()

        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_tittle, context=prompt_data)

        # 根据提示词模板构建提示词
        message = [
            SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        # 调用大语言模型
        response = llm_client.invoke(message)

        # 判断是否有结果，如果没有结果返回一个默认值
        if not response:
            return file_tittle
        return response.content

    def _embedding_to_vectors(self, item_name):
        # 使用bge-m3模型做向量嵌入
        client = get_bgem3_client()
        embedding_result = client.encode_documents([item_name])
        # 获取embedding_result之中的稠密向量和稀疏向量
        # dense 返回的结果是一个二维数组
        dense = embedding_result['dense'][0].tolist()
        # sparse 返回的结果是一个稀疏矩阵
        sparse = embedding_result['sparse']
        sparse = dict(zip(sparse.indices.tolist(), sparse.data.tolist()))

        return dense, sparse

    def _save_to_milvus(self, file_title, item_name, dense_vectors, sparse_vectors):
        # 1. 获取milvus客户端
        client = get_milvus_client()

        config = get_config()

        # 2. 幂等性(是否存在collection),不存在就创建
        if not client.has_collection(collection_name=config.item_name_collection,):
            self._create_collection(client,config.item_name_collection)

        # 3. 添加数据到collection之中
        data = {
            "file_title":file_title,
            "item_name":item_name,
            "dense_vector":dense_vectors,
            "sparse_vector":sparse_vectors
        }
        result = client.insert(
            collection_name=config.item_name_collection,
            data = [data])
        self.logger.info("向量数据库添加之后结果",result)


    def _create_collection(self, milvus_client, collection_name):
        # 1.创建约束
        schema = milvus_client.create_schema()

        schema.add_field(field_name = "id",
                         datatype=DataType.VARCHAR,
                         auto_id = True,
                         is_primary = True,
                         max_length = 100)
        schema.add_field(field_name="file_title",
                         datatype=DataType.VARCHAR,
                         max_length=1000)
        schema.add_field(field_name = "item_name",
                         datatype=DataType.VARCHAR,
                         max_length = 1000)
        schema.add_field(field_name = "dense_vector",
                         datatype=DataType.FLOAT_VECTOR,
                         dim=1024)
        schema.add_field(field_name = "sparse_vector",
                         datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 添加索引
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

        # 3. 创建collection
        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema,
                                        index_params=index_params)

    def _update_state(self,state, item_name, chunks):
        for chunk in chunks:
            chunk["item_name"] = item_name
        state['item_name'] = item_name

if __name__ == '__main__':

    chunk_json_path = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\chunks.json"
    with open(chunk_json_path,"r",encoding="utf-8") as f:
        chunks_content = json.load(f)

    # file_title,chunks
    state = {
        "file_title":"H3CLA2608室内无线网关用户手册-6W100-整本手册",
        "chunks":chunks_content
    }
    itemNameRecognition = ItemNameRecognitionNode()
    result = itemNameRecognition.process(state)

    output_dir = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\H3CLA2608室内无线网关用户手册-6W100-整本手册\auto\chunks_item_name.json"
    with open(output_dir,"w",encoding="utf-8") as f:
        json.dump(state["chunks"],f,ensure_ascii=False,indent=4)

    print(result)