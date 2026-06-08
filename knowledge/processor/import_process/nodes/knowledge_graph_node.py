import json
import re
from typing import Set, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType, MilvusClient

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.upload.import_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.utils.bgem3_client_util import get_bgem3_client
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client
from knowledge.utils.neo4j_util import Neo4jGraphWriter, get_neo4j_driver

# 实体名称最大长度
MAX_ENTITY_NAME_LENGTH = 15
# 实体标签白名单
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Device", "Part", "Operation", "Step",
    "Warning", "Condition", "Tool",
}
# 关系类型白名单
ALLOWED_RELATION_TYPES: Set[str] = ({
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
})
DEFAULT_RELATION_TYPES = "RELATED_TO"


# 1.定义内部类，用于进行数据库操作
class _MilvusEntityOperation:
    # 初始化：定义自己的实例属性：collection名字
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    # 静态方法1：进行集合创建(如果没有的话)
    @staticmethod
    def _ensure_collection(client, collection_name):

        # 如果不存在则创建
        if client.has_collection(collection_name):
            return

        # 构建schema
        schema = client.create_schema(enable_dynamic_fields=True)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

        # 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 创建集合
        client.create_collection(collection_name, schema=schema, index_params=index_params)

    # 方法1: 根据item_name进行删除
    def clear(self, milvus_client: MilvusClient, item_name:str):
        collection_name = self.collection_name
        if milvus_client.has_collection(collection_name):
            milvus_client.delete(collection_name
                                 ,filter=f'item_name=="{item_name}"')

    # 方法2：添加
    def insert(self, milvus_client: MilvusClient, entities:List, chunk_id:str,
               item_name:str, content:str):
        # entities是所有实体列表，获取列表所有实体名称
        # 遍历得到没有entity，获取每个entity里面name值非空
        # 把每个entity里面name去重操作，转换list列表
        # ['实体1','实体2']
        entities_names = list(dict.fromkeys(entity['name'] for entity in entities if entity.get('name')))

        if not entities_names:
            return
            # raise Exception(f"实体数据为空")

        # 调用方法：判断集合是否存在，如果不存在创建
        self._ensure_collection(milvus_client,self.collection_name)

        # 构建记录
        # 获取向量模型客户端对象
        bgem3_client = get_bgem3_client()
        # ['1','2']
        embedding_result = bgem3_client.encode_documents(entities_names)

        # 调用方法构建结果，添加向量数据库里面
        records = self._build_records(entities_names,embedding_result,
                            chunk_id,content,item_name)
        milvus_client.insert(
            collection_name=self.collection_name,
            data=records,
        )

        # 构建数据过程方法

    """组装插入记录。"""

    @staticmethod
    def _build_records(
            entities_names: List[str],
            embedded_result: Dict[str, Any],
            chunk_id: str,
            content: str,
            item_name: str,
    ) -> List[Dict[str, Any]]:
        """组装插入记录。"""

        # 1. 校验嵌入结果
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        # 2. 获取稠密向量和稀疏向量
        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")

        # 3. 校验向量是否存在
        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        # 4. 获取对应块的部分内容作为上下文
        context = content[:200]
        records: List[Dict] = []

        # 5. 遍历每一个实体名，构建记录
        for idx, entity_name in enumerate(entities_names):
            # 5.1 边界检查
            if idx >= len(dense_vector_list):
                break

            # 5.2 获取稠密向量
            dense = dense_vector_list[idx].tolist()

            # 5.3 解构稀疏向量（从 CSR 矩阵中提取当前实体的稀疏向量）
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]

            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()

            sparse_dict = dict(zip(indices, data))

            # 5.4 构建单条记录
            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "source_chunk_id": chunk_id,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }

            records.append(record)

        return records

class KnowledgeGraphNode(BaseNode):
    def __init__(self):
        super().__init__()
        config = get_config()
        self._milvus_obj = _MilvusEntityOperation(
            config.entity_name_collection
        )
        self._neo4j_obj = Neo4jGraphWriter(
            config.neo4j_database
        )

    def process(self, state:ImportGraphState):
        # 1. 参数校验
        chunks, item_name = self._validate_params(state)

        # 2 删除已经存在数据，milvus 和 neo4j数据
        print("步骤2 ： 删除已经存在数据")
        milvus_client = get_milvus_client()
        self._milvus_obj.clear(milvus_client, item_name)

        # 3. 数据处理
        # 3.1 把所有的chunks进行遍历，分别对每一个chunk处理
        # 3.2 结合提示词构建message送入大语言模型提取实体和关系
        self.execute_all_chunks(chunks)

    def _validate_params(self, state) -> List[Dict] or None:
        chunks = state.get("chunks")
        item_name = state.get("item_name")
        if not chunks:
            self.logger.info("chunks为空")
            return
        return chunks, item_name

    def execute_all_chunks(self, chunks: List[Dict]):
        for index, chunk in enumerate(chunks):
            # 进行数据校验，检验chunk是不是字典
            if not isinstance(chunk, dict):
                # 如果不是字典，跳过这个切片
                continue
            chunk_id = str(chunk.get("chunk_id"))
            item_name = chunk.get("item_name")
            content = chunk.get("content")

            # 调用方法，利用大语言模型进行处理，并且进行双写操作
            self._llm_call(chunk_id, item_name, content)

    def _llm_call(self, chunk_id, item_name, content):
        message = [
            SystemMessage(content=KNOWLEDGE_GRAPH_SYSTEM_PROMPT),
            HumanMessage(content=f"切片内容{content}")
        ]
        llm_client = get_llm_client()
        llm_response = llm_client.invoke(message)

        # 获取llm的返回结果
        llm_result = getattr(llm_response, "content", "").strip()# 去掉制表符等
        print("llm_result", llm_result)
        print("=="*50)

        # 清洗llm获得的结果
        graph_result = self._parser_and_clean(llm_result)

        entity = graph_result.get("entities")
        relations = graph_result.get("relations")

        # 写入milvus数据库
        milvus_client = get_milvus_client()
        self._milvus_obj.insert(milvus_client, entity, chunk_id, content, item_name)
        print("添加向量数据库完成")
        # 写入neo4j数据库
        neo4j_driver = get_neo4j_driver()
        self._neo4j_obj.insert(neo4j_driver, entity,
                               relations, chunk_id, item_name)

    def _parser_and_clean(self, llm_result):
        # 1.去掉json符号可能没有
        cleand = re.sub(r"^```(?:json)?\s*", "", llm_result.strip())
        cleand = re.sub(r"\s*```$", "", cleand)

        # 2.反序列化
        parsed_llm_response = json.loads(cleand)

        # 3.分别获得解析后的实体和关系列表
        parsed_entities = parsed_llm_response.get("entities",[])
        parsed_relations = parsed_llm_response.get("relations",[])

        # 4.调用方法清洗实体，清洗关系
        final_entities = self._clean_entities(parsed_entities)
        entities_name = {entity.get("name") for entity in final_entities}
        final_relations = self._clean_relations(entities_name, parsed_relations)

        # 需要返回的是字典形式
        final_data = {
            "entities": final_entities,
            "relations": final_relations,
        }
        return final_data

    def _clean_entities(self, parsed_entities):
        # 清洗掉没有名称的实体
        # 对名称过长的进行截取
        # 去掉不在实体标签白名单的实体
        # 去重
        unique_data = set() # 使用集合进行去重

        final_result = []
        for entity in parsed_entities:
            # 跳过实体名字为空的实体
            name = str(entity.get("name")).strip()
            if not name:
                continue
            # 对名称过长的进行截取
            if len(name) > MAX_ENTITY_NAME_LENGTH:
                name = name[:MAX_ENTITY_NAME_LENGTH]

            # 判断标签是否在白名单之中
            label = entity.get("label")
            if label not in ALLOWED_ENTITY_LABELS:
                continue

            # 去重
            unique_key = f"{name}_{label}" # 如果名字和标签都一样就认为相同
            if unique_key in unique_data:
                continue
            unique_data.add(unique_key)
            # 最终的数据
            clean_entities = {
                "name": name,
                "label": label,
            }
            description = entity.get("description")
            # 如果有description
            if description:
                clean_entities["description"] = description

            final_result.append(clean_entities)
        return final_result

    def _clean_relations(self, entities_name, parsed_relations):
        final_result = []

        for relation in parsed_relations:
            # 头和尾不能为空
            head = str(relation.get("head")).strip()
            tail = str(relation.get("tail")).strip()
            if not head or not tail:
                continue

            # 实体名称进行清洗
            if len(head) > MAX_ENTITY_NAME_LENGTH:
                head = head[:MAX_ENTITY_NAME_LENGTH]
            if len(tail) > MAX_ENTITY_NAME_LENGTH:
                tail = tail[:MAX_ENTITY_NAME_LENGTH]

            if head not in entities_name or tail not in entities_name:
                continue

            # type必须在白名单之中
            type = str(relation.get("type")).strip()
            if type not in ALLOWED_RELATION_TYPES:
                # 如果不在允许范围内附一个默认值
                type = DEFAULT_RELATION_TYPES

            # 构造最终数据
            clean_relation = {
                "head": head,
                "tail": tail,
                "type": type,
            }

            final_result.append(clean_relation)
        return final_result

def test_kg_extraction():
    """测试：模拟单个切片，跑通 LLM → 解析 → 清洗全流程。"""
    mock_state = {
        "item_name": "测试万用表",
        "chunks": [
            {
                "content": """# 电池安装
                    警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。
                    1. 把表笔与仪表断开。
                    2. 用螺丝刀拧开电池后盖上的螺母。
                    3. 正确安装电池，正负极应一致。
                    4. 盖上电池后盖并拧紧螺丝钉。
                    警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。
                    注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。""",
                "chunk_id": "18438591111",
                "item_name": "测试万用表",
            }
        ],
    }

    knowledge_graph_node = KnowledgeGraphNode()
    knowledge_graph_node.process(mock_state)

if __name__ == "__main__":
    test_kg_extraction()