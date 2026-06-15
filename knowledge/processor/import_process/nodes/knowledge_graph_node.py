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

# 实体名称最大长度 (旅游景点名字较长，放宽到30)
MAX_ENTITY_NAME_LENGTH = 30
# 旅游场景实体标签白名单
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Destination", "Attraction", "Hotel", "Restaurant",
    "Food", "Route", "Transport"
}
# 旅游场景关系类型白名单
ALLOWED_RELATION_TYPES: Set[str] = {
    "LOCATED_IN", "CONTAINS_ATTRACTION", "SERVES_FOOD",
    "NEARBY", "RECOMMENDED_HOTEL", "RECOMMENDED_FOOD", "RELATED_TO"
}
DEFAULT_RELATION_TYPES = "RELATED_TO"


class _MilvusEntityOperation:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    @staticmethod
    def _ensure_collection(client, collection_name):
        if client.has_collection(collection_name):
            return

        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=100)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        # 核心替换：使用 file_title 作为关联
        schema.add_field("file_title", DataType.VARCHAR, max_length=1000)

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

        client.create_collection(collection_name, schema=schema, index_params=index_params)

    def clear(self, milvus_client: MilvusClient, file_title: str):
        if not file_title:
            return
        collection_name = self.collection_name
        if milvus_client.has_collection(collection_name):
            milvus_client.delete(collection_name, filter=f'file_title=="{file_title}"')

    def insert(self, milvus_client: MilvusClient, entities: List, chunk_id: str,
               file_title: str, content: str):
        entities_names = list(dict.fromkeys(entity['name'] for entity in entities if entity.get('name')))

        if not entities_names:
            return

        self._ensure_collection(milvus_client, self.collection_name)
        bgem3_client = get_bgem3_client()
        embedding_result = bgem3_client.encode_documents(entities_names)

        records = self._build_records(entities_names, embedding_result, chunk_id, content, file_title)
        milvus_client.insert(collection_name=self.collection_name, data=records)

    @staticmethod
    def _build_records(
            entities_names: List[str],
            embedded_result: Dict[str, Any],
            chunk_id: str,
            content: str,
            file_title: str,
    ) -> List[Dict[str, Any]]:

        if not embedded_result:
            raise ValueError("嵌入结果为空")

        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")

        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        context = content[:200]
        records: List[Dict] = []

        for idx, entity_name in enumerate(entities_names):
            if idx >= len(dense_vector_list):
                break

            dense = dense_vector_list[idx].tolist()
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]

            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_dict = dict(zip(indices, data))

            record = {
                "entity_name": entity_name,
                "context": context,
                "file_title": file_title,
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
        self._milvus_obj = _MilvusEntityOperation(config.entity_name_collection)
        self._neo4j_obj = Neo4jGraphWriter(config.neo4j_database)

    def process(self, state: ImportGraphState):
        # 1. 参数校验
        chunks, file_title = self._validate_params(state)
        if not chunks or not file_title:
            return state

        # 2. 删除已经存在数据
        self.logger.info(f"正在清理旧的图谱实体数据 [{file_title}]...")
        milvus_client = get_milvus_client()
        self._milvus_obj.clear(milvus_client, file_title)

        # 3. 数据处理
        self.execute_all_chunks(chunks, file_title)

        state['status'] = 'graph_extracted'
        return state

    def _validate_params(self, state):
        chunks = state.get("chunks")
        file_title = state.get("file_title")
        if not chunks:
            self.logger.warning("chunks为空，跳过图谱提取")
        return chunks, file_title

    def execute_all_chunks(self, chunks: List[Dict], file_title: str):
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue

            # 使用前面Milvus上传节点赋予的 chunk_id，如果没有则用索引构造
            chunk_id = str(chunk.get("chunk_id", f"{file_title}_part_{index}"))
            content = chunk.get("content", "")

            if not content.strip():
                continue

            self._llm_call(chunk_id, file_title, content)

    def _llm_call(self, chunk_id, file_title, content):
        message = [
            SystemMessage(content=KNOWLEDGE_GRAPH_SYSTEM_PROMPT),
            HumanMessage(content=f"切片内容：\n{content}")
        ]
        llm_client = get_llm_client()

        try:
            llm_response = llm_client.invoke(message)
            llm_result = getattr(llm_response, "content", "").strip()

            self.logger.info(f"--- 知识图谱提取结果 (Chunk: {chunk_id}) ---\n{llm_result[:200]}...")

            graph_result = self._parser_and_clean(llm_result)
            entity = graph_result.get("entities", [])
            relations = graph_result.get("relations", [])

            if entity:
                # 写入 Milvus 实体库
                milvus_client = get_milvus_client()
                self._milvus_obj.insert(milvus_client, entity, chunk_id, file_title, content)

                # 写入 Neo4j 关系库
                neo4j_driver = get_neo4j_driver()
                self._neo4j_obj.insert(neo4j_driver, entity, relations, chunk_id, file_title)

        except Exception as e:
            self.logger.error(f"图谱提取发生异常 (Chunk: {chunk_id}): {e}")

    def _parser_and_clean(self, llm_result):
        content = re.sub(r"^```json\s*", "", llm_result, flags=re.IGNORECASE)
        cleand = re.sub(r"\s*```$", "", content)

        try:
            parsed_llm_response = json.loads(cleand)
        except Exception:
            return {"entities": [], "relations": []}

        parsed_entities = parsed_llm_response.get("entities", [])
        parsed_relations = parsed_llm_response.get("relations", [])

        final_entities = self._clean_entities(parsed_entities)
        entities_name = {entity.get("name") for entity in final_entities}
        final_relations = self._clean_relations(entities_name, parsed_relations)

        return {"entities": final_entities, "relations": final_relations}


    def _clean_entities(self, parsed_entities: List[Dict]):
        unique_data = set()
        final_result = []

        for entity in parsed_entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue

            if len(name) > MAX_ENTITY_NAME_LENGTH:
                name = name[:MAX_ENTITY_NAME_LENGTH]

            label = entity.get("label")
            if label not in ALLOWED_ENTITY_LABELS:
                continue

            unique_key = f"{name}_{label}"
            if unique_key in unique_data:
                continue

            unique_data.add(unique_key)

            clean_entities = {"name": name, "label": label}
            description = entity.get("description")
            if description:
                clean_entities["description"] = description

            final_result.append(clean_entities)

        return final_result


    def _clean_relations(self, entities_name: Set[str], parsed_relations: List[Dict]):
        final_result = []

        for relation in parsed_relations:
            head = str(relation.get("head", "")).strip()
            tail = str(relation.get("tail", "")).strip()
            if not head or not tail:
                continue

            if len(head) > MAX_ENTITY_NAME_LENGTH:
                head = head[:MAX_ENTITY_NAME_LENGTH]
            if len(tail) > MAX_ENTITY_NAME_LENGTH:
                tail = tail[:MAX_ENTITY_NAME_LENGTH]

            if head not in entities_name or tail not in entities_name:
                continue

            type_val = str(relation.get("type", "")).strip()
            if type_val not in ALLOWED_RELATION_TYPES:
                type_val = DEFAULT_RELATION_TYPES

            clean_relation = {
                "head": head,
                "tail": tail,
                "type": type_val,
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