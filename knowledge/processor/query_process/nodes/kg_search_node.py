"""
node_query_kg — 旅游知识图谱查询节点。

类结构（与导入侧 kg_graph_node.py 的 Writer 模式对称）:
─────────────────────────────────────────────────────────
  _EntityExtractor    LLM 实体抽取 + JSON 解析
  _EntityAligner      Milvus 实体对齐 (Entity Name Collection)
  _Neo4jGraphReader   Neo4j 种子节点 / 一跳关系 / chunk 反查
  _ChunkBackfiller    Milvus Chunk 回填
  KnowledgeGraphSearchNode 主编排器
─────────────────────────────────────────────────────────
"""
import logging
import re
import json
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple, Union

from pymilvus import MilvusClient
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.milvus_client_util import (
    get_milvus_client,
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    fetch_chunks_by_chunk_ids
)
from knowledge.prompts.query.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.neo4j_util import get_neo4j_driver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# 旅游场景常量
# -------------------------------------------------
# 旅游实体名字可能较长（如：三亚亚特兰蒂斯酒店海底餐厅）
_ENTITY_NAME_MAX_LENGTH = 30
_DEFAULT_ENTITY_NAME_ALIGN = 0.5

# -------------------------------------------------
# Neo4J 类型定义
# -------------------------------------------------
DocEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]
OneHopRelation = Dict[str, Any]

# -------------------------------------------------
# Neo4j Cypher 语句 (修正：底层数据库属性仍为 item_name)
# -------------------------------------------------
_CYPHER_EXACT_SEEDS = """
MATCH (n:Entity)
WHERE n.item_name=$file_title AND n.name=$name
RETURN n.item_name as file_title, n.name as name
LIMIT 1
"""

_CYPHER_FUZZY_SEEDS = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($name)
      AND n.item_name = $file_title
RETURN n.name AS name, n.item_name AS file_title
LIMIT $limit
"""

_CYPHER_ONE_HOP_RELATIONS = """
MATCH (seed:Entity {name:$name, item_name:$file_title})-[r]-(nbr:Entity)
WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name=$file_title
RETURN 
  CASE WHEN startNode(r)=seed THEN seed.name ELSE nbr.name END AS head,
  type(r) as rel,
  CASE WHEN startNode(r)=seed THEN nbr.name ELSE seed.name END AS tail
LIMIT $limit
"""

_CYPHER_LOOKUP_CHUNK = """
UNWIND $weighted_nodes as n
MATCH (e:Entity{name:n.entity_name, item_name:n.file_title})-[r:MENTIONED_IN]->(c:Chunk{item_name:n.file_title})
WITH c, sum(n.weight) AS score, count(e) AS cnt
RETURN c.id AS chunk_id, c.item_name AS file_title, score, cnt
ORDER BY score DESC, cnt DESC, chunk_id DESC
LIMIT $limit
"""

SEED_NODE_WEIGHT = 2.0
NER_NODE_WEIGHT = 1.0


# -------------------------------------------------
# 工具函数
# -------------------------------------------------
def _clean_parse_llm_content(llm_response_content: str) -> List[str]:
    """清洗以及解析LLM输出"""
    if not llm_response_content:
        return []

    text = re.sub(r"^json?\s*", "", llm_response_content.strip(), flags=re.IGNORECASE)
    re_sub = re.sub(r"\s*```$", "", text)

    try:
        deserialized_result: Dict[str, Any] = json.loads(re_sub)
    except JSONDecodeError as e:
        logger.error(f"JSON 反序列失败，原因: {str(e)}")
        return []

    entities_name = deserialized_result.get('entities', [])
    if not entities_name or not isinstance(entities_name, list):
        return []

    seen = set()
    entities_name_result = []

    for entity_name in entities_name:
        if not entity_name or not isinstance(entity_name, str):
            continue
        truncated_entity_name = truncate_entity_name_length(entity_name)
        if truncated_entity_name not in seen:
            seen.add(truncated_entity_name)
            entities_name_result.append(truncated_entity_name)

    return entities_name_result


def truncate_entity_name_length(entity_name: str) -> str:
    name = entity_name.strip()
    return name[:_ENTITY_NAME_MAX_LENGTH] if len(name) > _ENTITY_NAME_MAX_LENGTH else name


def _file_title_filter_expr(file_titles: List[str]) -> str:
    if not file_titles:
        return ""
    quoted = ", ".join(f"'{title}'" for title in file_titles)
    return f"file_title in [{quoted}]"


def _clean_seed_rows(rows: List[Dict[str, Any]]) -> List[EntitySeedNode]:
    """清洗查询种子节点的数据"""
    if not rows:
        return []

    clean_seeds_result: List[EntitySeedNode] = []
    for row in rows:
        file_title = row.get('file_title', '').strip()
        entity_name = row.get('name', '').strip()
        if not file_title or not entity_name:
            continue
        clean_seeds_result.append({
            "file_title": file_title,
            "entity_name": entity_name
        })
    return clean_seeds_result


def _one_hop_relations_to_texts(triples: List[OneHopRelation]) -> List[str]:
    if not triples:
        return []
    docs: List[str] = []
    for tr in triples:
        ft = (tr.get("file_title") or "").strip()
        h = (tr.get("head") or "").strip()
        r = (tr.get("rel") or "").strip()
        t = (tr.get("tail") or "").strip()
        if not (h and r and t):
            continue
        docs.append(f"[{ft}] {h} -({r})-> {t}" if ft else f"{h} -({r})-> {t}")
    return docs


class _EntityExtractor:
    """实体提取器：利用LLM从查询问题中提取实体"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def extract(self, user_query: str) -> List[str]:
        llm_client = get_llm_client(response_format=True)
        if llm_client is None:
            return []

        # 获取提示词 (需确保 query_prompt.py 中有此常量)
        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(
            MAX_ENTITY_NAME_LENGTH=_ENTITY_NAME_MAX_LENGTH
        )

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=entities_name_extract_system_prompt),
                HumanMessage(content=f"用户问题:{user_query}")
            ])
            llm_response_content = getattr(llm_response, 'content', "").strip()
            print(llm_response_content)
            return _clean_parse_llm_content(llm_response_content)
        except Exception as e:
            self._logger.error(f"LLM 实体抽取调用失败: {str(e)}")
            return []


class _EntityAligner:
    """实体对齐器：根据LLM提取到的实体名查询Milvus进行纠错对齐"""

    def __init__(self, collection_name: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name

    def align(self, entity_names: List[str], confirmed_docs: List[str]) -> Dict[str, Any]:
        fallback_result = {"entities_aligned_name": [], "entities_aligned_elements": []}
        if not entity_names:
            return fallback_result

        embedding_model = get_bgem3_client()
        milvus_client = get_milvus_client()
        if not embedding_model or not milvus_client:
            return fallback_result

        embedding_result = generate_hybrid_embeddings(embedding_model=embedding_model, embedding_documents=entity_names)
        if not embedding_result:
            return fallback_result

        embedding_result_dense = embedding_result['dense']
        embedding_result_sparse = embedding_result['sparse']

        # 弹性过滤：如果上游锁定了目标文档，则限定范围对齐；否则全局对齐
        file_title_expr = _file_title_filter_expr(confirmed_docs) if confirmed_docs else None

        aligned_entities_name: List[str] = []
        aligned_entity_elements: List[Dict[str, Any]] = []
        seen = set()

        for index, entity_name in enumerate(entity_names):
            align_one_result: List[Dict[str, Any]] = self._align_one(
                milvus_client, self._collection_name, file_title_expr,
                embedding_result_dense, embedding_result_sparse, index, entity_name
            )
            aligned_entity_elements.extend(align_one_result)

            for detail in align_one_result:
                aligned_name = detail.get("aligned")
                file_title = detail.get("file_title")
                if aligned_name:
                    key = (file_title, aligned_name)
                    if key not in seen:
                        seen.add(key)
                        aligned_entities_name.append(aligned_name)

        self._logger.info(f"对齐后的实体个数 {len(aligned_entities_name)} 实体的名字：{aligned_entities_name}")

        return {
            "entities_aligned_name": aligned_entities_name,
            "entities_aligned_elements": aligned_entity_elements
        }

    def _align_one(self, milvus_client: MilvusClient, _collection_name: str,
                   file_title_expr: str, embedding_result_dense: List, embedding_result_sparse: List,
                   index: int, entity_name: str) -> List[Dict[str, Any]]:

        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]

        if not dense_vector or not sparse_vector:
            return [{"original": entity_name, "aligned": "", "context": "", "reason": "vector values not exist"}]

        # 1. 第一次尝试：带上级锁定的文档范围进行精确检索
        hybrid_search_requests = create_hybrid_search_requests(
            dense_vector=dense_vector, sparse_vector=sparse_vector, expr=file_title_expr, limit=5
        )

        reps = execute_hybrid_search_query(
            milvus_client=milvus_client, collection_name=_collection_name,
            search_requests=hybrid_search_requests, ranker_weights=(0.4, 0.6), norm_score=True, limit=5,
            output_fields=["source_chunk_id", "file_title", "context", "entity_name"]
        )

        hits = reps[0] if reps else []

        # === ✨ 新增核心逻辑：无缝全局兜底 ✨ ===
        # 如果带 file_title 限制没查到，而且确实传了 expr，我们就去掉 expr 全局查一次！
        if not hits and file_title_expr:
            self._logger.warning(f"在限定文档 [{file_title_expr}] 下未找到实体 '{entity_name}'，降级为全局图谱检索...")
            fallback_requests = create_hybrid_search_requests(
                dense_vector=dense_vector, sparse_vector=sparse_vector, expr=None, limit=5
            )
            fallback_reps = execute_hybrid_search_query(
                milvus_client=milvus_client, collection_name=_collection_name,
                search_requests=fallback_requests, ranker_weights=(0.4, 0.6), norm_score=True, limit=5,
                output_fields=["source_chunk_id", "file_title", "context", "entity_name"]
            )
            hits = fallback_reps[0] if fallback_reps else []
        # ==================================

        if not hits:
            return [{"original": entity_name, "aligned": "", "score": "", "reason": "no_hit"}]

        best_by_title: Dict[str, Dict] = {}
        for hit in hits:
            entity = hit.get("entity")
            file_title = entity.get("file_title", "").strip()
            if file_title not in best_by_title:
                best_by_title[file_title] = hit

        if not best_by_title:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "no_valid_file_title"}]

        results: List[Dict[str, Any]] = []
        for file_title, best in best_by_title.items():
            score = best.get("distance")
            if float(score) < float(_DEFAULT_ENTITY_NAME_ALIGN):
                continue
            ent = best.get("entity")
            results.append({
                "original": entity_name,
                "aligned": ent.get("entity_name"),
                "score": score,
                "file_title": file_title,
                "source_chunk_id": ent.get("source_chunk_id"),
                "reason": "top1_per_doc",
            })

        if not results:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "all_below_threshold"}]

        return results


class _Neo4jGraphReader:
    """职责：所有对Neo4j的读操作"""

    def __init__(self, database: str, kg_max_seed_candidates: int, kg_max_total_seeds: int,
                 kg_max_triples_per_seed: int, kg_max_total_triples: int, kg_max_total_chunks: int):
        self._database = database
        self._kg_max_seed_candidates = kg_max_seed_candidates
        self._kg_max_total_seeds = kg_max_total_seeds
        self.kg_max_triples_per_seed = kg_max_triples_per_seed
        self._kg_max_total_triples = kg_max_total_triples
        self._kg_max_total_chunks = kg_max_total_chunks
        self._logger = logging.getLogger(self.__class__.__name__)

    def _session(self):
        neo4j_driver = get_neo4j_driver()
        if neo4j_driver is None:
            raise RuntimeError("Neo4J驱动获取失败")
        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self, pairs: List[DocEntityPair]) -> List[EntitySeedNode]:
        if not pairs:
            return []

        final_seeds_result: List[EntitySeedNode] = []
        for pair in pairs:
            file_title = pair.get('file_title', '').strip()
            entity_name = pair.get('entity_name', '').strip()
            if not file_title or not entity_name:
                continue
            try:
                with self._session() as session:
                    candidates_seed_nodes = self._execute_seed_nodes(session, file_title, entity_name,
                                                                     self._kg_max_seed_candidates)
                    final_seeds_result.extend(candidates_seed_nodes)
                    if len(final_seeds_result) > self._kg_max_total_seeds:
                        final_seeds_result = final_seeds_result[:self._kg_max_total_seeds]
                        break
            except Exception as e:
                self._logger.error(f"获取种子节点失败,原因 :{str(e)}")

        self._logger.info(f"获取种子节点 {len(final_seeds_result)} 个")
        return final_seeds_result

    def _execute_seed_nodes(self, session, file_title: str, entity_name: str, _kg_max_seed_candidates: int) -> List[
        EntitySeedNode]:
        exact_rows = session.execute_read(
            lambda tx: tx.run(_CYPHER_EXACT_SEEDS, file_title=file_title, name=entity_name).data()
        )
        if exact_rows:
            return _clean_seed_rows(exact_rows)

        fuzzy_rows = session.execute_read(
            lambda tx: tx.run(_CYPHER_FUZZY_SEEDS, file_title=file_title, name=entity_name,
                              limit=_kg_max_seed_candidates).data()
        )
        return _clean_seed_rows(fuzzy_rows)

    def find_one_hop_relations(self, seed_nodes: List[EntitySeedNode]) -> List[OneHopRelation]:
        if not seed_nodes:
            return []
        seen = set()
        one_hop_relations_final_result = []

        for seed_node in seed_nodes:
            file_title = seed_node.get('file_title', "")
            seed_name = seed_node.get('entity_name', "")
            if not file_title or not seed_name:
                continue

            try:
                with self._session() as session:
                    seed_one_hop_relations = self._execute_one_hop_relations(session, file_title, seed_name,
                                                                             self.kg_max_triples_per_seed)
                    if not seed_one_hop_relations:
                        continue

                    for relation in seed_one_hop_relations:
                        head = relation.get('head')
                        rel = relation.get('rel')
                        tail = relation.get('tail')

                        key = (file_title, head, rel, tail)
                        if key not in seen:
                            seen.add(key)
                            one_hop_relations_final_result.append(relation)

                    if len(one_hop_relations_final_result) > self._kg_max_total_triples:
                        one_hop_relations_final_result = one_hop_relations_final_result[:self._kg_max_total_triples]
                        break
            except Exception as e:
                self._logger.error(f"查询 {seed_name} 一跳关系失败: {str(e)}")
                return []

        return one_hop_relations_final_result

    def _execute_one_hop_relations(self, session, file_title: str, seed_name: str, kg_max_triples_per_seed: int) -> \
    List[OneHopRelation]:
        one_hop_relations = session.execute_read(
            lambda tx: tx.run(_CYPHER_ONE_HOP_RELATIONS, file_title=file_title, name=seed_name,
                              limit=kg_max_triples_per_seed).data()
        )
        if not one_hop_relations:
            return []

        result = []
        for relation in one_hop_relations:
            head = relation.get('head', '').strip()
            rel = relation.get('rel', '').strip()
            tail = relation.get('tail', '').strip()

            if not (head and rel and tail):
                continue
            result.append({
                "head": head,
                "rel": rel,
                "tail": tail,
                "file_title": file_title
            })
        return result

    def collect_node_weight(self, seed_nodes: List[EntitySeedNode], one_hop_relations: List[OneHopRelation]) -> List[
        Dict[str, Any]]:
        if not seed_nodes or not one_hop_relations:
            return []

        weight_map = {}
        seen = set()
        for seed_node in seed_nodes:
            file_title = seed_node.get('file_title')
            seed_name = seed_node.get('entity_name')
            key = (file_title, seed_name)
            if key not in seen:
                seen.add(key)
                weight_map[key] = SEED_NODE_WEIGHT

        for relation in one_hop_relations:
            head = relation.get('head')
            tail = relation.get('tail')
            file_title = relation.get('file_title')

            if head and (file_title, head) not in weight_map:
                weight_map[(file_title, head)] = NER_NODE_WEIGHT
            if tail and (file_title, tail) not in weight_map:
                weight_map[(file_title, tail)] = NER_NODE_WEIGHT

        return [{"file_title": ft, "entity_name": en, "weight": w} for (ft, en), w in weight_map.items()]

    def find_nodes_chunk_id(self, weighted_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            with self._session() as session:
                sorted_node_chunk_id = session.execute_read(lambda tx: tx.run(
                    _CYPHER_LOOKUP_CHUNK, weighted_nodes=weighted_nodes, limit=self._kg_max_total_chunks
                ).data())
        except Exception as e:
            self._logger.error(f"反查chunk_id失败,原因:{str(e)}")
            return []

        hits = []
        for chunk_row in sorted_node_chunk_id:
            chunk_id = chunk_row.get('chunk_id', "").strip()
            file_title = chunk_row.get('file_title', "").strip()
            score = chunk_row.get('score')

            if chunk_id and file_title:
                hits.append({
                    "id": None,
                    "distance": float(score or 0.0),
                    "entity": {"chunk_id": str(chunk_id), "file_title": str(file_title)}
                })
        return hits


def _build_item_entity_pairs(aligned_entities_info: List[Dict[str, Any]]) -> List[DocEntityPair]:
    """从对齐后的实体详情中获取 file_title + entity_name 的 pair 对"""
    if not aligned_entities_info:
        return []

    seen = set()
    pairs = []
    for info in aligned_entities_info:
        file_title = info.get('file_title', "").strip()
        aligned_name = info.get('aligned', "").strip()
        if not (file_title and aligned_name):
            continue
        key = (file_title, aligned_name)
        if key not in seen:
            seen.add(key)
            pairs.append({"file_title": file_title, "entity_name": aligned_name})
    return pairs


class _ChunkBackFiller:
    """职责：根据 chunk_ids 查询 Milvus 获取切片原文"""

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def back_fill(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunk_nodes_sorted:
            return []

        chunk_ids: List[Union[str, int]] = self._collect_chunk_ids(chunk_nodes_sorted)

        try:
            chunks: List[Dict[str, Any]] = fetch_chunks_by_chunk_ids(
                collection_name=self._collection_name,
                chunk_ids=chunk_ids,
                output_fields=['chunk_id', 'content', 'title', 'file_title'],
                batch_size=30
            )
            if not chunks:
                return []
        except Exception as e:
            self.logger.error(f"批量反查chunk对象失败：{str(e)}")
            return []

        chunk_id_map = {str(chunk.get('chunk_id')): chunk for chunk in chunks if chunk.get('chunk_id') is not None}
        return [{"entity": chunk_id_map.get(str(chunk_id))} for chunk_id in chunk_ids]

    def _collect_chunk_ids(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Union[str, int]]:
        chunk_ids = []
        for node in chunk_nodes_sorted:
            if not node: continue
            entity = node.get('entity', '')
            if not entity: continue
            chunk_id = entity.get('chunk_id')
            if not chunk_id: continue

            try:
                chunk_ids.append(int(chunk_id))
            except (ValueError, TypeError):
                chunk_ids.append(str(chunk_id))
        return chunk_ids


class KnowledgeGraphSearchNode(BaseNode):
    """
    知识图谱查询主编排器。
    Pipeline: 抽取实体 ──▶ 对齐实体 ──▶ Neo4j查询 ──▶ 回填chunk
    """
    name = "kg_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        validated_query, confirmed_docs = self._validate_inputs(state)

        # 如果既没有确认的文档约束，由于图谱检索具有很强的定向性，我们依然可以依赖 Aligner 的全局匹配
        kg_result: Dict[str, Any] = self._run_pipeline(validated_query, confirmed_docs)

        return {
            "kg_chunks": kg_result.get('kg_chunks', []),
            "kg_triples": kg_result.get('kg_triples', [])
        }

    def _validate_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        rewritten_query = state.get('rewritten_query', "")
        # 获取意图阶段确认的目标文档名列表
        confirmed_docs = state.get('confirmed_docs', [])

        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        user_query = rewritten_query
        for name in confirmed_docs:
            if not name:
                continue
            pattern = r"\s*".join(re.escape(ch) for ch in name.replace(" ", ""))
            user_query = re.sub(pattern, "", user_query, flags=re.IGNORECASE)

        user_query = " ".join(user_query.split()).strip()
        return user_query, confirmed_docs

    def _run_pipeline(self, validated_query: str, confirmed_docs: List[str]) -> Dict[str, Any]:
        entity_extractor = _EntityExtractor()
        entity_aligner = _EntityAligner(collection_name=self.config.entity_name_collection)
        neo4g_graph_reader = _Neo4jGraphReader(
            database=self.config.neo4j_database,
            kg_max_seed_candidates=self.config.kg_max_seed_candidates,
            kg_max_total_seeds=self.config.kg_max_total_seeds,
            kg_max_triples_per_seed=self.config.kg_max_triples_per_seed,
            kg_max_total_triples=self.config.kg_max_total_triples,
            kg_max_total_chunks=self.config.kg_max_total_chunks
        )
        chunk_back_filler = _ChunkBackFiller(collection_name=self.config.chunks_collection)

        # 1. 提取与对齐
        entities_name = entity_extractor.extract(user_query=validated_query)
        entities_name_aligned = entity_aligner.align(entities_name, confirmed_docs=confirmed_docs)

        aligned_entities_info = entities_name_aligned.get('entities_aligned_elements')
        item_entity_pairs = _build_item_entity_pairs(aligned_entities_info)

        # 2. Neo4J 关系拉取
        seed_nodes = neo4g_graph_reader.find_seed_nodes(item_entity_pairs)
        one_hop_relations = neo4g_graph_reader.find_one_hop_relations(seed_nodes)
        weighted_nodes = neo4g_graph_reader.collect_node_weight(seed_nodes, one_hop_relations)
        chunk_nodes_sorted = neo4g_graph_reader.find_nodes_chunk_id(weighted_nodes)

        # 3. Milvus Chunk 回填
        kg_chunks = chunk_back_filler.back_fill(chunk_nodes_sorted)
        triples_docs = _one_hop_relations_to_texts(one_hop_relations)

        return {
            "kg_chunks": kg_chunks,
            "kg_triples": triples_docs,
        }

if __name__ == '__main__':

    # 旅游知识图谱检索本地测试
    kg_search_node = KnowledgeGraphSearchNode()

    # 模拟上一层路由与状态传递过来的数据
    mock_state = {
        "rewritten_query": "杭州去西湖路线？",
        "confirmed_docs": ["杭州-交通指南"]
    }

    print("开始图谱检索测试...")
    result = kg_search_node.process(mock_state)

    print("\n=== 图谱检索结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))