import os
from neo4j import GraphDatabase
import logging
from knowledge.processor.import_process.exceptions import Neo4jError

logger = logging.getLogger(__name__)
_neo4j_driver = None

# ------------------------------------------
# Neo4J的Cypher语句
# ------------------------------------------
# Chunk标签节点创建
CYPHER_MERGE_CHUNK = """
    MERGE (c:Chunk {id: $chunk_id, item_name: $item_name})
"""

# Entity标签节点的创建
CYPHER_MERGE_ENTITY_TEMPLATE = """
    MERGE (n:Entity {{name: $name, item_name: $item_name}})
    ON CREATE SET
        n.source_chunk_id = $chunk_id,
        n.description     = $description
    ON MATCH SET
        n.description = CASE
            WHEN $description <> "" THEN $description
            ELSE coalesce(n.description, "")
        END
    SET n:`{label}`
"""
# Entity关联Chunk
CYPHER_LINK_ENTITY_TO_CHUNK = """
    MATCH (n:Entity {name: $name, item_name: $item_name})
    MATCH (c:Chunk  {id: $chunk_id, item_name: $item_name})
    MERGE (n)-[:MENTIONED_IN]->(c)
"""

# Entity与Entity的关系
CYPHER_MERGE_RELATION_TEMPLATE = """
    MATCH (h:Entity {{name: $head, item_name: $item_name}})
    MATCH (t:Entity {{name: $tail, item_name: $item_name}})
    MERGE (h)-[:{rel_type}]->(t)
"""

# 清理Neo4J数据
CYPHER_CLEAR_ITEM = """
    MATCH (n {item_name: $item_name}) DETACH DELETE n
"""

######### Neo4j操作类 #########
class Neo4jGraphWriter:
    def __init__(self, database: str = ""):
        self._database = database
        self._logger = logging.getLogger(self.__class__.__name__)

    def clear(self, neo4j_driver, item_name: str) -> None:
        if not neo4j_driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(neo4j_driver) as session:
                session.execute_write(
                    lambda tx, name: tx.run(CYPHER_CLEAR_ITEM, item_name=name),
                    item_name,
                )
        except Exception as e:
            raise Neo4jError(f"Neo4j 清理失败: {e}")

    def insert(self, driver, entities, relations, chunk_id, item_name):
        """
        Neo4J的写入

        Args:
            driver: neo4j的驱动
            entities:  清洗后的实体
            relations: 清洗后的关系链
            chunk_id:  实体对应的chunk_id
            item_name: 文档对应LLM提取的商品名

        Returns:

        """
        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体列表为空")

        # 2.  判断驱动
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    self._write_graph_tx, entities, relations, chunk_id, item_name,
                )
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}")

    def _write_graph_tx(self, tx, entities, relations, chunk_id, item_name):

        # 1. 创建 Chunk 节点
        tx.run(CYPHER_MERGE_CHUNK, chunk_id=chunk_id, item_name=item_name)

        # 2. 创建实体节点 + 关联到 Chunk
        for entity in entities:
            name = entity.get("name")
            raw_label = entity.get("label")
            description = entity.get("description")

            # 动态格式化 Cypher，将安全标签注入(TODO )
            cypher_query = CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)

            tx.run(cypher_query, name=name, description=description,
                   chunk_id=chunk_id, item_name=item_name)

            # 关联实体到 Chunk
            tx.run(CYPHER_LINK_ENTITY_TO_CHUNK,
                   name=name, chunk_id=chunk_id, item_name=item_name)

        # 3. 创建实体间关系
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            rel_type = rel.get("type")

            cypher = CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher, head=head, tail=tail, item_name=item_name)

    def _session(self, driver):
        return driver.session(database=self._database)

#########2 获取 Neo4j 驱动实例 #########
def get_neo4j_driver() -> GraphDatabase:

    global _neo4j_driver
    try:
        if _neo4j_driver is None:
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            _neo4j_driver = GraphDatabase.driver(
                uri=uri,
                auth=(username, password)
            )
            # Neo4j 驱动默认是懒加载，这行代码能确保如果账号密码错误或网络不通，当场就会抛出异常，而不是等到插入数据时才报错。
            _neo4j_driver.verify_connectivity()
        return _neo4j_driver

    except Exception as e:
        # exc_info=True 会在日志中打印出完整的 Error Traceback 堆栈，方便排查到底是密码错还是网络不通
        logger.error(f"初始化 Neo4j 驱动失败: {e}", exc_info=True)
        return None