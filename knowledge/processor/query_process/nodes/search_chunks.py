
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.milvus_client_util import get_milvus_client, create_hybrid_search_requests, \
    execute_hybrid_search_query


# 向量检索节点
class VectorSearchNode(BaseNode):
    def process(self, state):
        # 根据item_name,rewritten_query查询向量数据库之中的chunks
        # 1.参数校验
        item_name, rewritten_query = self._validate_params(state)
        # 2.对重写问题向量化
        bge_m3_client = get_bgem3_client()
        # {
        #     "dense": [den.tolist() for den in embedding_result["dense"]],
        #     "sparse": processed_sparse_result
        # }
        query_embedded_vector = generate_hybrid_embeddings(bge_m3_client, [rewritten_query])
        # 进行非空判断
        if not query_embedded_vector:
            return state

        # 如果非空，进行向量检索
        searched_result = self._search_vectors(item_name, query_embedded_vector)
        if not searched_result:
            return state
        return {"embedding_chunks": searched_result}

    def _validate_params(self, state):
        item_name = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not item_name:
            raise ValueError("item_names is required")
        if not rewritten_query:
            raise ValueError("rewritten_query is required")
        return item_name, rewritten_query

    def _search_vectors(self, item_name, query_embedded_vector):
        # 创建milvus链接对象
        milvus_client = get_milvus_client()

        # 构建搜索条件
        # 构建标量字段搜索条件 item_name in ["xxx","yy"]
        search_condition = self._create_search_item_name(item_name)
        # 构建混合搜索条件
        search_requests = create_hybrid_search_requests(
            dense_vector=query_embedded_vector["dense"][0],
            sparse_vector=query_embedded_vector["sparse"][0],
            expr=search_condition,
            limit=5
        )
        config = get_config()
        # 执行搜索
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name = config.chunks_collection,
            search_requests=search_requests,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            limit=5,
            output_fields=["chunk_id","content","item_name"],
        )
        if not res or not res[0]:
            return []
        return res[0]

    # 构建标量字段搜索条件 item_name in ["xxx","yy"]
    def _create_search_item_name(self, item_name):
        condition = ", ".join([f'"{name}"' for name in item_name])
        return f" item_name in [{condition}]"# 前面有一个空格

if __name__ == "__main__":
    state = {
        "rewritten_query":"关于H3C LA2608，如何使用？",
        "item_names":["H3C LA2608 室内无线网关"],
    }

    vector_search_node = VectorSearchNode()
    result = vector_search_node.process(state)
    print(result)
