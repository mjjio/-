from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.milvus_client_util import (
    get_milvus_client,
    create_hybrid_search_requests,
    execute_hybrid_search_query
)


class VectorSearchNode(BaseNode):
    """
    三路召回之：向量混合检索召回节点
    基于大语言模型重写后的查询语句，生成稠密与稀疏向量，并在切片库中进行检索。
    """

    def process(self, state):
        # 1. 参数校验：获取重写后的查询，以及上一环节确定的目标文档
        rewritten_query, confirmed_docs = self._validate_params(state)

        # 2. 对重写后的查询问题进行向量化
        bge_m3_client = get_bgem3_client()
        query_embedded_vector = generate_hybrid_embeddings(bge_m3_client, [rewritten_query])

        # 异常判断
        if not query_embedded_vector or not query_embedded_vector.get("dense"):
            self.logger.warning("向量化失败，跳过向量召回")
            return {"embedding_chunks": []}

        # 3. 执行向量检索
        searched_result = self._search_vectors(query_embedded_vector, confirmed_docs)

        if not searched_result:
            self.logger.info("未检索到相关的切片信息")
            return {"embedding_chunks": []}

        # 4. 更新状态
        state["embedding_chunks"] = searched_result
        return {"embedding_chunks": searched_result}

    def _validate_params(self, state):
        rewritten_query = state.get("rewritten_query")
        # confirmed_docs 是我们在 IntentConfirmNode 中确定的 file_title 列表
        confirmed_docs = state.get("confirmed_docs", [])

        if not rewritten_query:
            raise ValueError("rewritten_query (重写后的查询) 不存在，无法进行向量检索")

        return rewritten_query, confirmed_docs

    def _search_vectors(self, query_embedded_vector, confirmed_docs):
        milvus_client = get_milvus_client()
        config = get_config()
        collection_name = getattr(config, 'chunks_collection', 'tourism_local_chunks_v1')

        expr = None
        if confirmed_docs:
            expr = self._create_search_file_title_expr(confirmed_docs)

        search_requests = create_hybrid_search_requests(
            dense_vector=query_embedded_vector["dense"][0],
            sparse_vector=query_embedded_vector["sparse"][0],
            expr=expr,
            limit=5
        )

        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=collection_name,
            search_requests=search_requests,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            limit=5,
            output_fields=["chunk_id", "content", "file_title"],
        )

        hits = res[0] if res and len(res) > 0 else []

        # === ✨ 新增核心逻辑：无缝全局兜底 ✨ ===
        if not hits and expr:
            self.logger.warning(f"在限定文档 [{expr}] 下未找到切片，触发全局向量兜底搜索...")
            fallback_requests = create_hybrid_search_requests(
                dense_vector=query_embedded_vector["dense"][0],
                sparse_vector=query_embedded_vector["sparse"][0],
                expr=None,  # 去除标量限制
                limit=5
            )
            fallback_res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=collection_name,
                search_requests=fallback_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=5,
                output_fields=["chunk_id", "content", "file_title"],
            )
            hits = fallback_res[0] if fallback_res and len(fallback_res) > 0 else []
        # ==================================

        return hits

    def _create_search_file_title_expr(self, confirmed_docs):
        """
        构建标量字段搜索条件: file_title in ["成都指南", "三亚攻略"]
        """
        condition = ", ".join([f'"{name}"' for name in confirmed_docs])
        return f"file_title in [{condition}]"

if __name__ == "__main__":
    state = {
            "destinations": ['三亚'],
            "intents": ['交通指南'],
            "rewritten_query": "如何从前往亚龙湾？",
            "confirmed_docs": [],
    }

    vector_search_node = VectorSearchNode()
    result = vector_search_node.process(state)
    print(result)
