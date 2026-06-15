from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    get_milvus_client
)


class HydeSearchNode(BaseNode):
    """
    假设性文档嵌入(HyDE)搜索节点：
    先利用大模型针对用户问题生成一段“完美的旅游攻略回答”，
    再将原问题与该回答拼接，利用更丰富的语义特征去向量库中召回相似切片。
    """

    def process(self, state):

        # 1.参数校验，获取锁定的文档、目的地和重写后的问题
        confirmed_docs, destinations, rewritten_query = self._validate_params(state)

        # 2.调用大语言模型，生成假设性旅游攻略
        response = self._call_llm(confirmed_docs, destinations, rewritten_query)
        self.logger.info(f"HyDE生成的假设性文档片段: {response[:50]}...")

        # 3.拼接字符串做向量搜索 (利用原问题+完美回答来扩充特征)
        embedding_content = f"{rewritten_query}\n{response}"

        # 4.做向量化
        bge_m3_client = get_bgem3_client()
        embedding_vectors = generate_hybrid_embeddings(bge_m3_client, [embedding_content])

        # 异常判断
        if not embedding_vectors or not embedding_vectors.get("dense"):
            self.logger.warning("HyDE向量化失败，跳过该路召回")
            return {"hyde_embedding_chunks": []}

        # 5.构建联合搜索条件
        expr = None
        if confirmed_docs:
            expr = self._create_search_file_title_expr(confirmed_docs)

        search_requests = create_hybrid_search_requests(
            dense_vector=embedding_vectors["dense"][0],
            sparse_vector=embedding_vectors["sparse"][0],
            expr=expr,
        )

        # 6.执行搜索
        config = get_config()
        milvus_client = get_milvus_client()
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=config.chunks_collection,
            search_requests=search_requests,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            limit=5,
            output_fields=["chunk_id", "content", "file_title"],
        )

        if not res or not res[0]:
            return {"hyde_embedding_chunks": []}

        return {"hyde_embedding_chunks": res[0]}

    def _validate_params(self, state):
        # 兼容新的意图识别架构
        confirmed_docs = state.get("confirmed_docs", [])
        destinations = state.get("destinations", [])
        rewritten_query = state.get("rewritten_query")

        if not rewritten_query:
            raise ValueError("rewritten_query is required for HyDE search")

        return confirmed_docs, destinations, rewritten_query

    def _call_llm(self, confirmed_docs, destinations, rewritten_query):
        # 构建大语言模型client
        llm_client = get_llm_client()

        # 根据当前状态智能选择主题提示词
        topic_hint = ", ".join(confirmed_docs) if confirmed_docs else ", ".join(destinations)
        if not topic_hint:
            topic_hint = "通用旅游指南"

        # 构建提示词
        prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            topic_hint=topic_hint,
            rewritten_query=rewritten_query
        )

        # 构建message，赋予旅游专家人设
        message = [
            SystemMessage(
                content="您是一位资深的旅游体验师和向导，主要擅长撰写高质量的旅游攻略、交通路线指导和目的地风情介绍。"),
            HumanMessage(content=prompt)
        ]

        # 调用llm
        res = llm_client.invoke(message)
        result = getattr(res, "content", "").strip()

        if not result:
            return ""
        return result

    def _create_search_file_title_expr(self, confirmed_docs):
        # 构建针对旅游指南标题的标量过滤条件 file_title in ["xxx","yy"]
        condition = ", ".join([f'"{name}"' for name in confirmed_docs])
        return f"file_title in [{condition}]"


if __name__ == "__main__":
    state = {
        "destinations": ['成都'],
        "intents": ['交通指南'],
        "rewritten_query": "如何从成都前往春熙路？",
        "confirmed_docs": [],
    }

    vector_search_node = HydeSearchNode()
    result = vector_search_node.process(state)
    print(result)
