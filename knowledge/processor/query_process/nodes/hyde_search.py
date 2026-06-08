from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import get_config
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import create_hybrid_search_requests, execute_hybrid_search_query, \
    get_milvus_client


class HydeSearchNode(BaseNode):
    def process(self, state):
        # 1.参数校验
        item_names, rewritten_query = self._validate_params(state)
        # 2.调用大语言模型，获得hyde
        response = self._call_llm(item_names, rewritten_query)
        print(response)

        # 3.拼接字符串做向量搜索
        embedding_content = f"{rewritten_query} - {response}"

        # 4.做向量化
        bge_m3_client = get_bgem3_client()
        embedding_vectors = generate_hybrid_embeddings(bge_m3_client, [embedding_content])

        # 5.做联合搜索
        expr = self._create_search_item_name(item_names)
        search_requests = create_hybrid_search_requests(
            dense_vector=embedding_vectors["dense"][0],
            sparse_vector=embedding_vectors["sparse"][0],
            expr=expr,
        )

        # 执行搜索
        config = get_config()
        milvus_client = get_milvus_client()
        res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=config.chunks_collection,
            search_requests=search_requests,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            limit=5,
            output_fields=["chunk_id", "content", "item_name"],
        )
        if not res or not res[0]:
            return state
        return {"hyde_embedding_chunks": res[0]}

    def _validate_params(self, state):
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not item_names:
            raise ValueError("item_names is required")
        if not rewritten_query:
            raise ValueError("rewritten_query is required")
        return item_names, rewritten_query

    def _call_llm(self, item_names, rewritten_query):
        # 构建大语言模型client
        llm_client = get_llm_client()

        # 构建提示词
        prompt = USER_HYDE_PROMPT_TEMPLATE.format(item_hint=item_names,rewritten_query=rewritten_query)

        # 构建message
        message = [
            SystemMessage(content=f"您是一位{item_names}的技术文档领域的专家，"
                         f"主要擅长编写技术文档、操作手册、文档规格说明"),
            HumanMessage(content=prompt)
        ]

        # 调用llm
        res = llm_client.invoke(message)

        result = getattr(res,"content","").strip()
        if not result:
            return ""
        # 获得content返回
        return result

    # 构建标量字段搜索条件 item_name in ["xxx","yy"]
    def _create_search_item_name(self, item_name):
        condition = ", ".join([f'"{name}"' for name in item_name])
        return f" item_name in [{condition}]"# 前面有一个空格

if __name__ == "__main__":
    state = {
        "rewritten_query":"关于H3C LA2608，如何使用？",
        "item_names":["H3C LA2608 室内无线网关"],
    }

    vector_search_node = HydeSearchNode()
    result = vector_search_node.process(state)
    print(result)