from knowledge.processor.query_process.base import BaseNode


class RerankerNode(BaseNode):
    def process(self, state):
        # 1.获得query
        user_query = state.get('rewritten_query','') or state.get('original_query','')
        # 2.获得rrf_chunks 和 mcp调用的web_search_docs
        # 3.送入reranker模型
        # 4.返回结果更新state