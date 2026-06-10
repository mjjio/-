from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def process(self, state:QueryGraphState):
        # 1.获得向量搜索，hyde搜索后的向量结果
        # 网络检索的结果没有做过向量嵌入，不存在chunk_id等字段
        embedding_chunks = state.get("embedding_chunks") or []
        hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []

        # 2.根据不同的权重把需要rrf的数据结合起来
        search_source = {
            "vector_search_results" : (self._normal_vector(embedding_chunks),1.0),
            "hyde_search_results" : (self._normal_vector(hyde_embedding_chunks),1.0)
        }

        # 3.构建rrf_inputs
        rrf_inputs = list(search_source.values())

        # 4.利用rrf计算公式计算所有向量的score
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        # 5.只要rrf_merge_results之中的chunks
        rrf_chunks = [chunk for chunk,_ in rrf_merge_results]

        state["rrf_chunks"] = rrf_chunks

        return state

    # 标准化输入内容
    def _normal_vector(self, embedding_chunks):
        # 'hyde_embedding_chunks': [
        #     {
        #         'chunk_id': 466834516599376102,
        #         'distance': 0.7836564183235168,
        #         'entity': {
        #             'content': 'xxx',
        #             'item_name': 'H3CLA2608室内无线网关',
        # 'chunk_id': 466834516599376102
        # 保存最终结果
        result = []
        for doc in embedding_chunks:
            entity = doc.get("entity")
            if not entity:
                continue
            result.append(entity)
        return result

    def _rrf_merge(self, rrf_inputs):
        rrf_chunks = {}
        chunk_data = {}
        # 计算多路的rrf评分，根据评分倒序召回
        for entity, weight in rrf_inputs:
            for index, doc in enumerate(entity):
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    continue

                rrf_score = weight / (self._rrf_k + index + 1)
                # 同一个chunk_id 只记录一次值
                rrf_chunks[chunk_id] = rrf_chunks.get(chunk_id, float(0)) + rrf_score
                # 存放chunk内容
                chunk_data.setdefault(chunk_id, doc)

        # 根据分数降序排列
        sorted_results = sorted(
            [(chunk_data[chunk_id], score) for chunk_id, score in rrf_chunks.items()],
            key=lambda x: x[1], reverse=True
        )
        return sorted_results[:self._top_k] if self._top_k else sorted_results

if __name__ == "__main__":
    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)
    print(result)