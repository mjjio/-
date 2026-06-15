from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    def __init__(self):
        super().__init__()
        # 从配置中读取 RRF 核心参数
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k
        self._kg_weight = self.config.rrf_kg_weight

    def process(self, state: QueryGraphState):
        # 1. 获得三路召回的向量结果 (向量搜索、HyDE搜索、知识图谱搜索)
        # 网络检索的结果因为没有 chunk_id 不参与 RRF，后续直接放入大模型上下文中
        embedding_chunks = state.get("embedding_chunks") or []
        hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []
        kg_chunks = state.get("kg_chunks") or []

        # 2. 根据不同的权重把需要 RRF 的数据结合起来
        # 知识图谱往往匹配度极高，可以通过 config 独立调整 kg_weight (默认 0.7)
        search_source = {
            "vector_search_results": (self._normal_vector(embedding_chunks), 1.0),
            "hyde_search_results": (self._normal_vector(hyde_embedding_chunks), 1.0),
            "kg_search_results": (self._normal_vector(kg_chunks), self._kg_weight)
        }

        # 3. 构建 rrf_inputs
        rrf_inputs = list(search_source.values())

        # 4. 利用 RRF 计算公式计算所有向量的综合 score
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        # 5. 提取融合后排序完毕的 chunks
        rrf_chunks = [chunk for chunk, _ in rrf_merge_results]

        state["rrf_chunks"] = rrf_chunks
        self.logger.info(f"RRF 融合排序完成，共输出 {len(rrf_chunks)} 个去重切片")

        return state

    def _normal_vector(self, chunks):
        """
        标准化输入内容，兼容不同数据源返回的字典结构差异。
        Milvus 嵌套结构示例：
        {
            'distance': 0.783,
            'entity': {
                'chunk_id': 'xxx',
                'content': '正文...',
                'file_title': '成都交通指南'
            }
        }
        """
        result = []
        for doc in chunks:
            # 兼容知识图谱直接返回扁平字典的情况
            if "entity" not in doc:
                if "chunk_id" in doc:
                    result.append(doc)
                continue

            # 处理 Milvus 向量库返回的嵌套结构
            entity = doc.get("entity")
            if not entity:
                continue

            # 兜底机制：如果 chunk_id 在外层而不在 entity 中，强行并入
            if "chunk_id" not in entity and "chunk_id" in doc:
                entity["chunk_id"] = doc["chunk_id"]
            elif "chunk_id" not in entity and "id" in doc:
                entity["chunk_id"] = doc["id"]

            result.append(entity)

        return result

    def _rrf_merge(self, rrf_inputs):
        rrf_chunks = {}
        chunk_data = {}

        # 计算多路的 RRF 评分，根据评分倒序召回
        for entity_list, weight in rrf_inputs:
            for index, doc in enumerate(entity_list):
                chunk_id = doc.get("chunk_id")
                if not chunk_id:
                    continue

                # RRF 核心公式: score = weight / (k + rank)
                rrf_score = weight / (self._rrf_k + index + 1)

                # 同一个 chunk_id 在多路中被命中时，分数累加
                rrf_chunks[chunk_id] = rrf_chunks.get(chunk_id, 0.0) + rrf_score
                # 记录 chunk 的原始数据，以备最终输出
                chunk_data.setdefault(chunk_id, doc)

        # 根据累加的分数降序排列
        sorted_results = sorted(
            [(chunk_data[chunk_id], score) for chunk_id, score in rrf_chunks.items()],
            key=lambda x: x[1],
            reverse=True
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
