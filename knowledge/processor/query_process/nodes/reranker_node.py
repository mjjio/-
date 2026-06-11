from typing import Any, Dict

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.utils.bge_rerank_util import get_reranker_model


class RerankerNode(BaseNode):
    def process(self, state):
        # 1.获得query
        user_query = state.get('rewritten_query','') or state.get('original_query','')
        # 2.获得rrf_chunks 和 mcp调用的web_search_docs做合并
        merged_multi_docs = self._merge_multi_docs(state)
        # 3.送入reranker模型,进行精排打分
        reranked_docs = self._call_reranker_model(user_query, merged_multi_docs)
        # 4.做断崖检测
        cutoff_docs = self._cliff_cutoff(reranked_docs)
        # 更新state对应字段
        state['reranked_docs'] = cutoff_docs
        return state

    def _merge_multi_docs(self, state):
        final_docs = []
        # 1.读取本地的rrf_docs
        for rrf_doc in (state.get('rrf_chunks') or []):
            # 1.校验文档对象类型
            if not isinstance(rrf_doc, dict):
                continue
            # 2.提取文档内容
            content = rrf_doc.get('content','').strip()

            if not content:
                continue
            title = rrf_doc.get('title',"").strip()
            chunk_id = rrf_doc.get("chunk_id")
            # 3.格式化需要的数据
            final_docs.append(self._format_rrf_docs(content=content, title=title, chunk_id=chunk_id, source="local"))

        # 2.读取web数据
        for web_doc in (state.get("web_search_docs") or []):
            if not isinstance(web_doc, dict):
                continue
            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()

            if not content:
                continue
            title = web_doc.get("title","").strip()
            url = web_doc.get("url","").strip()
            final_docs.append(self._format_rrf_docs(content=content, title=title, url=url,source="web"))
        return final_docs

    def _format_rrf_docs(self, content: str, title: str = "", chunk_id=None, url: str = "", source: str = "") -> Dict[
        str, Any]:
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source
        }

    def _call_reranker_model(self, user_query, merged_multi_docs):
        """

                Args:
                    user_query: 用户输出的查询问题
                        merged_multi_docs: 不同来源的两路合并之后的文档（all_in_docs）
                    Returns:

        """
        # 判断是否存在多源文档
        if not merged_multi_docs:
            return []

        # 获取reranker模型
        rerank_model = get_reranker_model()
        if not rerank_model:
            self.logger.error("重排序模型加载失败")
            return []
        # 构建(q->d)的pair对
        query_doc_content_pairs = [(user_query,doc.get("content")) for doc in merged_multi_docs]
        try:
            # 进行计算
            rerank_scores = rerank_model.compute_score(query_doc_content_pairs)
            # 把score和doc放到一个列表之中
            result = [{**doc, "score":score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            # 根据score倒数排序返回
            rerank_docs = sorted(result, key=lambda doc: doc["score"], reverse=True)

            return rerank_docs
        except Exception as e:
            self.logger.error(f"重排序失败{e}")
            return [{**merged_multi_docs, "score": None}]

    def _cliff_cutoff(self, reranked_docs):
        # 确定截断内容上下界
        upper = min(self.config.rerank_max_top_k, len(reranked_docs))
        lower = min(self.config.rerank_min_top_k, len(reranked_docs))

        # 从lower+1,upper-1做截取
        cutoff_pos = upper
        for i in range(lower-1, upper-1):
            current_pos_score = reranked_docs[i].get('score')
            next_pos_score = reranked_docs[i+1].get('score')
            # 空值校验
            if not current_pos_score or not next_pos_score:
                continue

            abs_gap = current_pos_score - next_pos_score
            abs_ratio = abs_gap / (abs(next_pos_score) + 1e-6) # 防止分母为0
            # 如果差距过大，或者降低幅度过大，做截断
            if abs_gap > self.config.rerank_gap_abs or abs_ratio > self.config.rerank_gap_ratio:
                cutoff_pos = i+1
                self.logger.info(f"触发截断{cutoff_pos=},{abs_gap=},{abs_ratio=}")
                break

        return reranked_docs[:cutoff_pos]

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")
