from typing import Any, Dict

from knowledge.processor.query_process.base import BaseNode, setup_logging
# 导入我们刚刚封装好的线程安全方法
from knowledge.utils.bge_rerank_util import compute_rerank_scores


class RerankerNode(BaseNode):
    def process(self, state):

        # 1.获得query，优先使用重写后的精准问题
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # 2.获得rrf_chunks 和 mcp调用的 web_search_docs 做合并归一化
        merged_multi_docs = self._merge_multi_docs(state)

        # 3.送入reranker模型,进行精排打分
        reranked_docs = self._call_reranker_model(user_query, merged_multi_docs)

        # 4.做断崖检测 (Cliff Cutoff)，剔除低相关性尾部文档
        cutoff_docs = self._cliff_cutoff(reranked_docs)

        # 更新state对应字段
        state['reranked_docs'] = cutoff_docs
        self.logger.info(f"重排序完成，输入 {len(merged_multi_docs)} 篇，截断后保留 {len(cutoff_docs)} 篇")

        return state


    def _merge_multi_docs(self, state):
        final_docs = []

        # 1.读取本地的rrf_docs
        for rrf_doc in (state.get('rrf_chunks') or []):
            if not isinstance(rrf_doc, dict):
                continue

            content = rrf_doc.get('content', '').strip()
            if not content:
                continue

            title = rrf_doc.get('file_title', '') or rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get("chunk_id")
            final_docs.append(self._format_docs(content=content, title=title, chunk_id=chunk_id, source="local"))

        # 2.读取web联网数据
        for web_doc in (state.get("web_search_docs") or []):
            if not isinstance(web_doc, dict):
                continue

            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()
            if not content:
                continue

            title = web_doc.get("title", "").strip()
            url = web_doc.get("url", "").strip()
            final_docs.append(self._format_docs(content=content, title=title, url=url, source="web"))

        return final_docs


    def _format_docs(self, content: str, title: str = "", chunk_id=None, url: str = "", source: str = "") -> Dict[str, Any]:
        return {
            "content": content,
            "title": title,
            "chunk_id": chunk_id,
            "url": url,
            "source": source
        }


    def _call_reranker_model(self, user_query, merged_multi_docs):
        if not merged_multi_docs:
            return []

        # 构建(query -> doc)的pair对
        query_doc_content_pairs = [(user_query, doc.get("content")) for doc in merged_multi_docs]

        try:
            # 【核心修复】：统一使用带线程锁的 compute_rerank_scores 替代原先直接调用 model.compute_score
            rerank_scores = compute_rerank_scores(query_doc_content_pairs)

            # 把score和doc组装到一个列表之中
            result = [{**doc, "score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]

            # 根据score倒序排序返回
            rerank_docs = sorted(result, key=lambda doc: doc["score"], reverse=True)
            return rerank_docs

        except Exception as e:
            self.logger.error(f"重排序计算失败: {e}", exc_info=True)
            return []


    def _cliff_cutoff(self, reranked_docs):
        if not reranked_docs:
            return []

        upper = min(self.config.rerank_max_top_k, len(reranked_docs))
        lower = min(self.config.rerank_min_top_k, len(reranked_docs))

        cutoff_pos = upper

        for i in range(lower - 1, upper - 1):
            current_pos_score = reranked_docs[i].get('score')
            next_pos_score = reranked_docs[i + 1].get('score')

            if current_pos_score is None or next_pos_score is None:
                continue

            abs_gap = current_pos_score - next_pos_score
            abs_ratio = abs_gap / (abs(next_pos_score) + 1e-6)

            if abs_gap > self.config.rerank_gap_abs or abs_ratio > self.config.rerank_gap_ratio:
                cutoff_pos = i + 1
                self.logger.info(f"触发断崖截断: cutoff_pos={cutoff_pos}, abs_gap={abs_gap:.4f}, abs_ratio={abs_ratio:.4f}")
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
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
        ],
    }

    node = RerankerNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")