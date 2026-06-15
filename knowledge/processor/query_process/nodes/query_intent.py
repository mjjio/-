import json
import re
from typing import List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import INTENT_DESTINATION_EXTRACT_TEMPLATE, QUERY_REWRITE_TEMPLATE
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import (
    get_milvus_client,
    create_hybrid_search_requests,
    execute_hybrid_search_query
)
from knowledge.utils.mongo_history_util import get_recent_messages
from knowledge.processor.import_process.config import get_config


class QueryIntentLlm:
    def extract_intent(self, origin_query, history):
        # 1. 阶段一：提取用户原本的目的地和意图
        llm_client = get_llm_client(response_format=True)
        human_prompt = INTENT_DESTINATION_EXTRACT_TEMPLATE.format(history_text=history, query=origin_query)
        messages = [
            SystemMessage(content="你是一个意图提取引擎。擅长理解用户意图、指代消解和提取关键旅游信息。"),
            HumanMessage(content=human_prompt)
        ]

        llm_response = llm_client.invoke(messages)
        llm_result = getattr(llm_response, "content", "")

        return self._clean_extract_result(llm_result)

    def rewrite_query(self, origin_query, history, confirmed_info):
        # 阶段二：根据确定的标准主题进行查询重写
        llm_client = get_llm_client(response_format=False)
        human_prompt = QUERY_REWRITE_TEMPLATE.format(
            history_text=history,
            query=origin_query,
            confirmed_info=confirmed_info
        )
        messages = [
            SystemMessage(content="你是一个专业的查询改写引擎。"),
            HumanMessage(content=human_prompt)
        ]
        llm_response = llm_client.invoke(messages)
        rewritten = getattr(llm_response, "content", "").strip()

        # 清除可能带有的首尾引号
        return re.sub(r'^["\']|["\']$', '', rewritten)

    def _clean_extract_result(self, llm_result):
        content = re.sub(r"^```json\s*", "", llm_result, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)

        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            return {"destinations": [], "intents": []}

        destinations = [d.strip() for d in parsed_content.get("destinations", []) if d and d.strip()]
        intents = [i.strip() for i in parsed_content.get("intents", []) if i and i.strip()]

        return {
            "destinations": destinations,
            "intents": intents
        }


class GlobalIntentVector:
    def search_vector(self, destinations: List[str], intents: List[str]) -> Tuple[List[str], List[str]]:
        search_texts = self._build_search_texts(destinations, intents)
        if not search_texts:
            return [], []

        searched_result = self.match_vector(search_texts)
        confirmed, options = self._align_score(searched_result)

        return confirmed, options

    def _build_search_texts(self, destinations, intents):
        search_texts = []
        intent_str = ",".join(intents) if intents else ""

        if destinations:
            for dest in destinations:
                text = f"{dest} {intent_str}".strip()
                search_texts.append(text)
        elif intents:
            search_texts.append(intent_str)

        return search_texts

    def match_vector(self, search_texts):
        match_vector_result = []

        milvus_client = get_milvus_client()
        bge_m3_client = get_bgem3_client()
        config = get_config()

        global_collection = getattr(config, 'global_collection', 'tourism_global_docs_v1')

        hybrid_embedding_result = generate_hybrid_embeddings(bge_m3_client, search_texts)

        for index, text in enumerate(search_texts):
            dense_vector = hybrid_embedding_result["dense"][index]
            sparse_vector = hybrid_embedding_result["sparse"][index]

            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
            )

            hybrid_search_result = execute_hybrid_search_query(
                milvus_client,
                collection_name=global_collection,
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["destination", "file_class"]
            )

            matches = []
            if hybrid_search_result and len(hybrid_search_result) > 0:
                for h in hybrid_search_result[0]:
                    entity = h["entity"]
                    dest = entity.get("destination", "")
                    cls = entity.get("file_class", "")
                    # 拼装出全局库中存在的标准主题作为选项依据
                    topic_name = f"{dest}-{cls}" if dest and cls else (dest or cls)
                    matches.append({
                        "topic_name": topic_name,
                        "score": h["distance"]
                    })

            match_vector_result.append({
                "search_text": text,
                "matches": matches
            })

        return match_vector_result

    def _align_score(self, searched_result):
        confirmed, options = [], []

        for result in searched_result:
            extracted_text = result["search_text"].replace(" ", "-")
            matches = sorted(result.get("matches", []), key=lambda x: x["score"], reverse=True)

            # 高置信度判断
            high = [m for m in matches if m.get('score') >= 0.7]

            if high:
                extract = next((h for h in high if str(h['topic_name']) == extracted_text), None)
                if extract:
                    picked = extract.get('topic_name')
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0].get('topic_name')
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    for h in high[:3]:
                        picked = h.get('topic_name')
                        if picked not in confirmed and picked not in options:
                            options.append(picked)
            else:
                # 中低置信度降级为提供选项
                middle = [m for m in matches if m.get('score') >= 0.6]
                if middle:
                    for m in middle[:3]:
                        picked = m.get('topic_name')
                        if picked not in confirmed and picked not in options:
                            options.append(picked)

        return confirmed, options[:3]


class QueryIntentConfirmNode(BaseNode):
    def __init__(self):
        super().__init__()
        self._intent_llm = QueryIntentLlm()
        self._global_vector = GlobalIntentVector()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        origin_query = state.get("original_query", "")
        session_id = state.get("session_id", "default_session")

        chat_history = get_recent_messages(session_id, limit=10)
        history = ""
        for msg in chat_history:
            history += f"{msg.get('role', 'user')} - {msg.get('text', '')}\n"

        # 步骤 1：大语言模型提取原始意图和目的地
        extracted_res = self._intent_llm.extract_intent(origin_query, history)
        destinations = extracted_res.get("destinations", [])
        intents = extracted_res.get("intents", [])

        self.logger.info(f"LLM 阶段1 (提取) -> 目的地:{destinations}, 意图:{intents}")

        # 步骤 2：向量库特征搜索与置信度对齐
        if destinations or intents:
            confirmed, options = self._global_vector.search_vector(destinations, intents)
        else:
            confirmed, options = [], []

        self.logger.info(f"向量对齐结果 -> 唯一高置信主题: {confirmed}, 备选项: {options}")

        # 步骤 3：根据置信度分流（LLM二次重写 或 选项拦截）
        rewritten_query = origin_query

        if len(confirmed) == 1:
            # 只有获得了唯一的高置信度结果，才调用 LLM 进行问题重写
            standard_topic = confirmed[0]
            rewritten_query = self._intent_llm.rewrite_query(origin_query, history, standard_topic)
            self.logger.info(f"LLM 阶段2 (重写) -> 原问题:[{origin_query}] => 重写后:[{rewritten_query}]")

            state["rewritten_query"] = rewritten_query
            state["confirmed_docs"] = confirmed

        elif len(confirmed) > 1 or options:
            # 存在分歧或不唯一，返回给用户选择
            all_opts = confirmed + options
            unique_opts = list(dict.fromkeys(all_opts))[:3]
            state["answer"] = f"请问您具体想了解关于哪个方面的：{', '.join(unique_opts)}？"

        else:
            # 置信度全低于 0.6，底层拦截
            state["answer"] = "抱歉，知识库中未能找到完全匹配您目的地和意图的信息，请尝试更换关键词。"

        # 更新通用状态
        state["destinations"] = destinations
        state["intents"] = intents
        state["history"] = chat_history

        return state


if __name__ == "__main__":
    test_state = {
        # "original_query": "我想知道这个如何使用？"
        "original_query": "成都怎么去春熙路？"
    }
    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = QueryIntentConfirmNode()
    result = node_item_name_confirm.process(test_state)

    print(f"用户目的地: {result.get('destinations')}")
    print(f"用户意图: {result.get('intents')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")
