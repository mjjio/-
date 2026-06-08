import json
import re
from typing import List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bgem3_client_util import get_bgem3_client, generate_hybrid_embeddings
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_client_util import get_milvus_client, create_hybrid_search_requests, \
    execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages

# 大语言模型操作类
class ItemNameLlm:
    def extract_item_name(self, origin_query, history):
        # 1. 创建大语言模型调用对象
        llm_client = get_llm_client(response_format=True) # 是否输出json格式

        # 2. 根据prompt构建提示词
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history, query=origin_query)
        messages = [
            SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
            HumanMessage(content=human_prompt)
        ]

        # 3. 调用大语言模型
        llm_response = llm_client.invoke(messages)

        # 4. 获得大语言模型处理结果
        llm_result = llm_response.content

        # 5. 对llm_result进行结果清洗
        final_result = self._clean_llm_result(llm_result)

        return final_result

    def _clean_llm_result(self, llm_result):

        # 使用正则表达式去除···json ···
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_result)
        content = re.sub(r"\s*```$", "", cleaned)

        # 反序列化
        parsed_content = json.loads(content)

        # 去掉item_name之中空内容和空格
        item_name = [item_name.strip() for item_name in parsed_content["item_names"] if item_name]

        return {
            "item_names": item_name,
            "rewritten_query" : parsed_content.get("rewritten_query")
        }

# 向量数据库操作类
class ItemNameVector:
    # {
    #     "item_names": [list],
    #     "rewritten_query": ]
    # }
    def search_vector(self, item_names:List[str]) -> Tuple[List[str], List[str]]:
        # 1. 根据item_name查询向量数据库，得到稠密和稀疏向量
        searched_result = self.match_vector(item_names)
        # 2. 评分对齐
        confirmed, options = self._align_score(searched_result)

        return confirmed, options

    # 根据商品名获得稠密和稀疏向量
    def match_vector(self, item_names):
        # 包装最后的结果
        match_vector = []

        # 获得向量数据库客户端和嵌入模型
        milvus_client = get_milvus_client()
        bge_m3_client = get_bgem3_client()

        # 调用工具类的方法实现
        # {
        #     "dense": 密稠列表,
        #     "sparse": 稀疏向量
        # }
        hybrid_embedding_result = generate_hybrid_embeddings(bge_m3_client, item_names)

        # 获得每一个item_name对应的稠密和稀疏向量
        for index, item_name in enumerate(item_names):

            dense_vector = hybrid_embedding_result["dense"][index]
            sparse_vector = hybrid_embedding_result["sparse"][index]

            # 混合查询条件
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
            )

            # 调研方法执行查询
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client,
                collection_name="kb_item_names_v2",
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"]
            )
            item_name_search_result = {
                "extracted_name": item_name,
                "matches": [
                    {
                        "item_name": h["entity"]["item_name"],
                        "score": h["distance"]
                    }
                    for h in (hybrid_search_result[0]
                              if hybrid_search_result else [])
                ]
            }
            match_vector.append(item_name_search_result)
        return match_vector

    # 评分对齐
    def _align_score(self, searched_result):
        confirmed, options = [],[]
        # {
        #     "extracted_name": item_name,
        #     "matches": [
        #         {
        #             "item_name": h["entity"]["item_name"],
        #             "score": h["distance"]
        #         }
        #         for h in (hybrid_search_result[0]
        #                   if hybrid_search_result else [])
        #     ]
        # }
        for item_name_result in searched_result:
            # extracted_name = item_name_result["extracted_name"]
            matches = sorted(item_name_result.get("matches"), key=lambda x: x["score"], reverse=True)
            # 1 matches列表遍历，得到列表中每部分数据分数 score值
            # 1.1 score 大于 0.7 处理 放到confirmed列表
            # confirmed表示确认的一个值，options表示提供给用户的三个选项
            high = [m for m in matches if m.get('score') >= 0.7]
            # 如果high只有一个值
            if len(high) == 1:
                confirmed.append(high[0].get("item_name"))
            elif len(high) > 1:
                # 如果不只一个高置信度的值
                option_result = [i.get("item_name") for i in high[:3]]
                options.extend(option_result)
            else:
                # 如果没有高置信度的值
                middle = [m for m in matches if m.get('score') >= 0.6]
                options = [i.get("item_name") for i in middle[:3]]

        return confirmed, options

class ItemNameConfirmNode(BaseNode):
    def __init__(self):
        super().__init__()
        # llm操作对象
        self._item_name_llm = ItemNameLlm()
        # 向量嵌入对象
        self._item_name_vector = ItemNameVector()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1.校验用户输入的original_query参数
        origin_query = state.get("original_query")

        # 2.获取用户上下文会话
        session_id = state.get("session_id")
        # 调用mongodb
        chat_history = get_recent_messages(session_id, limit=10)
        # 拼接上下文
        history = ""
        for msg in chat_history:
            role = msg.get("role")
            text = msg.get("text")
            history += f"{role} - {text}\n"
        
        # 3.调用大语言模型，提取商品名
        # llm返回格式：在提示词约定好格式
        # {
        #     "item_names": ["商品A", "商品B"],
        #     "rewritten_query": "关于商品A和商品B，..."
        # }
        llm_result = self._item_name_llm.extract_item_name(origin_query, history)

        item_names = llm_result.get("item_names")
        rewritten_query = llm_result.get("rewritten_query")

        # 判断item_names是否非空
        # 4.根据大语言模型返回的商品名，进行向量检索
        if item_names:
            # 根据item_name搜索item_name
            confirmed, options = self._item_name_vector.search_vector(item_names)
        else:
        # llm没有提取到item_name
            confirmed, options = [],[]
        # 5.更新state并返回
        self._update_state(state, item_names, rewritten_query, confirmed, options)

        state["history"] = chat_history
        return state

    # 更新state
    def _update_state(self, state, item_names, rewritten_query, confirmed, options):
        # 如果商品名确定唯一
        if confirmed:
            state["rewritten_query"] = rewritten_query
            state["item_names"] = confirmed
        # 如果需要用户自己选择
        elif options:
            state["answer"] = f"请选择具体问题:{','.join(options)}"
        # 没有提取出商品名
        else:
            state["answer"] = "当前问题无法识别..."


if __name__ == "__main__":
    test_state = {
        # "original_query": "我想知道这个如何使用？"
        "original_query": "我想知道H3C LA2608如何使用？"
    }
    print(f"输入: {json.dumps(test_state,ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result=node_item_name_confirm.process(test_state)

    print(f"确认商品: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")