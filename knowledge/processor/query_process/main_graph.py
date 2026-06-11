"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.nodes.answer import AnswerOutputNode
from knowledge.processor.query_process.nodes.hyde_search import HydeSearchNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.rrf_node import RrfNode
from knowledge.processor.query_process.nodes.search_chunks import VectorSearchNode
from knowledge.processor.query_process.nodes.mcp_web_search import MCPWebSearchNode
from knowledge.processor.query_process.nodes.reranker_node import RerankerNode
from knowledge.processor.query_process.state import QueryGraphState

# 加载环境变量
load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> CompiledStateGraph:
    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState) # type:ignore

    # 2. 实例化节点
    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda x: x,   # 虚拟节点
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HydeSearchNode(),
        "web_search_mcp": MCPWebSearchNode(),
        "join": lambda x: {},  # 多路搜索汇合（虚节点）
        "rrf": RrfNode(),
        "rerank": RerankerNode(),
        # "answer_output": AnswerOutputNode()

    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)  # type:ignore

    # 4. 设置入口点
    workflow.set_entry_point("item_name_confirm")

    # 5. 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: END
        }
    )

    # 6. 多路搜索分发（并行执行）
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "web_search_mcp")

    # 7. 多路搜索汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("web_search_mcp", "join")

    # 8. 顺序边
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", END)
    # workflow.add_edge("rerank", "answer_output")
    # workflow.add_edge("answer_output", END)

    # 9. 返回可运行的状态
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()


if __name__ == "__main__":
    setup_logging()

    print("=" * 60)
    print("开始测试: 查询流程主图 (main_graph)")
    print("=" * 60)

    mock_state_1 = {
        "original_query": "关于H3C LA2608，如何使用？",
        "session_id": "test_session_main_graph",
        "task_id": "test_task_001",
        "is_stream": False,
    }

    print(f"  查询: {mock_state_1['original_query']}")
    print(f"  session_id: {mock_state_1['session_id']}")
    print(f"  is_stream: {mock_state_1['is_stream']}")

    result_1 = query_app.invoke(mock_state_1)
    print(result_1)
