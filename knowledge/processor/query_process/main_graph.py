"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv
from knowledge.processor.query_process.state import QueryGraphState

# 加载环境变量
load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Args:
        state: 查询图状态。

    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> CompiledStateGraph:
    pass


# 创建全局图实例
query_app = create_query_graph()


if __name__ == "__main__":
    pass