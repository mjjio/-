"""查询业务服务"""

import uuid
import logging
from typing import List, Dict, Any

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import (
    update_task_status,
    get_task_result,
    get_task_status,
    get_done_task_list,
    get_running_task_list,
    add_done_task,               # 新增：记录节点完成状态
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED
)
from knowledge.utils.sse_util import create_sse_queue, push_sse_event

logger = logging.getLogger(__name__)

# 为前端进度条提供高级感、友好的中文提示
NODE_NAME_MAP = {
    "intent_confirm": "识别旅行意图与目的地",
    "search_embedding": "搜索本地图文攻略 (向量检索)",
    "search_embedding_hyde": "关联拓展攻略 (HyDE检索)",
    "web_search_mcp": "接入全网实时旅游资讯",
    "kg_search_node": "提取旅游知识图谱特征",
    "rrf": "多路召回结果融合排序 (RRF)",
    "rerank": "大模型语义精排优选",
    "answer_output": "撰写专属旅行指南"
}

class QueryService:

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def submit_query(self, task_id: str, is_stream: bool):
        """提交查询任务：更新状态 + 流式模式创建 SSE 队列。"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    def run_query_graph(self, task_id: str, session_id: str, user_query: str, is_stream: bool):
        """执行 LangGraph 查询流程（流式追踪进度）"""
        try:
            default_state = {
                "original_query": user_query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
            }

            # 【核心修复】：将 invoke 改为 stream，以便捕获每个节点的完成事件
            for event in query_app.stream(default_state):
                for node_name, node_state in event.items():
                    # 过滤掉辅助分发用的虚拟节点，不让它们显示在前端
                    if node_name in ["multi_search", "join"]:
                        continue

                    # 翻译成友好的中文名
                    display_name = NODE_NAME_MAP.get(node_name, node_name)
                    logger.info(f"[{task_id}] 完成节点: {display_name}")

                    # 标记节点完成，存入 task_util 的 done_list
                    add_done_task(task_id, display_name)

                    # 实时推送进度事件给前端进度条
                    if is_stream:
                        push_sse_event(task_id, "progress", {
                            "status": get_task_status(task_id),
                            "done_list": get_done_task_list(task_id),
                            "running_list": get_running_task_list(task_id),
                        })

        except Exception as e:
            logger.error(f"查询流程执行失败: {e}", exc_info=True)
            # 异常时可以推送 failed 状态，让前端知道崩溃了
            update_task_status(task_id, "failed")
            if is_stream:
                push_sse_event(task_id, "error", {"error": str(e)})
        finally:
            # 如果没崩溃，确保最终状态是 completed
            if get_task_status(task_id) != "failed":
                update_task_status(task_id, TASK_STATUS_COMPLETED)
                if is_stream:
                    push_sse_event(task_id, "progress", {
                        "status": get_task_status(task_id),
                        "done_list": get_done_task_list(task_id),
                        "running_list": get_running_task_list(task_id),
                    })

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)