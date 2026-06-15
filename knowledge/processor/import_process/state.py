"""
导入流程状态类型定义
定义完整的状态结构和辅助函数
"""
from typing import TypedDict, List, Any
import copy

class ImportGraphState(TypedDict, total=False):
    """
    导入流程图状态
    包含整个 LangGraph 导入流程中传递的所有数据
    """
    # ==================== 任务与流程追踪 ====================
    task_id: str                # 任务 ID，用于并发任务的追踪与防重
    status: str                 # 当前阶段状态 (如: initialized, loading, chunking, extracting, completed, failed)
    progress_msg: str           # 进度提示信息，方便后续接前端展示给用户看
    errors: List[str]           # 异常收集列表，保证图节点运行不轻易崩溃

    # ==================== 控制标志 ====================
    is_md_read_enabled: bool    # 是否启用 MD 读取

    # ==================== 路径信息 ====================
    import_file_path: str       # 导入文件的物理/相对路径
    file_dir: str               # 导入(出)文件目录
    md_path: str                # 转换后/原始 Markdown 文件路径

    # ==================== 核心数据 (意图提取节点产出) ====================
    file_title: str             # 文件标题（不含扩展名，常用作主键或溯源字段）
    destination: str            # 识别出的目的地名称 (如: 三亚)
    file_class: str             # 识别出的文档类型/意图 (如: 景区资料)

    # 意图全局向量，以便存储或供后续检索召回使用
    global_dense_vector: list
    global_sparse_vector: dict

    # ==================== 处理中间数据 ====================
    md_content: str             # 读取的完整 Markdown 文档文本内容

    # 文档切片列表。保持原文往下流转，等待后续的“切片嵌入节点”去处理
    chunks: List[Any]


# ==================== 默认状态与工厂函数 ====================

GRAPH_DEFAULT_STATE: ImportGraphState = {
    # 追踪初始化
    "task_id": "",
    "status": "initialized",
    "progress_msg": "准备开始导入",
    "errors": [],

    # 控制与路径初始化
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "md_path": "",

    # 核心数据初始化
    "file_title": "",
    "destination": "",
    "file_class": "",
    "global_dense_vector": [],
    "global_sparse_vector": {},

    # 中间数据初始化
    "md_content": "",
    "chunks": [],               # 初始化为安全的空列表
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持在图的起点注入初始参数

    Args:
        **overrides: 要覆盖的字段字典

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_123", file_title="三亚攻略")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态的深拷贝（绝对避免多任务并发时的全局变量污染问题）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)