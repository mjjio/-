import json

from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.state import ImportGraphState, create_default_state

# 引入我们重构后的旅游知识库核心节点
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.document_splitter_node import DocumentSplitterNode
from knowledge.processor.import_process.nodes.meta_data_extract import  LocationIntentRecognitionNode# 替代了原本的 ItemName
from knowledge.processor.import_process.nodes.chunks_embedding_node import ChunkEmbeddingNode
from knowledge.processor.import_process.nodes.chunks_upload_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.knowledge_graph_node import KnowledgeGraphNode


def create_graph_import():
    # 创建 langGraph 的 builder
    builder = StateGraph(ImportGraphState)

    # 设置入口节点
    builder.set_entry_point("entry_node")

    # 注册所有节点
    nodes = {
        "entry_node": EntryNode(),
        "document_split_node": DocumentSplitterNode(),
        "intent_rec_node": LocationIntentRecognitionNode(),  # 替换为旅游意图识别节点
        "bge_embedding_node": ChunkEmbeddingNode(),
        "import_milvus_node": ImportMilvusNode(),
        "kg_node": KnowledgeGraphNode()
    }

    # 遍历添加节点
    for key, value in nodes.items():
        builder.add_node(key, value)

    # ==========================================
    # 去掉条件路由，直接添加线性边 (一条龙执行)
    # ==========================================
    builder.add_edge("entry_node", "document_split_node")
    builder.add_edge("document_split_node", "intent_rec_node")
    builder.add_edge("intent_rec_node", "bge_embedding_node")
    builder.add_edge("bge_embedding_node", "import_milvus_node")
    # builder.add_edge("import_milvus_node", END)
    builder.add_edge("import_milvus_node", "kg_node")
    builder.add_edge("kg_node", END)

    # 图编译，返回编译之后对象
    graph = builder.compile()
    return graph


import_graph_app = create_graph_import()


# 构建状态数据，流式输出
def run_graph_import(import_file_path: str, file_dir: str):
    # 构建状态数据
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    # 初始化状态
    init_state = create_default_state(**state)

    # 图执行
    final_state = None
    for event in import_graph_app.stream(init_state):
        # event 字典遍历
        for node_name, current_state in event.items():
            print(f">>> [图流转追踪] 节点 '{node_name}' 执行完毕。当前状态: {current_state.get('status')}")
            final_state = current_state

    return final_state


if __name__ == "__main__":
    setup_logging()

    # 【注意】因为我们的 EntryNode 里写死了只放行 .md，所以这里请改成 md 文件路径
    import_file_path = r"E:\project\初始化项目\交通指南\成都交通指南.md"
    file_dir = r"E:\project\初始化项目\交通指南"

    # 测试编排流程
    print("========== 开始启动旅游知识库图谱构建流程 ==========")
    final_state = run_graph_import(
        import_file_path=import_file_path,
        file_dir=file_dir
    )

    # 打印最终结果摘要 (避免直接 dumps 打印巨大的向量数组导致控制台崩溃)
    if final_state:
        print("\n========== 导入流程完美结束 ==========")
        print(f"-> 文件标题   : {final_state.get('file_title')}")
        print(f"-> 识别意图   : {final_state.get('file_class')}")
        print(f"-> 目的地     : {final_state.get('destination')}")
        print(f"-> 产生切片数 : {len(final_state.get('chunks', []))} 个")
        print(f"-> 最终状态   : {final_state.get('status')}")
        if final_state.get('errors'):
            print(f"-> 流程警告   : {final_state.get('errors')}")