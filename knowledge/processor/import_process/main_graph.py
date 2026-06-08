import json

from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.chunks_embedding_node import ChunkEmbeddingNode
from knowledge.processor.import_process.nodes.document_splitter_node import DocumentSplitterNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.chunks_upload_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.knowledge_graph_node import KnowledgeGraphNode
from knowledge.processor.import_process.nodes.md_img_node import MdImgNode
from knowledge.processor.import_process.nodes.pdf2md_node import Pdf2Md_Node
from knowledge.processor.import_process.state import ImportGraphState, create_default_state

# 路由方法
def import_router(state:ImportGraphState):
    # 判断文件类型 md  pdf
    if state.get('is_pdf_read_enabled'):
        return "pdf_to_md"

    if state.get('is_md_read_enabled'):
        return "md_img_node"
    return END

# 通过langGraph机制把多个节点执行
# 两个节点
# entry_node : 文件类型检查
# pdf_to_md: pdf转换md
# entry_node =》 pdf_to_md
# 创建langGraph的builder，添加节点，添加边，编译，返回编译对象结果
def create_graph_import() -> StateGraph:
    # 创建langGraph的builder
    builder = StateGraph(ImportGraphState)

    # 添加节点
    # 设置入口节点
    builder.set_entry_point("entry_node")
    # 添加节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md": Pdf2Md_Node(),
        "md_img_node": MdImgNode(),
        "document_split_node": DocumentSplitterNode(),
        "item_name_rec_node":ItemNameRecognitionNode(),
        "bge_embedding_node": ChunkEmbeddingNode(),
        "import_milvus_node": ImportMilvusNode(),
        "kg_node":KnowledgeGraphNode()
    }
    # 遍历
    for key,value in nodes.items():
        builder.add_node(key, value)

    # 条件边
    # 根据entry_node节点返回数据判断
    # 如果文档类型是md， 进入到 md_img_node
    # 如果文档类型是pdf，进入到 pdf_to_md
    # 参数一：开始节点位置
    # 参数二：判断（路由）方法
    # 参数三：根据参数二返回结果，决定进入哪个节点
    builder.add_conditional_edges(
        "entry_node",
        import_router,
        {
            "md_img_node":"md_img_node",
            "pdf_to_md":"pdf_to_md",
            END:END
        }
    )

    # 添加边
    builder.add_edge("entry_node", "pdf_to_md")
    builder.add_edge("pdf_to_md", "md_img_node")
    builder.add_edge("md_img_node", "document_split_node")
    builder.add_edge("document_split_node", "item_name_rec_node")
    builder.add_edge("item_name_rec_node", "bge_embedding_node")
    builder.add_edge("bge_embedding_node", "import_milvus_node")
    builder.add_edge("import_milvus_node", "kg_node")
    builder.add_edge("kg_node", END)

    # 图编译，返回编译之后对象
    graph = builder.compile()
    return graph

graph = create_graph_import()

# 测试
# 构建状态数据，流式输出
def run_graph_import(import_file_path:str,
                     file_dir:str):
    # 获取graph对象
    # graph = create_graph_import()
    # 构建状态数据
    state = {
        "import_file_path":import_file_path,
        "file_dir":file_dir
    }
    # 解包
    init_state = create_default_state(**state)
    # 图执行
    final_state = None
    for event in graph.stream(init_state):
        # event字典遍历
        for node_name,state in event.items():
            print(f"运行节点的:{node_name},state:{state}")
            final_state = state
    return final_state

if __name__ == "__main__":
    setup_logging()
    import_file_path = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\H3CLA2608室内无线网关用户手册-6W100-整本手册.pdf"
    file_dir = r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir"
    # 1. 测试编排流程
    final_state=run_graph_import(
        import_file_path=import_file_path,
                    file_dir=file_dir)
    print(json.dumps(final_state, indent=2,
                     ensure_ascii=False))