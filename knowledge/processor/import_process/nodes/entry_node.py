import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    # 实现抽象方法process
    def process(self, state:ImportGraphState) -> ImportGraphState:
        # 使用日志输出信息
        self.log_step("步骤1","[开始检查文件类型]")
        # 导入文件目录和地址
        file_path = state.get("import_file_path") # 都是用.get方法防止空值出现
        file_dir = state.get("file_dir")

        # 校验导入的文件路径以及文件所在的目录
        if not file_path or not file_dir:
            raise ValidationError("文件目录或者文件不存在")

        # 获取文件后缀
        path = Path(file_path)
        # .pdf .md格式
        suffix = path.suffix.lower()
        # 判断
        if suffix == '.pdf':
            # 日志输出
            self.log_step("pdf","[pdf检查通过]")
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = file_path
        elif suffix == '.md':
            # 日志输出
            self.log_step("md","[md检查通过]")
            state["is_md_read_enabled"] = True
            state["md_path"] = file_path
        else:
            self.log_step("other","检查不通过")
            raise ValidationError("文件格式错误")
        # state写入文件标题
        file_title = path.stem
        state["file_title"] = file_title
        return state

if __name__ == '__main__':
    # 日志初始化
    setup_logging()

    # 构建字典数据
    enty_state = {
        "file_dir":r"knowledge/processor/import_process/import_temp_dir",
        "import_file_path":r"knowledge/processor/import_process/import_temp_dir/hak180产品安全手册.pdf",
    }

    # EntryNode实例化,执行父类里面 __init__方法
    entry_node = EntryNode()
    # 调用实例，自动执行父类里面 __call__，__call__方法帮调用process方法
    res = entry_node(enty_state)
    print(json.dumps(res, ensure_ascii=False, indent=4))