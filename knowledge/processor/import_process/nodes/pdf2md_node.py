import json
import os
import subprocess
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


# pdf2md
# 1. 对文件校验，判断文件是否存在
# 2. 使用mineru工具把pdf转换md
# 3. 获得转换之后的md路径
# 4. 返回需要数据
class Pdf2Md_Node(BaseNode):
    # 重写process方法进行处理逻辑
    def process(self, state:ImportGraphState):
        # 1.对文件进行校验
        import_file_path, file_dir = self.validate_path(state)

        # 2.转换文件格式
        # 如果已经是md则不需要转换
        if state["md_path"]:
            return state
        # 如果不是进行转换
        process_code = self.pdf2md(import_file_path, file_dir)
        if process_code != 0:
            raise PdfConversionError("mineru执行失败")

        # 3.获得转换之后的md路径
        state["md_path"] = self.get_path(import_file_path, file_dir)

        return state

    # 1. 验证文件存在
    def validate_path(self, state:ImportGraphState):
        # 1. 从path获取文件
        import_file_path = state["import_file_path"]
        file_path = Path(import_file_path)
        # 判断文件是否存在
        if not file_path.exists():
            raise FileProcessingError(f"文件不存在{file_path}")

        file_dir = state["file_dir"]
        if not file_dir:
            file_dir = file_path.parent

        return file_path, Path(file_dir)

    # 2. 把pdf文件转为md格式
    def pdf2md(self, import_file_path, file_dir):
        self.log_step("pdf2md......")
        # 命令行本地调用
        os.environ["HF_ENDPOINT"] = "http://hf-mirror.com"
        os.environ["MINERU_MODEL_SOURCE"] = "local"

        cmd = [
            "mineru","-p",
            str(import_file_path),
            "-o",
            str(file_dir),
            "--source", "local",
            "--device", "cpu",
            "--backend", "pipeline",
            "--batch-size", "1",
            "--no-auto-download"
        ]


        # 构建命令执行
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            text=True,
            encoding="utf-8",
            bufsize=1
        )

        for event in proc.stdout:
            self.log_step(f"mineru执行日志{event}")

        process_code = proc.wait()
        if process_code != 0:
            self.log_step("执行mineru失败了...")
        else:
            self.log_step("执行mineru成功了...")
        return process_code

    # 3. 拼出md_path
    def get_path(self, import_file_path, file_dir):
        file_name = import_file_path.stem
        md_path = file_dir/ file_name / "auto" / f"{file_name}.md"
        return str(md_path)

if __name__ == "__main__":
    # 日志初始化
    setup_logging()
    # PdfToMd实例
    pdf_to_md_node = Pdf2Md_Node()
    # 构建参数
    pdf_to_md_node_state = {
        "import_file_path":r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\华为擎云B530 用户指南-(PUCZ,Windows11_03,zh-cn).pdf",
        "file_dir":r"E:\project\shopkeer_brain\knowledge\processor\import_process\output_temp",
        "md_path":""
    }
    # 调用对象的方法
    process_result = pdf_to_md_node.process(pdf_to_md_node_state)
    # 输出
    print(json.dumps(process_result, indent=4,ensure_ascii=False))