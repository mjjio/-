import base64
import os
import re
from pathlib import Path
from typing import Tuple

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_util import get_minio_client


class MdImgNode(BaseNode):
    # 核心方法process
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.验证md_path之下文件存在,并且获得md_content, img_path
        md_content, md_path_obj, img_dir_obj = self._validate_path(state)
        # 2.根据图片和md_content,获得图片的上下文
        # 2.1 如果没图片，不需要进行这一个节点的处理，直接返回
        if not img_dir_obj.exists():
            state["md_content"] = md_content
            return state
        # 2.2 有图片正常处理
        # list[tuple[str, Path, tuple[str, str, str]]] 图片名字，图片路径，[图片的前一个标题，上文，下文]
        img_context = self._get_img_context(md_content, img_dir_obj)

        # 3 送入多模态大模型生成摘要
        img_abstract = self._create_img_abstract(md_path_obj, img_context)
        print(img_abstract)

        # 4 替换上传minio获得新地址
        state["md_content"] = self._upload_img_minio(md_path_obj, md_content, img_context, img_abstract)

        return state

    # 内部方法1：获得md_content, img_path
    def _validate_path(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        1. 验证md_path之下存在文件
        2. 存在文件读取文件获得md_content
        3. 根据md_path路径拼出img_dir
        :param state:
        :return: Tuple[str, Path, Path]
        """
        self.log_step("开始读取md文件......")
        md_path_obj = Path(state.get("md_path"))
        # 校验文件存在性
        if not md_path_obj.exists():
            raise ImageProcessingError("md文件不存在")
        # 读取文件获得md_content
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        img_dir_obj = md_path_obj.parent / "images"
        self.log_step("获得md_content, img_dir")
        return md_content, md_path_obj, img_dir_obj

    # 内部方法2：获得图片上下文
    # 返回格式：图片名称,图片路径,(图片上一个标题,图片的上文,图片的下文)
    def _get_img_context(self, md_content, img_dir_obj) -> list[tuple[str, Path, tuple[str, str, str]]]:
        self.log_step("开始获取图片上下文")
        # 获得所有的图片列表
        img_list = os.listdir(img_dir_obj)
        # 结果列表
        target_img_list = []

        for img in img_list:
            # 判断图片是否是标准格式
            img_suffix = os.path.splitext(img)[-1]  # split也可以做，需要修改config里面的后缀去掉点
            # 获得配置文件之中的可以处理的图片格式
            config = get_config()
            if img_suffix not in config.image_extensions:
                # 不在范围内，不做处理，跳过这张图片
                continue
            # 进行处理获得上下文
            # 返回格式(图片的上一个标题，图片的上文，图片的下文)
            img_context = self._build_image_context(md_content, img)
            # 如果没有内容，跳过这个图片
            if not img_context:
                continue
            final_img_context = img_context[0]
            # 拼出所有的答案返回 (图片标题，图片地址，图片的上下文)
            target_img_list.append((img, img_dir_obj, final_img_context))

        return target_img_list

    # 内部方法2.1 获取图片的上下文
    # 返回内容：图片的上一个标题，图片上文，图片下文
    def _build_image_context(self, md_content, img_name) -> list[Tuple[str, str, str]]:
        # 正则匹配，找到图片对应的行
        # 匹配规则：!第一个[中间随意内容 第一个] 第一个(图片名字 第一个)
        re_pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(img_name) + r".*?\)")
        # 按照行批开md_content,排除空行
        md_lines = [line for line in md_content.split("\n") if line]
        # 结果
        img_context = []

        # 获得图片的行索引
        for idx, line in enumerate(md_lines):
            if not re_pattern.match(line):
                continue
            # 匹配到了图片位置
            former_tittle = ""
            former_index = 0
            for i in range(idx - 1, -1, -1):
                if not re.match(r"^#{1,6}\s+", md_lines[i]):
                    continue
                # 匹配到了上一个标题，赋值
                former_tittle = md_lines[i]
                former_index = i
                break
            former_content = md_lines[former_index + 1:idx]
            final_former_content = self._image_context_limit(former_content, "former")

            # later_tittle = ""
            later_index = len(md_lines)
            for j in range(idx + 1, len(md_lines)):
                if not re.match(r"^#{1,6}\s+", md_lines[j]):
                    continue
                # later_tittle = md_lines[j] # 其实不需要后面的标题
                later_index = j
                break
            later_content = md_lines[idx + 1:later_index]
            final_later_content = self._image_context_limit(later_content, "later")

            img_context.append((former_tittle, final_former_content, final_later_content))

        return img_context

    # 内部方法2.2 上下文进行截取
    # 核心逻辑：设定限制字符长度，遇到图片行或者超出字符则break，否则继续取
    def _image_context_limit(self, sub_str_content, sub_type) -> str:
        """
        对上下文内容进行截取
        :param sub_str_content:
        :param sub_type:
        :return:
        """
        final_content = []
        if sub_type == "former":
            sub_str_content.reverse()

        max_len = 200
        total = 0
        for line in sub_str_content:
            line = line.strip()  # 去掉制表符
            line_len = len(line)
            # 逻辑合并 如果是图片或者超出限制字符 break
            if total + line_len > max_len or re.match(r"^!\[.*?]\(.*?\)$", line):
                break
            # 遇到图片一样break
            # if re.match(r"^!\[.*?\]\(.*?\)$",line):
            #     break
            final_content.append(line)
            total += line_len
        if sub_type == "former":
            final_content.reverse()
        return "\n\n".join(final_content)

    # 内部方法3 进入多模态大语言模型生成摘要
    def _create_img_abstract(self, md_path, img_content):
        summaries = {}
        for img_name, img_dir, img_context in img_content:
            img_path = str(img_dir / img_name)
            # 获得一张图片的摘要
            summary = self._get_single_summary(md_path, img_path, img_context)
            # 字典增加键值对
            summaries[img_name] = summary

        return summaries

    # 内部方法3.1：获得一张图片的摘要
    def _get_single_summary(self, md_path, img_path, img_context):
        # 获得大模型的api和base
        config = get_config()

        api_key = config.openai_api_key
        url_base = config.openai_api_base

        former_tittle, former_content, later_content = img_context
        content = []
        if former_tittle:
            content.append(former_tittle)
        if former_content:
            content.append(former_content)
        if later_content:
            content.append(later_content)
        # 拼接成一个字符串
        final_content = "\n".join(content)

        # 图片内容需要读取成二进制
        with open(img_path, "rb") as f:
            image_data_str = base64.b64encode(f.read()).decode("utf-8")

        # 创建vlm客户端
        client = OpenAI(
            api_key=api_key,
            base_url=url_base,
        )

        document_title = md_path.stem
        messages = [
            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                             背景信息：
                                 1. 所属文档标题："{document_title}"
                                 2. 图片上下文：{final_content}
                                 请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                                 生成一个精准的中文标题（不要包含"图片"二字）。""",
                 },
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{image_data_str}"
                     }
                 }
             ]
             }
        ]

        response = client.chat.completions.create(
            model=config.vl_model,
            messages=messages,
        )

        # 获取结果
        summary = response.choices[0].message.content.strip()
        return summary

    # 内部方法4：把图片上传minio服务器,并且更新md_content
    def _upload_img_minio(self, md_path, md_content, img_context, img_abstract):
        remote_urls = {}
        client = get_minio_client()
        config = get_config()

        for img_name, img_path, _ in img_context:
            # 图片的本地地址
            img_file_path = str(img_path / img_name)
            # 构建上传minio服务器的路径和名字
            # a.jpg
            obj_name = f"{md_path.stem} / {img_name}"

            # 上传minio服务器
            client.fput_object(
                config.minio_bucket,
                obj_name,
                img_file_path
            )

            # 生成图片minio可以访问地址
            remote_url = ("http://"
                          + config.minio_endpoint
                          + "/" + config.minio_bucket
                          + "/" + obj_name
                          )
            remote_urls[img_name] = remote_url
        print(f"remote url: {remote_urls}")

        # 把新的content更新
        new_md_content = md_content
        for img_name, img_summary in img_abstract.items():
            remote_url = remote_urls.get(img_name)
            if not remote_url:
                continue

            replace_pattern = re.compile(
                r"!\[(.*?)]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE)

            new_md_content = replace_pattern.sub(f"![{img_summary}]({remote_url})", new_md_content)

        return new_md_content


if __name__ == '__main__':
    md_image_node = MdImgNode()
    state = {
        "md_path": r"E:\project\shopkeer_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册\hybrid_auto\hak180产品安全手册.md"
    }
    print(md_image_node.process(state))
