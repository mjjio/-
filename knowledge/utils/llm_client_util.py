import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm_client(model_name:str=None,
                   temperature: float = 0.0,
                   response_format: bool = False):
    try:
        model_name = os.getenv("ITEM_MODEL")
        api_Key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")

        model_kwargs = {}
        if response_format:
            model_kwargs['response_format'] = {"type": "json_object"}

        client = ChatOpenAI(
            model_name= model_name,
            openai_api_key=api_Key,
            openai_api_base=base_url,
            # 下面参数可选的
            temperature=temperature,
            extra_body={"enable_thinking": False},
            model_kwargs=model_kwargs
        )
        return client
    except Exception as e:
        raise e

