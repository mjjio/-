import os
from typing import List

from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 获取嵌入模型客户端对象
load_dotenv()

def get_bgem3_client():
    try:
        bge_m3 = BGEM3EmbeddingFunction(
            model_name=os.getenv("BGE_M3_PATH"),
            device="cpu",
            use_fp16=False,
            return_colbert_vecs=False
        )
        return bge_m3
    except Exception as e:
        raise e

##################################
def generate_hybrid_embeddings(embedding_model: BGEM3EmbeddingFunction,
                               embedding_documents: List[str]):
    """
    为文本生成向量嵌入
    :param embedding_model: 嵌入模型(这里使用BGEM3)
    :param embedding_documents: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    try:
        # 1. 生成嵌入
        embedding_result=(embedding_model.
                   encode_documents(embedding_documents))

        processed_sparse_result = []
        # 2. 遍历每一个文档
        for index in range(len(embedding_documents)):
            # 2.1 解构csr矩阵&获取稀疏向量
            csr_array = embedding_result['sparse']
            # a) 行索引
            ind_ptr = csr_array.indptr
            # b) 获取行索引的起始值
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index+1]
            # c) 获取token_id
            token_id=(csr_array.indices[start_ind_ptr:end_ind_ptr]
                                                        .tolist())
            # d) 获取权重
            weight=(csr_array.data[start_ind_ptr:end_ind_ptr]
                                                    .tolist())
            # 2.2 获取稀疏向量
            sparse_vector = dict(zip(token_id, weight))
            processed_sparse_result.append(sparse_vector)
        # 3. 返回
        return {
            "dense": [den.tolist() for den in embedding_result["dense"]],
            "sparse": processed_sparse_result
        }
    except Exception as e:
        return None