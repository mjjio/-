import os
import logging
import threading
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_reranker_model = None

# 两把锁：一把管模型加载，一把管模型推理计算
_rerank_init_lock = threading.Lock()
_rerank_infer_lock = threading.Lock()


def get_reranker_model() -> FlagReranker:
    global _reranker_model
    try:
        if _reranker_model is None:
            # 第一把锁：防止多线程同时挤进去加载模型
            with _rerank_init_lock:
                if _reranker_model is None:
                    model_path = os.getenv("BGE_RERANKER_LARGE")
                    device = os.getenv("BGE_RERANKER_DEVICE", "cpu")
                    use_fp16 = os.getenv("BGE_RERANKER_FP16", "False").lower() == "true"

                    logger.info(f"正在初始化 Reranker 模型，路径: {model_path}, 设备: {device}, fp16: {use_fp16}")

                    # 【核心修复】：增加 low_cpu_mem_usage 和 device_map
                    # 强制关闭 transformers 的懒加载机制，彻底消灭 Meta Tensor 空指针 Bug！
                    _reranker_model = FlagReranker(
                        model_name_or_path=model_path,
                        device=device,
                        use_fp16=use_fp16,
                        low_cpu_mem_usage=False,
                        device_map=None
                    )

                    logger.info("Reranker 模型初始化成功！")

        return _reranker_model

    except Exception as e:
        logger.error(f"初始化 Reranker 模型失败: {e}", exc_info=True)
        return None


def compute_rerank_scores(pairs: list) -> list:
    """
    提供一个线程安全的推理入口
    """
    model = get_reranker_model()
    if not model:
        raise RuntimeError("Reranker 模型加载失败，无法计算得分")

    # 第二把锁：强制所有的计算任务排队，防止底层 PyTorch 显存分配冲突
    with _rerank_infer_lock:
        scores = model.compute_score(pairs)

        # 修复 FlagReranker 当 pairs 只有1对时返回 float 而不是 list 的 Bug
        if not isinstance(scores, list):
            scores = [scores]

        return scores


if __name__ == '__main__':
    query = "1956年发生了哪一事件标志着人工智能这一学科的正式诞生？"

    documents = [
        "1950 年，艾伦·图灵发表了其具有里程碑意义的论文《计算机与智能》。...",
        "1956 年的达特茅斯会议被认为是人工智能作为一个学科领域的诞生之地。...",
        "1951 年，英国数学家兼计算机科学家艾伦·图灵也开发出了首个用于下棋的程序。...",
    ]
    pairs = [(query, doc) for doc in documents]

    try:
        scores = compute_rerank_scores(pairs)
        print("计算得分:", scores)
    except Exception as e:
        print(f"执行失败: {e}")