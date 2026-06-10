from modelscope import snapshot_download

str = snapshot_download(
    model_id="BAAI/bge-reranker-large",
    local_dir="E:/project/reranker_model"
)
print(str)