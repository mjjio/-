from modelscope import snapshot_download

str = snapshot_download(
    model_id="BAAI/bge-m3",
    local_dir="D:/mineru"
)
print(str)