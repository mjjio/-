import os
from minio import Minio

# 获取minio客户端连接对象的方法
def get_minio_client():
    try:
        client = Minio(
            # MinIO 服务端点
            endpoint=os.getenv("MINIO_ENDPOINT"),
            # 用户名
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            # 密码
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            # 不支持https方式，也就是http访问
            secure=False,
        )
        # 创建bucket  client.make_bucket()
        return client
    except Exception as e:
        print(e)
        raise e

