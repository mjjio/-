from functools import lru_cache

# 统一使用正确的 service 路径，去掉历史残留的 upload.
from knowledge.service.import_file_service import ImportFileService
from knowledge.service.task_service import TaskService
from knowledge.service.query_service import QueryService

@lru_cache
def get_task_service() -> TaskService:
    return TaskService()

@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService(get_task_service())

@lru_cache
def get_query_service() -> QueryService:
    return QueryService()