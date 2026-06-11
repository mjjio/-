from functools import lru_cache
from knowledge.upload.service.import_file_service import ImportFileService
from knowledge.upload.service.task_service import TaskService
from knowledge.upload.service.query_service import QueryService


@lru_cache
def get_task_service() -> TaskService:
    return TaskService()

@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService(get_task_service())

@lru_cache
def get_query_service() -> QueryService:
    return QueryService()