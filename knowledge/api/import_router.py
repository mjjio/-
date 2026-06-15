import os.path
import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 修正：移除所有 .upload.utils. 相关错误路径
from knowledge.service.import_file_service import ImportFileService
from knowledge.service.task_service import TaskService
from knowledge.core.paths import get_front_page_dir
from knowledge.core.deps import get_task_service, get_import_file_service
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.schema.upload_schema import UploadResponse

def create_app() -> FastAPI:
    app = FastAPI(description="知识库导入")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app

def register_router(app: FastAPI):
    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/import")
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                 service: ImportFileService = Depends(get_import_file_service)):
        task_id, file_dir, import_file_path = service.process_upload_file(file)
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)
        return UploadResponse(message="文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(task_id: str, task_service: TaskService = Depends(get_task_service)):
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)

if __name__ == '__main__':
    setup_logging()
    uvicorn.run(app=create_app(), port=8000, host="0.0.0.0")