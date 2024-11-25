from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os
from api.model import model_router
from api.coversong import coversong_router
from api.Redis import process_task
import uvicorn
import asyncio
app = FastAPI()

os.environ["CUDA_VISIBLE_DEVICES"] = ""

origins = [ 
           "http://127.0.0.1:5173",
]

# CORS 설정

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함
app.include_router(model_router.router)
app.include_router(coversong_router.router)

@app.on_event("startup")
async def start_process_task():
    """서버 시작 시에 process_task를 비동기적으로 실행"""
    global task
    task = asyncio.create_task(process_task())

@app.on_event("shutdown")
async def stop_process_task():
    """서버 종료 시에 process_task를 취소"""
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, log_level="info", workers=1)
