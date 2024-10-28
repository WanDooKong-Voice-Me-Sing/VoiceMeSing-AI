from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os
import sys
from api.model import model_router
from api.coversong import coversong_router
import torch


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
if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, log_level="info", workers=1)
    