from fastapi import APIRouter, Depends, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
from starlette import status
from core.db import get_db
from api.model import model_schema, model_crud 
#from models import Model,Voice_Temp
import httpx

from infer_start import voice_extraction, preprocess_train, model_train, extraction_f0, train
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


router = APIRouter(
    prefix="/api/model"
)
import subprocess

#너무 오래 걸리면
#언제 끝나냐고 물어보면?
#하다가 오류나면? 어떤 오류?
#결국 만드려면 어느정도 시간 동안 목소리 파일 보관하고 있어야함
#소켓 통신 방식이 오히려 안좋을 것 같다 항상 열려 있을 이유가 없다
#비동기 작업 큐도 시험 해보면 좋을 듯 Celery
#@router.post("/create/{user_id}", status_code=status.HTTP_204_NO_CONTENT)

@router.post("/create", status_code=status.HTTP_204_NO_CONTENT)
def model_create(request: model_schema.ModelCreate, backgroundTasks: BackgroundTasks, db: Session = Depends(get_db)):

    # user_id = request.user_id
    # voice_id = request.voice_id
    model_name = request.voice_model_name
    # Voice_Temp = db.query(Voice_Temp).filter(Model.voice_Id == voice_id).first()
    # origin_voice = Voice_Temp.voice_File_Path
    origin_voice = request.voice_model_name
    voice_extraction(input=f"{origin_voice}/data",save_vocal=f"{origin_voice}/vocal",save_ins=origin_voice) #f로 받으면 중복 문제 있을 수 있음
    fin = train()
    # print("목소리 추출")
    # #voice_extraction(input=f"{origin_voice}/data",save_vocal=f"{origin_voice}/vocal",save_ins=origin_voice)
    
    # preprocess_train(trainset_dir = f"{origin_voice}/vocal",model_dir = model_name)
    
    # extraction_f0(model_dir=model_name)
    
    # print("전처리 완료")

    # print("모델 훈련시작")

    #model_train(trainset_dir = f"{origin_voice}/vocal",model_dir = model_name)
    
    # print("모델 학습완료")
    # trained_model_path="/abc/mart"

    # # model = Model(voice_Model_Name=model_name, voice_Model_File_path=trained_model_path, user_Entity=user_id)
    # db.add(model)
    # db.commit()
    # db.refresh(model)


    # 훈련 완료 후 알림을 보내는 부분
    # async with httpx.AsyncClient() as client:
    #     response = await client.post("https://3.36.63.85:8080", params={"status": "completed", "voice_model_id": model.voice_Model_Name})

    #     if response.status_code != 200:
    #         print("알림 전송 실패:", response.text)

    # return 0

    #subprocess 실험중
    # user_data = {1:"/app/app/source/data",2:"/app/app/source/vocal",3:"/app/app/source"}
    # arg1 = user_data[1]
    # arg2 = user_data[2]
    # arg3 = user_data[3]
    # subprocess.Popen([
    #     "python3", arg1, arg2, arg3,

    # ])

    
    
    #extraction_f0(trainset_dir = "/app/source/vocal",model_dir = "test_1")



    #return {"model_id": db_model.id, "model_path": db_model.model_path}

