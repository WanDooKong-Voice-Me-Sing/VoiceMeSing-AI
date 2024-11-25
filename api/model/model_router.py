from fastapi import APIRouter
from starlette import status
from api.model import model_schema
from api.worker.celery_worker import model_creation


router = APIRouter(
    prefix="/api/model"
)


@router.post("/create", status_code=status.HTTP_200_OK)
async def model_create(request: model_schema.ModelCreate):
    if not request.voiceId:
        return {"status": "error", "message": "voiceId is required."}
    task = model_creation.apply_async(args=[request.voiceId])   
    return {"status": "success", "message": f"Task queued for voice ID: {request.voiceId}"}

    


