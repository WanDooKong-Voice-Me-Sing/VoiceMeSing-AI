import uuid
from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.orm import Session
from core.db import get_db
from models import Model
from models import ModelResponse, ModelRequest

router = APIRouter()

@router.get("/")
def get_models():
    return {"message": "Models endpoint"}

@router.post("/", response_model=ModelResponse)
async def create_model_handler(request: ModelRequest, db: Session = Depends(get_db)):
    user_id = request.user_id
    model_id = generate_model_id()
    origin_voice = request.origin_voice

    # Perform the model training process
    trained_model_path = train_rvc_model(user_id, model_id, origin_voice)

    # Save the trained model to the database
    model = Model(user_id=user_id, model_id=model_id, trained_model_data=trained_model_path)
    db.add(model)
    db.commit()
    db.refresh(model)

    return ModelResponse(model_id=model_id, trained_model_path=trained_model_path)

def generate_model_id() -> str:
    return str(uuid.uuid4())

#DB save or Filesys save
def train_rvc_model(user_id: str, model_id: str, origin_voice: UploadFile) -> str:

    # rvc model train not implemented

    trained_model_path = f"/path/to/trained/models/{model_id}.model"
    return trained_model_path