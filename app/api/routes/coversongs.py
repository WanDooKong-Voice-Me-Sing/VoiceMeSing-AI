from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from core.db import get_db
from models import CoverSongRequest, Model, CoverSong

router = APIRouter()

@router.get("/")
def get_coversongs():
    return {"message": "Coversongs endpoint"}

@router.post("/", response_model=Model)
async def create_cover_song_handler(
    request: CoverSongRequest,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)):

    user_id = request.user_id
    model_id = request.model_id
    origin_song = audio_file.file.read()

    # Retrieve trained model from database
    model = db.query(Model).filter(Model.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Perform cover song creation logic here (example)
    cover_song_data = create_cover_song(user_id, model, origin_song)

    # Save the cover song to database
    cover_song = CoverSong(model_id=model_id, user_id=user_id, audio_data=cover_song_data, title="ANY") #If you need a new title
    db.add(cover_song)
    db.commit()
    db.refresh(cover_song)

    return cover_song.cover_song_data


def create_cover_song(user_id: int, model_id: str, origin_song: bytes) -> str:
    # RVC model training example (not implemented)
    trained_model_path = f"/path/to/trained/models/{model_id}.pth"

    # make_cover_song (not implemented)
    cover_song_data = 'not implemented'
    return cover_song_data
