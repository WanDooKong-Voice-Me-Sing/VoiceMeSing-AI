from sqlalchemy.orm import Session
from api.coversong.coversong_schema import CoverSongCreate
from models import User,CoverSong,Model

def create_coversong(db:Session, coversong_create: CoverSongCreate,
                     user: User, model: Model):
    db_coversong = CoverSong(user=user,
                             model=model,
                             audio_path=coversong_create.audio_path, #이쪽에서 저장하는게 아니라 저장된거 쓰는거임....백엔드에서 user 테이블에 넣어주던지...
                             title=coversong_create.title)
    db.add(db_coversong)
    db.commit()

def get_coversong(db: Session, coversong_id: int):
    return db.query(CoverSong).get(coversong_id)

def delete_coversong(db: Session, db_coversong: CoverSong):
    db.delete(db_coversong)
    db.commit()

