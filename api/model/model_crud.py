from sqlalchemy.orm import Session

from api.model.model_schema import ModelCreate
from models import User,CoverSong,Model

#modelid는 그냥 db 번호로
def create_model(db:Session, model_create: ModelCreate, user: User):
    db_model = Model(user=user,
                     model_path=model_create.model_path,
                     )
    db.add(db_model)
    db.commit()

def get_model(db: Session, model_id: int):
    return db.query(Model).get(model_id)

def delete_model(db: Session, db_model: Model):
    db.delete(db_model)
    db.commit()
