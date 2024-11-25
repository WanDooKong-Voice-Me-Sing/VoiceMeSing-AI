from sqlalchemy.orm import Session

from models import User


def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()
