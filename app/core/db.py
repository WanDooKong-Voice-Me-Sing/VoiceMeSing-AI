import contextlib
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

user = "admin"
pwd = "11112222"
host = "database-1.cjkegus2er4a.ap-northeast-2.rds.amazonaws.com"
port = 3306

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/db1?charset=utf8mb4"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread" : False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
#@contextlib.contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()