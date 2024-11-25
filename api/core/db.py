import contextlib
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError
from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger, LargeBinary, Boolean
from sqlalchemy.orm import relationship





user = "admin"
pwd = "11112222"
host = "database-1.cnok4wei8zqf.ap-northeast-2.rds.amazonaws.com"
port = 3306

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/db1?charset=utf8mb4"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()



# class Voice_Temp(Base):
#     __tablename__ = "voice_temp"
#     voice_id = Column(BigInteger, primary_key=True)
#     # voice_File = Column(String, unique=True, nullable=False)
#     voice_file = Column(LargeBinary)  # 파일 자체를 바이너리로 저장
#     voice_model_name = Column(String(255))
#     user_id = Column(String(255))

# class Model(Base):
#     __tablename__ = "voice_model"  
#     voice_model_id = Column(Integer, primary_key=True)
#     voice_model_name = Column(String(255), unique=True)
#     voice_model_file = Column(String(255))
#     user_entity = Column(Integer, ForeignKey("user.user_id"))

#     # User와의 관계 정의
#     user = relationship("User", back_populates="models")

# def inspect_table_columns():
#     # 테이블 검사 도구 생성
#     inspector = inspect(engine)
    
#     # voice_temp 테이블의 컬럼 정보 출력
#     try:
#         columns = inspector.get_columns('voice_model')
#         print("voice_model 테이블의 컬럼:")
#         for column in columns:
#             print(f"{column['name']} - {column['type']}")
#     except OperationalError as e:
#         print("테이블 또는 컬럼 정보 조회 실패:", e)

# def fetch_all_voice_temp_records():
#     # 세션 시작
#     session = SessionLocal()
#     try:
#         # Voice_Temp 테이블에서 모든 레코드 조회
#         records = session.query(Model).all()
#         print("voice_model 테이블의 레코드:")
#         for record in records:
#             print(record.__dict__)  # 각 레코드의 속성값 출력
#     except OperationalError as e:
#         print("레코드 조회 실패:", e)
#     finally:
#         session.close()

# def compare_model_and_db_columns():
#     # 테이블 검사 도구 생성
#     inspector = inspect(engine)
    
#     # 실제 voice_temp 테이블의 컬럼을 가져옵니다
#     db_columns = {col['name'] for col in inspector.get_columns('voice_model')}
#     print("데이터베이스 컬럼:", db_columns)

#     # Voice_Temp 모델의 컬럼을 가져옵니다
#     model_columns = {column.name for column in Model.__table__.columns}
#     print("모델 컬럼:", model_columns)

#     # 모델에만 있고 DB에 없는 컬럼
#     missing_in_db = model_columns - db_columns
#     print("모델에만 있고 DB에는 없는 컬럼:", missing_in_db)

#     # DB에만 있고 모델에는 없는 컬럼
#     missing_in_model = db_columns - model_columns
#     print("DB에만 있고 모델에는 없는 컬럼:", missing_in_model)

#         # 실제 voice_temp 테이블의 컬럼을 가져옵니다
#     db_columns = {col['name'] for col in inspector.get_columns('voice_temp')}
#     print("데이터베이스 컬럼:", db_columns)

#     # Voice_Temp 모델의 컬럼을 가져옵니다
#     model_columns = {column.name for column in Voice_Temp.__table__.columns}
#     print("모델 컬럼:", model_columns)

#     # 모델에만 있고 DB에 없는 컬럼
#     missing_in_db = model_columns - db_columns
#     print("모델에만 있고 DB에는 없는 컬럼:", missing_in_db)

#     # DB에만 있고 모델에는 없는 컬럼
#     missing_in_model = db_columns - model_columns
#     print("DB에만 있고 모델에는 없는 컬럼:", missing_in_model)
# # 모델과 DB의 컬럼을 비교하여 차이 출력
# compare_model_and_db_columns()



# # 테이블 컬럼 정보 확인
# inspect_table_columns()

# # voice_temp 테이블의 모든 레코드 출력
# fetch_all_voice_temp_records()







#@contextlib.contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()