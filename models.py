from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger, LargeBinary, Boolean
from sqlalchemy.orm import relationship

from api.core.db import Base
# class User(Base):
#     __tablename__ = "user"

#     id = Column(Integer, primary_key=True)
#     username = Column(String, unique=True, nullable=False)
#     password = Column(String, nullable=False)
#     email = Column(String, unique=True, nullable=False)

# class Model(Base):
#     __tablename__ = "question"

#     model_id = Column(Integer, primary_key=True)
#     model_path = Column(String,nullable=False )
#     user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
#     user = relationship("User", backref="model")



# class CoverSong(Base):
#     __tablename__ = "coversong"

#     coversong_id = Column(Integer, primary_key=True)
#     audio_path = Column(String,nullable=False)
#     title = Column(String,nullable=False)
#     model_id = Column(Integer, ForeignKey("model.model_id"), nullable=True)
#     model = relationship("Model", backref="coversong")
#     user_id = Column(Integer, ForeignKey("user.id"), nullable=True)
#     user = relationship("User", backref="coversong_user")

class User(Base):
    __tablename__ = "user"
    user_id = Column(BigInteger, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    nickname = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)

    # Model과 CoverSong에서 이 유저를 참조하는 관계를 정의
    models = relationship("Model", back_populates="user")
    coversong = relationship("CoverSong", back_populates="user")


class Voice_Temp(Base):
    __tablename__ = "voice_temp"
    voice_id = Column(BigInteger, primary_key=True)
    voice_file = Column(LargeBinary)  # 파일 자체를 바이너리로 저장
    voice_model_name = Column(String(255))
    user_id = Column(String(255))


class Model(Base):
    __tablename__ = "voice_model"  
    voice_model_id = Column(Integer, primary_key=True)
    voice_model_name = Column(String(255), unique=True)
    voice_model_file = Column(String(255))
    user_id = Column(Integer, ForeignKey("user.user_id"))

    # User와의 관계 정의
    user = relationship("User", back_populates="models")



class Song_Temp(Base):
    __tablename__ = "song_temp" 
    song_id = Column(BigInteger,  primary_key=True)
    voice_model_id  = Column(String(255))
    result_song_name = Column(String(255))
    cover_song_file = Column(String(255))
    user_id = Column(String(255), unique=True, nullable=False)


class CoverSong(Base):
    __tablename__ = "cover_song"
    cover_song_id = Column(Integer, primary_key=True)
    cover_song_file = Column(LargeBinary)
    result_song_name = Column(String(255))
    is_public = Column(Boolean)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=True)

    # User와의 관계 정의
    user = relationship("User", back_populates="coversong")
