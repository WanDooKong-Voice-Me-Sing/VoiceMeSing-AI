from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger
from sqlalchemy.orm import relationship

from core.db import Base
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
    user_Id = Column(BigInteger, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    nickname = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, nullable=False)

    # Model과 CoverSong에서 이 유저를 참조하는 관계를 정의
    models = relationship("Model", back_populates="user")
    cover_songs = relationship("CoverSong", back_populates="user")

class Voice_Temp(Base):
    __tablename__ = "voice_temp"
    voice_Id = Column(BigInteger, primary_key=True)
    original_Voice_File_Name = Column(String, unique=True, nullable=False)
    stored_Voice_File_Name = Column(String, unique=True, nullable=False)
    voice_File_Path = Column(String, unique=True, nullable=False)

class Model(Base):
    __tablename__ = "voice_model"  
    voice_Model_Id = Column(Integer, primary_key=True)
    voice_Model_Name = Column(String, unique=True, nullable=False)
    voice_Model_File_path = Column(String, nullable=False)
    #orinal_Voice_File_Name = Column(String, nullable=False)
    user_Entity = Column(Integer, ForeignKey("user.user_Id"), nullable=True)

    # User와의 관계 정의
    user = relationship("User", back_populates="models")
    cover_songs = relationship("CoverSong", back_populates="model")



class Song_Temp(Base):
    __tablename__ = "song_temp" 
    song_Id = Column(BigInteger,  primary_key=True)
    original_Song_File_Name  = Column(String, nullable=False)
    stored_Song_File_Name = Column(String, nullable=False)
    song_File_Path = Column(String, nullable=False)

class CoverSong(Base):
    __tablename__ = "coversong"
    coversong_id = Column(Integer, primary_key=True)
    coversong_path = Column(String, nullable=False)
    title = Column(String, nullable=False)
    model_id = Column(Integer, ForeignKey("voice_model.voice_Model_Id"), nullable=True)
    user_id = Column(Integer, ForeignKey("user.user_Id"), nullable=True)

    # Model과 User와의 관계 정의
    model = relationship("Model", back_populates="cover_songs")
    user = relationship("User", back_populates="cover_songs")