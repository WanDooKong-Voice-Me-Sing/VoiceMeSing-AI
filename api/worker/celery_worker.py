
from celery import current_task
from celery.utils.log import get_task_logger
from .celery_app import celery_app

from api.core.db import get_db
from sqlalchemy.orm import Session
from api.core.db import SessionLocal

from models import Model,Voice_Temp, CoverSong, Song_Temp
from infer_start import voice_extraction, train, save_origin_music, save_cover_music, coversong_train, mixing

logger = get_task_logger(__name__)


@celery_app.task
def model_creation(voice_id: int):
    db = SessionLocal()
    try:
        # 1. Voice_Temp에서 해당 데이터 조회
        voice = db.query(Voice_Temp).filter(Voice_Temp.voice_id == voice_id).first()
        if not voice:
            return {"status": "error", "message": f"Voice ID {voice_id} not found."}

        model_name = voice.voice_model_name
        user_id = voice.user_id
        origin_voice = voice.voice_file

        # 2. 파일 저장 및 경로 생성
        path = save_origin_music(user_id, origin_voice, model_name)

        # 3. 음성 추출 작업
        try:
            voice_extraction(
                input=f"{path}/origin",
                save_vocal=f"{path}/vocal",
                save_ins=f"{path}/inst"
            )
        except Exception as e:
            return {"status": "error", "message": f"Voice extraction failed: {str(e)}"}

        # 4. 모델 학습
        try:
            Temp = train(exp_dir1=model_name, trainset_dir4=f"{path}/vocal")
        except Exception as e:
            return {"status": "error", "message": f"Model training failed: {str(e)}"}

        # 5. 모델 정보 저장
        model = Model(voice_model_name=model_name, voice_model_file=model_name, user_id=user_id)
        db.add(model)
        db.commit()
        db.refresh(model)

        return {"status": "success", "message": f"Model task completed for Voice ID {voice_id}"}
    except Exception as e:
        db.rollback()
        raise e  
    finally:
        db.close()

@celery_app.task
def coversong_creation(song_id: int):
    db = SessionLocal() 
    try:
        # 1. Song_Temp에서 song_id로 데이터 조회
        song_id = int(song_id)
        coversong = db.query(Song_Temp).filter(Song_Temp.song_id == song_id).first()
        if not coversong:
            return {"status": "error", "message": f"Song ID {song_id} not found."}

        model_id = coversong.voice_model_id
        song_name = coversong.result_song_name
        origin_cover = coversong.cover_song_file
        user_id = coversong.user_id

        # 2. 원본 커버 음악 저장
        origin_song = save_cover_music(user_id, origin_cover)

        # 3. Model 테이블에서 모델 조회
        model = db.query(Model).filter(
            Model.voice_model_id == model_id,
            Model.user_id == user_id
        ).first()
        if not model:
            return {"status": "error", "message": "Model not found."}

        model_name = model.voice_model_name

        # 4. 커버송 생성
        try:
            coversong_train(
                sid0=f"{model_name}.pth",
                user_id=user_id,
                model_id=model_id,
                input_audio_path=f"{origin_song}/{user_id}_{song_name}.mp3",
                index_path=""
            )
        except Exception as e:
            return {"status": "error", "message": f"Coversong training failed: {str(e)}"}

        # 5. 믹싱 작업
        try:
            result_path = mixing(
                f"result/{user_id}/{model_id}_output_audio.wav",
                f"source/{user_id}/inst/instrument_{user_id}_{model_name}.mp3_10.wav",
                f"result/{user_id}/Cover_{model_name}.wav"
            )
        except Exception as e:
            return {"status": "error", "message": f"Mixing failed: {str(e)}"}

        # 6. 데이터베이스에 커버송 저장
        cover_song = CoverSong(
            cover_song_file=result_path,
            result_song_name=song_name,
            is_public=True,
            user_id=user_id
        )
        db.add(cover_song)
        db.commit()
        db.refresh(cover_song)

        return {"status": "success", "message": f"Cover song task completed for song_id {song_id}"}

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()