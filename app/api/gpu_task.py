import asyncio
import httpx
from core.db import get_db
from models import Model, CoverSong, Voice_Temp
from infer_start import voice_extraction,train,coversong_train, mixing


gpu_task_queue = asyncio.Queue()
task_results = {}

async def process_gpu_queue():
    while True:
        task_id, task_type, request, db = await gpu_task_queue.get()
        try:
            if task_type == "model_creation":
                # 모델 생성 로직
                user_id = request.user_id
                voice_id = request.voice_id
                model_name = request.voice_model_name
                voice_temp = db.query(Voice_Temp).filter(Voice_Temp.voice_Id == voice_id).first()
                origin_voice = voice_temp.voice_File_Path
                voice_extraction(input=f"{origin_voice}/data", save_vocal=f"{origin_voice}/vocal", save_ins=origin_voice)
                Temp = train(exp_dir1=model_name, trainset_dir4=origin_voice)
                trained_model_path = "/abc/mart"
                model = Model(voice_Model_Name=model_name, voice_Model_File_path=trained_model_path, user_Entity=user_id)
                db.add(model)
                db.commit()
                db.refresh(model)

                async with httpx.AsyncClient() as client:
                    await client.post("https://3.36.63.85:8080", params={"status": "completed", "voice_model_id": model.voice_Model_Name})

                task_results[task_id] = {"status": "completed", "voice_model_id": model.voice_Model_Name}

            elif task_type == "cover_song_creation":
                # 커버송 생성 로직
                title = request.title
                origin_song = request.audio_path
                user_id = request.user_id
                model_id = request.model_id

                model = db.query(Model).filter(Model.model_id == model_id, Model.user_id == user_id).first()
                model_name = model.voice_Model_Name
                coversong_train(sid0=f"{model_name}.pth", input_audio_path=f"{origin_song}.mp3", index_path="")
                output_path = "/app/result/Cover/output.wav"
                mixing("/app/result/song/output_audio.wav", "/app/source/inst/instrument.wav", output_path)

                cover_song = CoverSong(model_id=model_id, user_id=user_id, audio_path=output_path, title=title)
                db.add(cover_song)
                db.commit()
                db.refresh(cover_song)

                async with httpx.AsyncClient() as client:
                    await client.post("http://backend-server-url/notify", json={"status": "completed", "cover_song_id": cover_song.coversong_id})

                task_results[task_id] = {"status": "completed", "cover_song_id": cover_song.coversong_id}

        except Exception as e:
            task_results[task_id] = {"status": "failed", "error": str(e)}

        finally:
            gpu_task_queue.task_done()
