import redis
import time
from fastapi import BackgroundTasks, Depends, HTTPException
from .core.db import get_db
from sqlalchemy.orm import Session
from models import Model,Voice_Temp, CoverSong, Song_Temp
from infer_start import voice_extraction, train, save_origin_music, save_cover_music, coversong_train, mixing

r = redis.StrictRedis(host='localhost', port=6379, db=0)

async def process_task(db: Session = Depends(get_db)):
    while True:
        
        task = r.lpop('task queue')
        if task:
            print("처리")
            task_type, id = task.decode("utf-8").split(":")
            if task_type == "create_model":
                print("model")
                voice_id = id
                model_creation(voice_id, db)
            elif task_type == "create_cover_song":
                song_id = id
                coversong_creation(song_id, db)
            
        else:
            print("NO taks in queue")
            time.sleep(3)


async def model_creation(voice_id,  db: Session = Depends(get_db)):
    print("처리중")
    Voice = db.query(Voice_Temp).filter(Voice_Temp.voice_id == int(voice_id)).first()

    model_name =  Voice.voice_model_name
    user_id = Voice.user_id
    origin_voice = Voice.voice_file 
    
    path = save_origin_music(user_id, origin_voice, model_name)

    await voice_extraction(input=f"{path}/origin",save_vocal=f"{path}/vocal",save_ins=f"{path}/inst") 
    
    Temp = await train(exp_dir1=model_name, trainset_dir4=f"{path}/vocal") 
    

    model = Model(voice_model_name=model_name, voice_model_file=model_name, user_id=user_id)
    db.add(model)
    db.commit()
    db.refresh(model)




async def coversong_creation(song_id, db: Session = Depends(get_db), ):
    song_id = int(song_id)
    coversong = db.query(Song_Temp).filter(Song_Temp.song_id == song_id).first()


    model_id = coversong.voice_model_id
    song_name = coversong.result_song_name
    origin_cover = coversong.cover_song_file
    user_id = coversong.user_id

    origin_song = save_cover_music(user_id, origin_cover)


    model = db.query(Model).filter(
    Model.voice_model_id == model_id,
    Model.user_id == user_id
    ).first()
    

    model_name = model.voice_model_name


    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    
    await coversong_train(sid0=f"{model_name}.pth", user_id=user_id, model_id=model_id, input_audio_path=f"{origin_song}/{user_id}_{song_name}.mp3", index_path="")
    #output_path=""


    result = await mixing(f"result/{user_id}/{model_id}_output_audio.wav",f"source/{user_id}/inst/instrument_{user_id}_{model_name}.mp3_10.wav",f"result/{user_id}/Cover_{model_name}.wav")



    cover_song = CoverSong(cover_song_file=result.getvalue(), result_song_name=song_name,is_public=True, user_id=user_id) #If you need a new title
    db.add(cover_song)
    db.commit()
    db.refresh(cover_song)

