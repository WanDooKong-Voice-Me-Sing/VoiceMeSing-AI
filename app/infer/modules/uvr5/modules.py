import os
import traceback
import logging

logger = logging.getLogger(__name__)

import ffmpeg
import torch

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

config = Config()

def uvr(
    model_name,  # 기본 모델 이름
    inp_root,  # 기본 보컬 위치 디렉토리
    save_root_vocal,  # 기본 보컬 아웃풋 디렉토리
    paths,  # 파일 직접 입력 파일 객체를 처리해야함
    save_root_ins,  # 기본 보컬 제외 아웃풋 디렉토리
    agg,  # 기본 조절 값
    format0  # 기본 포맷
):
    print("uvr 함수가 호출되었습니다")
    infos = []
    print("실행중")
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )
        is_hp3 = "HP3" in model_name
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    'ffmpeg -i "%s" -vn -acodec pcm_s16le -ac 2 -ar 44100 "%s" -y'
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                try:
                    if done == 0:
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    yield "\n".join(infos)
                except:
                    infos.append(
                        "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                    )
                    yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
    yield "\n".join(infos)
