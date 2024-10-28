import os
import sys
from dotenv import load_dotenv

#현재 작업 디렉토리 가져옴
now_dir = os.getcwd()
# 현재 디렉토리 내의 모듈 쉽게 임포트
sys.path.append(now_dir)
#환경변수 사용가능하게
print(now_dir)
load_dotenv()

from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)

from configs.config import Config

from sklearn.cluster import MiniBatchKMeans
import torch, platform
import numpy as np
import faiss
import fairseq
import pathlib
import json
from time import sleep


from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
import subprocess
torch.cuda.set_per_process_memory_fraction(1.0, 0)  # 첫 번째 GPU의 메모리의 50%를 사용 추가한거임
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #추가한거임



torch.cuda.empty_cache()
#이름이 "numba"인 로거(Logger) 객체를 가져옵니다. 이 로거는 numba 라이브러리에서 발생하는 로그 메시지를 처리
logging.getLogger("numba").setLevel(logging.WARNING)
#해당 로거의 로그 레벨을 WARNING으로 설정합니다. 이 설정은 경고 이상의 심각성을 가진 로그 메시지만
logging.getLogger("httpx").setLevel(logging.WARNING)

#현재 모듈의 로거 가져오기
logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
#TEMP 디렉토리와 그 안의 모든 내용을 삭제합니다. ignore_errors=True는 디렉토리가 존재하지 않는 경우에도 에러를 무시
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
#임시파일 저장 위치
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
#PyTorch의 랜덤 시드를 설정합니다
torch.manual_seed(114514)
#클래스 인스턴스 생성 Config는 일반적인 설정, VC 변환 작업 관련 클래스
config = Config()
vc = VC(config)


if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

# 원래 GradMultiply.forward 메서드 대신 forward_dml 함수가 호출
    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "4060",
                "L",
                "6000",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "그래픽카드 없음"
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

#환경변수에서 특정 경로 읽어서 변수 저장
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []

#특정 디렉토리에서 인덱스 파일과 모델 가중치 파일을 찾고, 그 경로들을 저장
def lookup_indices(index_root):
    global index_paths
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))


lookup_indices(index_root)
lookup_indices(outside_index_root)
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

#pth파일과 index 파일 목록choices 키에 저장하기 gui용
def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }

#gui 용
def clean():
    return {"value": "", "__type__": "update"}

#onnx용인데 일단 필요 x
def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

#p라는 프로세스의 상태 확인 p.poll이 None이면 종료되지 않았음을 의미
def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None종료 안된거
        # 전부 종료할때까지 반복
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    # SR 설정 및 디렉토리 생성
    sr = sr_dict[sr]
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    
    log_path = f"{now_dir}/logs/{exp_dir}/preprocess.log"
    with open(log_path, "w") as log_file:
        log_file.write("로그 파일 초기화 완료\n")
    
    # 명령어 생성
    cmd = f'"{config.python_cmd}" infer/modules/train/preprocess.py "{trainset_dir}" {sr} {n_p} "{now_dir}/logs/{exp_dir}" {config.noparallel} {config.preprocess_per}'
    
    logger.info(f"Execute: {cmd}")
    
    # 명령어 실행 및 로그 기록
    with open(log_path, "a") as log_file:
        process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
        done = [False]
        threading.Thread(target=if_done, args=(done, process)).start()
    
    # 프로세스 완료 여부 확인 및 로그 업데이트
    while True:
        with open(log_path, "r") as log_file:
            print(log_file.read())  # 터미널에 출력하거나 다른 방식으로 처리 가능
            
        sleep(1)
        if done[0]:
            break

    # 최종 로그 출력
    with open(log_path, "r") as log_file:
        final_log = log_file.read()
    logger.info(final_log)
    print(final_log)
        
# f0 추출 함수
# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    log_path = "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir)
    
    with open(log_path, "w")as log_file:
        log_file.write("로그 파일 초기화 완료\n")
        
    if if_f0:
        if gpus_rmvpe != "-":
            if isinstance(gpus_rmvpe, str):
                gpus_rmvpe = gpus_rmvpe.split("-")
            leng = len(gpus_rmvpe)
            ps = []
            for idx, n_g in enumerate(gpus_rmvpe):
                cmd = (
                    '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                    % (
                        config.python_cmd,
                        leng,
                        idx,
                        n_g,
                        now_dir,
                        exp_dir,
                        config.is_half,
                    )
                )
                logger.info("Execute: " + cmd)
                
                # 비동기로 프로세스 실행
                p = subprocess.Popen(cmd, shell=True, cwd=now_dir)
                ps.append(p)

            # 모든 프로세스가 완료될 때까지 대기
            for p in ps:
                p.wait()  # 각 프로세스가 종료될 때까지 대기

            # 모든 작업이 끝나면 다음 단계로 진행
            done = [False]
            threading.Thread(
                target=if_done_multi,  #
                args=(done, ps),
            ).start()

            torch.cuda.empty_cache()  # 캐시 지우기
            print("모든 f0 추출 완료")            
            
            
            # for idx, n_g in enumerate(gpus_rmvpe):
            #     cmd = (
            #         '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
            #         % (
            #             config.python_cmd,
            #             leng,
            #             idx,
            #             n_g,
            #             now_dir,
            #             exp_dir,
            #             config.is_half,
            #         )
            #     )
            #     logger.info("Execute: " + cmd)
            #     p = Popen(
            #         cmd, shell=True, cwd=now_dir
            #     )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
            #     ps.append(p)
            # done = [False]
            # threading.Thread(
            #     target=if_done_multi,  #
            #     args=(
            #         done,
            #         ps,
            #     ),
            # ).start()

    # 서로 다른 부분(part)을 각각 별도의 프로세스로 실행
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        logger.info("Execute: " + cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)

    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()

#프리 트레인 모델 파일 확인 및 반환
def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )

#~k 할건지
def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    
    # filelist 생성
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    print("리스트 생성완료")
    # 파일 경로와 관련된 정보 저장
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    print("경로 저장 완료")
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # config생성#config 불필요 (실제로는 불필요한 작업) 모델학습을 위한 파일리스트 준비, 구성파일저장, gpu, 프리트레인 모델 정보 로깅
    #cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    print("gpu관련 로그 설정")
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    print(config) 
    ####################################       
    if gpus16:
        print("gpu사용")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated()} bytes")
            print(f"총 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory} bytes")
        else:
            print("GPU 사용 불가능, CPU로 작업 중입니다.")
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == "예" else 0,
                1 if if_cache_gpu17 == "예" else 0,
                1 if if_save_every_weights18 == "예" else 0,
                version19,
            )
        )


    # else:
    #     cmd = (
    #         '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
    #         % (
    #             config.python_cmd,
    #             exp_dir1,
    #             sr2,
    #             1 if if_f0_3 else 0,
    #             batch_size12,
    #             total_epoch11,
    #             save_epoch10,
    #             "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
    #             "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
    #             1 if if_save_latest13 == i18n("是") else 0,
    #             1 if if_cache_gpu17 == i18n("是") else 0,
    #             1 if if_save_every_weights18 == i18n("是") else 0,
    #             version19,
    #         )
    #     )
    logger.info("Execute: " + cmd) # 명령어 실행전에 로그에 기록 어떤거 실행했는지 확인
    p = Popen(cmd, shell=True, cwd=now_dir) # 명령어를 새 프로세스에서 실행 shell
    p.wait()# 명령어 실행완료시까지 대기
    print("모델 훈련성공")
    return "훈련완료, 콘솔에서 훈련 로그를 확인하거나 실험 폴더 아래의 train.log 파일을 확인 가능"


# but4.click(train_index, [exp_dir1], info3)데이터 준비
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)로그파일 저장 디렉터리
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    #디렉터리 존재 확인
    if not os.path.exists(feature_dir):
        return "feature추출을 선행하세요!"
    listdir_res = list(os.listdir(feature_dir))
    #디렉터리 안의 파일 확인
    if len(listdir_res) == 0:
        return "feature추출을 선행하세요！"
    print("준비완료")
    infos = []
    npys = []
    #리스트의 모들 파일을 로드하여 배열에추가
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    #인덱스를 셔플링하여 데이터 섞기
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    print(big_npy.shape[0])
    #k-means클러스터링 데이터의 크기가 200,000개의 샘플초과시 10,000의 클러스터 중심으로 축소
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        print ("\n".join(infos))
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
            print("클러스터 축소 완료")
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            print("클러스터 축소 실패")
            print ("\n".join(infos))

   
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    #Inverted File 리스트의 개수 설정
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    print ("\n".join(infos))
    #벡터인덱스를 생성, 벡터의 차원수 지정 , n_ivf개의 ivf리스트, Flat IVF 리스트 내에서 직접 벡터를 저장
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    print ("\n".join(infos))
    #faiss인덱스에서 IVF인덱스 부분만 추출(검색빠름)
    index_ivf = faiss.extract_index_ivf(index)
    #검색 시 얼마나 많은 IVF리스트를 탐색할지 결정
    index_ivf.nprobe = 1
    #추출된 벡터를 이용하여 FAISS인덱스 훈련 : IVF리스트를 생성하고, 벡터들을 적절한 리스트에 할당하는 과정
    index.train(big_npy)
    #훈련된 인덱스를 경로에 저장
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("adding")
    print ("\n".join(infos))
    batch_size_add = 8192
    #배열을 8192개의 벡터씩 잘라서 FAISS인덱스에 추가, 메모리 사용량을 줄임
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "인덱스를 성공적으로 생성했습니다. added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    #저장된 인덱스 파일에 대해 외부 디렉터리에 링크를 생성하는 작업, 하드링크나 심볼링링크
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(
            "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
            "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (
                outside_index_root,
                exp_dir1,
                n_ivf,
                index_ivf.nprobe,
                exp_dir1,
                version19,
            ),
        )
        infos.append("인덱스를 외부로 연결하는데 성공하였습니다.-%s" % (outside_index_root))
    except:
        infos.append("인덱스 외부연결 실패-%s" % (outside_index_root))

    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    print("\n".join(infos))

# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    def process_steps(trainset_dir4, exp_dir1, sr2, np7, gpus16, f0method8, if_f0_3, version19, gpus_rmvpe, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, if_cache_gpu17, if_save_every_weights18):
        # step1: 데이터 처리
        get_info_str("step1: 데이터를 처리 중입니다")
        result = preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)
        print("preprocess_dataset result:", result)  # 반환 값 확인
        if result is not None:
            [get_info_str(_) for _ in result]
        else:
            print("preprocess_dataset returned None")
        print("1번")
        # step2: 음높이(F0) 추출 및 음성 데이터 처리
        get_info_str("step2: 음높이(F0)와 특징을 추출하는 중입니다.")
        results = extract_f0_feature(gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe)
        print("extract_f0_feature result:", results)  # 반환 값 확인
        if results is not None:
            [get_info_str(_) for _ in results]
        else:
            print("extract_f0_feature returned None")
        print("2번")
        # step3a: 모델 훈련
        get_info_str("step3a: 모델 훈련 중입니다.")
        click_train(
            exp_dir1,
            sr2,
            if_f0_3,
            spk_id5,
            save_epoch10,
            total_epoch11,
            batch_size12,
            if_save_latest13,
            pretrained_G14,
            pretrained_D15,
            gpus16,
            if_cache_gpu17,
            if_save_every_weights18,
            version19,
        )
        get_info_str("훈련이 끝났습니다. 콘솔 훈련 로그 또는 실험 폴더의 train.log 파일을 확인할 수 있습니다.")
        print("여기까지 오케이")
        # step3b: 인덱스 훈련 단계
        [get_info_str(_) for _ in train_index(exp_dir1, version19)]
        get_info_str("모든 실행이 종료되었습니다.")

        # 최종 결과 반환
        return get_info_str("모든 처리가 완료되었습니다.")

        # 함수 호출 예시 (필요한 인자를 넣어서 호출)
    result = process_steps(trainset_dir4, exp_dir1, sr2, np7, gpus16, f0method8, if_f0_3, version19, gpus_rmvpe, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, if_cache_gpu17, if_save_every_weights18)
    print(result)
    # infos = []
    # print("1번")
    # def get_info_str(strr):
    #     infos.append(strr)
    #     return "\n".join(infos)
    # print("2번")
    # # step1:데이터 처리
    # yield get_info_str("step1:데이터를 처리중입니다")
    # [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]
    # print("3번")
    # # step2a:음높이 추출 , 음성데이터 처리, 음높이와 다른 특징 추출
    # yield get_info_str("step2:음높이(F0)와 특징을 추출하는 중입니다.")
    # [
    #     get_info_str(_)
    #     for _ in extract_f0_feature(
    #         gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
    #     )
    # ]

    # # step3a:모델훈련
    # yield get_info_str("step3a:모델훈련")
    # click_train(
    #     exp_dir1,
    #     sr2,
    #     if_f0_3,
    #     spk_id5,
    #     save_epoch10,
    #     total_epoch11,
    #     batch_size12,
    #     if_save_latest13,
    #     pretrained_G14,
    #     pretrained_D15,
    #     gpus16,
    #     if_cache_gpu17,
    #     if_save_every_weights18,
    #     version19,
    # )
    # yield get_info_str(
    #     "훈련이 끝났습니다. 콘솔 훈련 로그 또는 실험 폴더의 train.log 파일을 확인할 수 있습니다"
    # )


    # # step3b:모델훈련, 인덱스 훈련단계
    # [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    # yield get_info_str("모든 실행 종료")











#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
#체크포인트 파일 경로를 입력받고, 로그 파일을 읽어서 해당하는 정보 추출
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

F0GPUVisible = config.dml == False

def change_f0_method(f0method8):
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}




#훈련 (사용자의 목소리)
#step1  기본 설정
# try:
#     info1 = preprocess_dataset(
#         trainset_dir="/app/source/vocal",  # 목소리 데이터 주소
#         exp_dir="user_2",  # 목소리 모델의 이름 설정 가능
#         sr="40k",  # 샘플링 레이트 40 or 48 (48000Hz 이상이면)
#         n_p=11,  # CPU 코어 수
#     )

#     print("preprocess_dataset 실행 성공")
# except Exception as e:
#     print(f"preprocess_dataset 호출 중 예외 발생: {e}")
    

# # step2a - 피처 추출
# try:
#     extract_f0_feature(
#         gpus="0",  # GPU 선택 (예: 0)
#         n_p=11,  # 학습에 사용할 CPU 코어 수
#         f0method="rmvpe_gpu",  # 추출 방법: "pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"
#         if_f0=True,  # 피치 포함 여부 (FALSE면 국어책 읽기)
#         exp_dir="myTest",
#         version19="v2",  # v1 또는 v2 (기본값)
#         gpus_rmvpe="%s-%s" % (gpus, gpus), # GPU 병렬 처리 방법, 기본값
#     )
#     print("extract_f0_feature 실행 성공")
# except Exception as e:
#     print(f"extract_f0_feature 호출 중 예외 발생: {e}")


# # step3 - 모델 훈련
# try:
#     info3 = click_train(
#         exp_dir1="user_2",
#         sr2="40k",
#         if_f0_3=True,  # 피치 포함 여부 (FALSE면 국어책 읽기)
#         spk_id5='0',  # 보컬 ID, 기본값 사용
#         save_epoch10='30',  # 학습 중간 저장 빈도
#         total_epoch11='15',  # 전체 epoch
#         batch_size12='3',  # 배치 사이즈
#         if_save_latest13="NO",  # 마지막 ckpt 파일만 저장할지 여부 (디스크 용량 문제)
#         pretrained_G14="/app/assets/pretrained_v2/f0G48k.pth",  # Pretrain 제너레이터
#         pretrained_D15="/app/assets/pretrained_v2/f0D48k.pth",  # Pretrain discriminator
#         gpus16="0",
#         if_cache_gpu17="NO",  # GPU에 트레이닝 셋 캐싱 (속도 증가, 메모리 많이 씀)
#         if_save_every_weights18="NO",  # 세이브 지점마다 모델 생성 여부
#         version19="v2",  # v1 또는 v2 (기본값)
#     )
#     print("click_train 실행 성공")
# except Exception as e:
#     print(f"click_train 호출 중 예외 발생: {e}")


#step3 - 피처 인덱스 훈련
# try:
#     info3 = train_index(
#         "user_2",
#         "v2",
#     )
#     print("train_index 실행 성공")
# except Exception as e:
#     print(f"train_index 호출 중 예외 발생: {e}")

# print("함수 호출 전")
# try:
#     vc_output4 = uvr(model_name=uvr5_names[2], inp_root="/app/source/data", save_root_vocal="/app/source/vocal", paths= "", save_root_ins="/app/source", agg=10, format0="wav")
#     # 결과 출력
#     for output in vc_output4:
#         print(output)
# except Exception as e:
#     print(f"함수 호출 중 예외 발생: {e}")

# print(f"Selected model: {uvr5_names[2]}")
# print(ngpu)

# 목소리 파일 추출하기
def voice_extraction(input, save_vocal, save_ins):
    print("함수 호출 전")
    try:
        vc_output4 = uvr(model_name=uvr5_names[2], inp_root=input, save_root_vocal=save_vocal, paths= "", save_root_ins=save_ins, agg=10, format0="wav")
        # 결과 출력
        for output in vc_output4:
            print(output)
    except Exception as e:
        print(f"함수 호출 중 예외 발생: {e}")

    print(f"Selected model: {uvr5_names[2]}")
    print(ngpu)


# 훈련 (사용자의 목소리)
#step1  기본 설정
def preprocess_train(trainset_dir,model_dir):
    try:
        info1 = preprocess_dataset(
            trainset_dir=trainset_dir,  # 목소리 데이터 주소
            exp_dir=model_dir,  # 목소리 모델의 이름 설정 가능
            sr="40k",  # 샘플링 레이트 40 or 48 (48000Hz 이상이면)
            n_p=11,  # CPU 코어 수
        )
        torch.cuda.empty_cache()
        print("preprocess_dataset 실행 성공")
    except Exception as e:
        print(f"preprocess_dataset 호출 중 예외 발생: {e}")
        


# step2a - 피처 추출
def extraction_f0(model_dir):
    try:
        info2 = extract_f0_feature(
            gpus="0",  # GPU 선택 (예: 0)
            n_p=11,  # 학습에 사용할 CPU 코어 수
            f0method="rmvpe_gpu",  # 추출 방법: "pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"
            if_f0=True,  # 피치 포함 여부 (FALSE면 국어책 읽기)
            exp_dir=model_dir,
            version19="v2",  # v1 또는 v2 (기본값)
            gpus_rmvpe="%s-%s" % (gpus, gpus),  # GPU 병렬 처리 방법, 기본값
        )
        torch.cuda.empty_cache()
        print("extract_f0_feature 실행 성공")
    except Exception as e:
        print(f"extract_f0_feature 호출 중 예외 발생: {e}")
        

#from memory_profiler import profile
#@profile
def model_train(trainset_dir,model_dir):
#step3 - 모델 훈련
    try:
        import psutil

        # 사용 중인 CPU 코어 수
        cpu_count = psutil.cpu_count(logical=True)
        print(f"총 CPU 코어 수: {cpu_count}")

        # 각 코어의 사용률
        cpu_usage = psutil.cpu_percent(percpu=True)
        print("각 코어 사용률:", cpu_usage)
        info3 = click_train(
            exp_dir1=model_dir,
            sr2="40k",
            if_f0_3=True,  # 피치 포함 여부 (FALSE면 국어책 읽기)
            spk_id5='0',  # 보컬 ID, 기본값 사용
            save_epoch10='30',  # 학습 중간 저장 빈도
            total_epoch11='15',  # 전체 epoch
            batch_size12='20',  # 배치 사이즈
            if_save_latest13="NO",  # 마지막 ckpt 파일만 저장할지 여부 (디스크 용량 문제)
            pretrained_G14="/app/assets/pretrained_v2/f0G40k.pth",  # Pretrain 제너레이터
            pretrained_D15="/app/assets/pretrained_v2/f0D40k.pth",  # Pretrain discriminator
            gpus16="0",
            if_cache_gpu17="NO",  # GPU에 트레이닝 셋 캐싱 (속도 증가, 메모리 많이 씀)
            if_save_every_weights18="NO",  # 세이브 지점마다 모델 생성 여부
            version19="v2",  # v1 또는 v2 (기본값)
        )
        torch.cuda.empty_cache()
        print("click_train 실행 성공")
    except Exception as e:
        print(f"click_train 호출 중 예외 발생: {e}")


# #step3 - 피처 인덱스 훈련
#     try:
#         info3 = train_index(
#             model_dir,
#             "v2",
#         )
#         print("train_index 실행 성공")
#     except Exception as e:
#         print(f"train_index 호출 중 예외 발생: {e}")




"""
# step3- 한번에 다하기
train1key,
[
    exp_dir1,
    sr2 = "40k",
    if_f0_3 = True, # 피치포함 여부 (FALSE면 국어책 읽기)
    trainset_dir4 = "", #목소리 데이터 주소 \\ 
    spk_id5 = 0,  # 보컬 id 뭔지 잘모름 기본값
    np7 = 11, #학습에 사용할 cpu 코어 수 
    f0method8 = "rmvpe_gpu", #추출방법 "pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"
    save_epoch10 = 30, #학습 중간 저장빈도
    total_epoch11 = 150, #전체 epoch
    batch_size12 = 12, #배치 사이즈
    if_save_latest13 = "NO", # 마지막 ckpt파일만 저장할건지 (디스크용량문제)
    pretrained_G14 = "/app/assets/pretrained/f0G40k.pth", #pretrain  제네레이터
    pretrained_D15 = "/app/assets/pretrained/f0D40k.pth",# pretrain discriminator
    gpus16 = gpus,
    if_cache_gpu17 = "YES",#GPU에 트레이닝 셋 캐싱 (속도 증가 - 메모리 많이 씀)
    if_save_every_weights18 = "NO", #세이브 지점마다 모델생성
    version19= "v2", #v1or v2(기본),
    gpus_rmvpe = "%s-%s" % (gpus, gpus), # GPU 병렬처리 방법 일단 기본값으로 
],
info3



#배치 컨버젼, 여러 오디오 파일 한번에 
but1.click(
                        vc.vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                        ],
                        [vc_output3],
                        api_name="infer_convert_batch",
                    )


"""
gpus="0"
try:
    train1key(
        exp_dir1="test_1103",
        sr2="40k",
        if_f0_3=True,
        trainset_dir4="/app/source/vocal",
        spk_id5="0",
        np7=11,
        f0method8="rmvpe_gpu",
        save_epoch10="30",
        total_epoch11="5",
        batch_size12="3",
        if_save_latest13="NO",
        pretrained_G14="/app/assets/pretrained_v2/f0G40k.pth",
        pretrained_D15="/app/assets/pretrained_v2/f0D40k.pth",
        gpus16="0",
        if_cache_gpu17="NO",
        if_save_every_weights18="NO",
        version19="v2",
        gpus_rmvpe="%s-%s" % (gpus, gpus),
    )
except Exception as e:
    print(f"train 호출 중 예외 발생: {e}")
def train():
    try:
        train1key(
            exp_dir1="user_2",
            sr2="40k",
            if_f0_3=True,
            trainset_dir4="/app/source/vocal",
            spk_id5="0",
            np7=11,
            f0method8="rmvpe_gpu",
            save_epoch10="30",
            total_epoch11="5",
            batch_size12="3",
            if_save_latest13="NO",
            pretrained_G14="/app/assets/pretrained_v2/f0G40k.pth",
            pretrained_D15="/app/assets/pretrained_v2/f0D40k.pth",
            gpus16="0",
            if_cache_gpu17="NO",
            if_save_every_weights18="NO",
            version19="v2",
            gpus_rmvpe="%s-%s" % (gpus, gpus),
        )
    except Exception as e:
        print(f"train 호출 중 예외 발생: {e}")



from pydub import AudioSegment
def mixing(vocal_path, inst_path, output_path):
    try:
        vocal = AudioSegment.from_file(vocal_path)
        instrumental = AudioSegment.from_file(inst_path)
        
        # diffrent length
        # if len(vocal) > len(instrumental):
        #     vocal = vocal[:len(instrumental)]
        # else:
        #     instrumental = instrumental[:len(vocal)]
        
        combined = vocal.overlay(instrumental)
        
        combined.export(output_path, format="wav")
    except Exception as e:
        print(f"mixing 호출 중 예외 발생: {e}")
    
    
    
    #############################################
def coversong_train(sid0, input_audio_path, index_path):
# 모델 infer(변환)
    file_to_index = {sid0 : 0}

    try:
        sid0_value = sid0  # sid0에 해당하는 값
        protect0_value = 0.33  # protect0에 해당하는 값
        protect1_value = 0.33  # protect1에 해당하는 값
        results = vc.get_vc(sid0_value, protect0_value, protect1_value)
        print("불러오기 성공")
        import psutil
        # 논리적 CPU 코어 개수
        num_logical_cores = psutil.cpu_count(logical=True)

        # 물리적 CPU 코어 개수
        num_physical_cores = psutil.cpu_count(logical=False)

        # 각 CPU 코어의 사용률
        cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)

        print(f"Number of logical cores: {num_logical_cores}")
        print(f"Number of physical cores: {num_physical_cores}")
        print(f"CPU usage per core: {cpu_percentages}")
        if results:
            spk_item, protect0, protect1, file_index2 = results
            #spk_item, protect0, protect1, file_index2, file_index4 = results
            print("Speaker Item:", spk_item)
            print("Protect0:", protect0)
            print("Protect1:", protect1)
            print("File Index2:", file_index2)
            #print("File Index4:", file_index4)
                
        vc_output1, vc_output2 = vc.vc_single(
            sid=sid0,  # 화자 선택 (기본값으로 사용)
            input_audio_path=input_audio_path,  # 변환할 노래/
            f0_up_key=int(0),  # 옥타브 조정: 정수로 변환 남-노래일 때 
            f0_file="",  # optional F0 커브파일
            f0_method="rmvpe",  # "pm", "harvest", "crepe", "rmvpe" 중 rmvpe사용
            file_index=index_path,  # 목소리 모델의 인덱스 파일
            file_index2="",  # 목소리 모델의 인덱스 파일 지정
            index_rate=float(0.75),  # 인덱스 파일 비율을 실수로 변환
            filter_radius=int(3),  # 필터 반지름을 정수로 변환
            resample_sr=int(0),  # 리샘플링 SR을 정수로 변환
            rms_mix_rate=float(0.25),  # RMS 믹스 비율을 실수로 변환
            protect=float(0.33),  # 보호 비율을 실수로 변환
        )
        print ("성공")
        print(vc_output1)
        print(vc_output2)
    except Exception as e:
        print(f"Error occurred: {e}")
##############################################################

# file_to_index = {"user_2.pth" : 0}
# print(torch.__version__)
# try:
#     sid0_value = "user_2.pth"  # sid0에 해당하는 값
#     protect0_value = 0.33  # protect0에 해당하는 값
#     protect1_value = 0.33  # protect1에 해당하는 값
#     results = vc.get_vc(sid0_value, protect0_value, protect1_value)
#     print("불러오기 성공")
#     if results:
#         spk_item, protect0, protect1, file_index2 = results
#         #spk_item, protect0, protect1, file_index2, file_index4 = results
#         print("Speaker Item:", spk_item)
#         print("Protect0:", protect0)
#         print("Protect1:", protect1)
#         print("File Index2:", file_index2)
#         #print("File Index4:", file_index4)
            
#     vc_output1, vc_output2 = vc.vc_single(
#         sid="user_2.pth",  # 화자 선택 (기본값으로 사용)
#         input_audio_path="/app/source/song/LiMYY.mp3",  # 변환할 노래/
#         f0_up_key=int(0),  # 옥타브 조정: 정수로 변환 남-노래일 때 
#         f0_file="",  # optional F0 커브파일
#         f0_method="rmvpe",  # "pm", "harvest", "crepe", "rmvpe" 중 rmvpe사용
#         file_index="",  # 목소리 모델의 인덱스 파일
#         file_index2="",  # 목소리 모델의 인덱스 파일 지정
#         index_rate=float(0.75),  # 인덱스 파일 비율을 실수로 변환
#         filter_radius=int(3),  # 필터 반지름을 정수로 변환
#         resample_sr=int(0),  # 리샘플링 SR을 정수로 변환
#         rms_mix_rate=float(0.25),  # RMS 믹스 비율을 실수로 변환
#         protect=float(0.33),  # 보호 비율을 실수로 변환
#     )
#     print ("성공")
#     print(vc_output1)
#     print(vc_output2)
# except Exception as e:
#     print(f"Error occurred: {e}")        