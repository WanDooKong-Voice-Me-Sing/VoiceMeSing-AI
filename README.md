# VoiceMeSing-AI
## 소개
이 프로젝트는 [RVC 모델](https://github.com/some-rvc-repo)을 기반으로 음성 변환 및 커버송 제작 기능을 구현한 프로젝트입니다. 기본 모델 및 구조는 해당 프로젝트를 참고하였으며, 이를 기반으로 학습 데이터 전처리 및 배포 환경을 최적화하였습니다. 

본 파트는 AI 서버 파트입니다.  서버에서는 음성 변환을 위한 데이터 전처리, 모델 학습, 모델을 이용한 목소리 변환 커버곡생성 등을 처리합니다.

## 주요 기능
- 음성 데이터를 학습 및 추론을 위한 전처리.
- 사용자 음성을 학습하여 RVC 모델을 훈련.
- 제작된 모델을 이용한 커버음원 생성
- REST API와 Redis를 통한 백엔드와 통신.

## 기술 스택
- 프레임워크: FastAPI
- 언어: Python
- 모델: RVC (Retrieval-based Voice Conversion)
- 배포: 로컬 우분투 서버
- 저장소: 로컬 우분투 서버
- 데이터 베이서: MySQL

## 환경 설정

다음 명령은 Python 버전이 3.8 이상인 환경에서 실행해야 합니다.
#### 1. pip를 통한 의존성 설치

1. Pytorch 및 의존성 모듈 설치, 이미 설치되어 있으면 생략. 참조: https://pytorch.org/get-started/locally/

```bash
pip install torch torchvision torchaudio
```

2. win 시스템 + Nvidia Ampere 아키텍처(RTX30xx) 사용 시, #21의 사례에 따라 pytorch에 해당하는 cuda 버전을 지정

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

3. 자신의 그래픽 카드에 맞는 의존성 설치

- N카드

```bash
pip install -r requirements.txt
```

- A카드/I카드

```bash
pip install -r requirements-dml.txt
```

- A카드ROCM(Linux)

```bash
pip install -r requirements-amd.txt
```

- I카드IPEX(Linux)

```bash
pip install -r requirements-ipex.txt
```

#### 2. poetry를 통한 의존성 설치

Poetry 의존성 관리 도구 설치, 이미 설치된 경우 생략. 참조: https://python-poetry.org/docs/#installation

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

poetry를 통한 의존성 설치

```bash
poetry install
```

### MacOS

`run.sh`를 통해 의존성 설치 가능

```bash
sh ./run.sh
```

## 기타 사전 훈련된 모델 준비

RVC는 추론과 훈련을 위해 다른 일부 사전 훈련된 모델이 필요합니다.

이러한 모델은 저희의 [Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)에서 다운로드할 수 있습니다.

### 1. assets 다운로드

다음은 RVC에 필요한 모든 사전 훈련된 모델과 기타 파일의 목록입니다. `tools` 폴더에서 이들을 다운로드하는 스크립트를 찾을 수 있습니다.

- ./assets/hubert/hubert_base.pt

- ./assets/pretrained

- ./assets/uvr5_weights

v2 버전 모델을 사용하려면 추가로 다음을 다운로드해야 합니다.

- ./assets/pretrained_v2

### 2. ffmpeg 설치

ffmpeg와 ffprobe가 이미 설치되어 있다면 건너뜁니다.

#### Ubuntu/Debian 사용자

```bash
sudo apt install ffmpeg
```

#### MacOS 사용자

```bash
brew install ffmpeg
```

#### Windows 사용자

다운로드 후 루트 디렉토리에 배치.

- [ffmpeg.exe 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- [ffprobe.exe 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

### 3. RMVPE 인간 음성 피치 추출 알고리즘에 필요한 파일 다운로드

최신 RMVPE 인간 음성 피치 추출 알고리즘을 사용하려면 음피치 추출 모델 매개변수를 다운로드하고 RVC 루트 디렉토리에 배치해야 합니다.

- [rmvpe.pt 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)

# 서버구성
1. Ubuntu-24.04.1-ubuntu-24.04.1-live-server-amd64

2. CUDA Toolkit (11.6.2) 및 cuDNN 8

3. FFmpeg

4. Local H/W(AMD Ryzen 7 7800X3D 8-Core & Nvidia GeForce RTX 4080 SUPER)

5. Python 의존성 설치:
```bash
pip install -r requirements.txt
```
