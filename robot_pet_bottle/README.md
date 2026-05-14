# robot_pet_bottle

AI Hub(행동 데이터) + Roboflow(물병) 혼합 **CNN 분류 학습**, 추론, 웹캠 **YOLO 박스(사람/물병)** 오버레이 서버.

## 저장소 구조

| 경로 | 설명 |
|------|------|
| `scripts/train_cnn_mixed.py` | 데이터 준비 + CNN 학습 (ResNet18 또는 **MobileNetV2** 경량) |
| `scripts/infer_cnn_mixed.py` | 학습된 `.pt`로 이미지/URL 추론 |
| `scripts/webcam_infer_server.py` | Flask 웹 UI + MJPEG 스트림 + YOLO 박스 |
| `requirements.txt` | x86_64 개발 PC 기준 고정 버전 |
| `requirements-rpi.txt` | 라즈베리파이용 (CPU torch, 동일 앱 버전) |
| `requirements-rpi-infer.txt` | 라즈베리파이 추론 서버용 최소 패키지 |
| `ros2_sllidar_reference/` | ROS2 라이다 실행 파일과 설명 문서 |
| `ros2_sllidar_with_sdk/` | SLLidar SDK까지 포함한 ROS2 업로드용 폴더 |

## Python / 라이브러리 버전 (개발 PC 기준)

| 항목 | 버전 |
|------|------|
| Python | 3.8+ 권장 |
| torch | 2.4.1 |
| torchvision | 0.19.1 |
| numpy | 1.24.4 |
| opencv-python-headless | 4.10.0.84 |
| ultralytics | 8.4.21 |
| flask | 3.0.3 |
| requests | 2.32.4 |
| roboflow | 1.2.16 |
| pillow | 10.4.0 |
| PyYAML | 5.3.1 |

## 설치 (개발 PC)

```bash
cd robot_pet_bottle
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 데이터 배치

```text
data/
  01_원천데이터/   # TS_*.zip
  02_라벨링데이터/ # TL_*.zip
```

`--aihub-base` 기본값은 `./data` 입니다.

## Roboflow

```bash
export ROBOFLOW_API_KEY="본인_키"
```

## 학습

**GPU PC 권장.** 라즈베리파이에서는 **학습 대신 추론만** 권장.

```bash
python3 scripts/train_cnn_mixed.py \
  --aihub-base ./data \
  --work-dir ./cnn_mixed_work \
  --backbone mobilenet_v2 \
  --aihub-ratio 0.3 \
  --epochs 20 \
  --batch-size 32 \
  --device auto
```

체크포인트 재개:

```bash
python3 scripts/train_cnn_mixed.py \
  --resume --skip-prepare \
  --work-dir ./cnn_mixed_work \
  --epochs 30 \
  --batch-size 32 \
  --device auto
```

출력: `cnn_mixed_work/cnn_mixed_<backbone>.pt`, `classes.txt`

## 추론

```bash
python3 scripts/infer_cnn_mixed.py \
  --model ./cnn_mixed_work/cnn_mixed_mobilenet_v2.pt \
  --sample
python3 scripts/infer_cnn_mixed.py --model ./cnn_mixed_work/cnn_mixed_mobilenet_v2.pt --image ./photo.jpg
```

## 웹캠 서버 (YOLO person/bottle 박스)

기본 포트 **8765**, 바인딩 **0.0.0.0**.

```bash
pip install -r requirements.txt
python3 scripts/webcam_infer_server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --model ./cnn_mixed_work/cnn_mixed_mobilenet_v2.pt \
  --camera 0
```

YOLO만 끄기: `--no-yolo`  
MJPEG: `/stream`, `/video_feed`

## 라즈베리파이 (경량 실행)

1. 학습은 PC에서 하고 **`.pt`만** Pi로 복사.
2. `requirements-rpi.txt` 참고 후 torch 설치, 나머지 동일.
3. 추론: `infer_cnn_mixed.py --device cpu`
4. 웹: 느리면 `--no-yolo`

PyTorch ARM: [pytorch.org](https://pytorch.org/get-started/locally/) 또는 [piwheels](https://www.piwheels.org/project/torch/).

## ROS2 SLLidar 예제

라즈베리파이에서 실제로 실행한 라이다 프로그램도 함께 정리했습니다. 실행 기준 명령은 아래와 같습니다.

```bash
ros2 launch slampibot_robot slam_robot.launch.py
```

현재는 두 가지 폴더가 있습니다.

- `ros2_sllidar_reference/`
  - 문서와 예제 파일 위주로 정리한 참고용 폴더
  - SDK 전체 소스는 포함하지 않음
- `ros2_sllidar_with_sdk/`
  - 참고용 파일에 더해 `slampibot_robot/sdk/` 전체를 같이 넣은 업로드용 폴더
  - 별도 SDK 복사 없이 바로 ROS2 워크스페이스에 넣어 빌드 가능

문서에는 아래 내용이 포함되어 있습니다.

- 현재 launch 파일이 실제로 무엇을 실행하는지
- 어떤 코드가 `/scan` 토픽을 퍼블리시하는지
- 필요한 ROS2/Ubuntu 패키지
- 워크스페이스에 복사해서 빌드하는 방법
- 확인용 토픽/TF 명령어

### ROS2 설치 예시

Ubuntu 22.04 + ROS2 Humble 기준:

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3-colcon-common-extensions
sudo apt install -y ros-humble-ros-base
sudo apt install -y \
  ros-humble-rclcpp \
  ros-humble-sensor-msgs \
  ros-humble-std-srvs \
  ros-humble-tf2-ros \
  ros-humble-ros2launch
```

### 동작 요약

- `slampibot_robot/launch/slam_robot.launch.py`
  - `sllidar_node`와 `static_transform_publisher`를 함께 실행
- `slampibot_robot/src/sllidar_node.cpp`
  - SLLidar SDK에서 스캔 데이터를 읽어 `/scan` 토픽으로 퍼블리시
- `tf2_ros/static_transform_publisher`
  - `base_link -> laser` 정적 TF를 발행

즉, 이 launch는 SLAM 전체가 아니라 "라이다 드라이버 + 정적 TF" 단계까지 담당합니다.

### 실행 후 확인

```bash
ros2 topic list
ros2 topic echo /scan --once
ros2 run tf2_ros tf2_echo base_link laser
```

## GitHub

```bash
git init
git remote add origin git@github.com:pumker5202/robot_pet_bottle.git
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main
```

HTTPS: `https://github.com/pumker5202/robot_pet_bottle.git`

## 라이선스

프로젝트용 코드. 데이터/가중치는 각 출처 정책을 따릅니다.
