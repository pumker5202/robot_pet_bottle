# ROS2 SLLidar With SDK

라즈베리파이에서 실제로 실행한 아래 명령 기준으로, 필요한 파일과 동작 방식을 보기 쉽게 정리한 폴더입니다. 이 버전은 `slampibot_robot/sdk/` 전체가 함께 들어 있는 업로드용 묶음입니다.

```bash
ros2 launch slampibot_robot slam_robot.launch.py
```

## 이 폴더에 들어 있는 것

- `slampibot_robot/launch/slam_robot.launch.py`
  - 라이다 노드와 정적 TF를 함께 띄우는 launch 파일
- `slampibot_robot/src/sllidar_node.cpp`
  - SLLidar SDK로부터 스캔 데이터를 읽어 `sensor_msgs/msg/LaserScan`으로 퍼블리시하는 ROS2 노드
- `slampibot_robot/CMakeLists.txt`
  - C++ 노드와 SDK 소스를 함께 빌드하는 설정
- `slampibot_robot/package.xml`
  - ROS2 의존성 정의
- `slampibot_robot/sdk/`
  - 실제 빌드에 사용하는 SLLidar SDK 소스 전체

즉, 이 폴더는 `ros2_sllidar_reference/`와 다르게 SDK를 별도 다운로드하거나 따로 복사할 필요가 없습니다.

## 실제로 무엇이 실행되나

현재 `slam_robot.launch.py`는 이름과 달리 SLAM 알고리즘까지 실행하지는 않습니다. 실제로 실행되는 것은 아래 두 개입니다.

1. `sllidar_node`
   - 시리얼 포트로 라이다에 연결
   - 장치 정보와 상태를 읽음
   - 스캔을 시작하고 `/scan` 토픽에 `LaserScan` 메시지를 퍼블리시
2. `tf2_ros/static_transform_publisher`
   - `base_link -> laser` 정적 TF를 발행
   - 라이다가 로봇 기준 좌표계 어디에 달려 있는지 알려 줌

즉, 이 launch는 "라이다 데이터 발행 준비"까지 담당하는 파일입니다.

## 터미널에서 확인된 실행 상태

실행 로그 기준으로 아래 항목이 확인되었습니다.

- 라이다 연결 성공
- 장치 정보 읽기 성공
- health status: `OK`
- scan mode: `Standard`
- sample rate: `2 Khz`
- max distance: `12.0 m`
- scan frequency: `10.0 Hz`

## 코드별 역할

### `slam_robot.launch.py`

- `channel_type`, `serial_port`, `serial_baudrate`, `frame_id`, `angle_compensate`, `scan_mode` 같은 launch 인자를 선언합니다.
- `slampibot_robot` 패키지의 `sllidar_node` 실행 파일을 띄웁니다.
- `tf2_ros`의 `static_transform_publisher`로 `base_link -> laser` 변환을 고정값으로 발행합니다.

### `sllidar_node.cpp`

- `SLlidarNode`
  - ROS2 노드 클래스입니다.
- `init_param()`
  - launch 파일에서 넘어온 파라미터를 읽습니다.
- `getSLLIDARDeviceInfo()`
  - 시리얼 번호, 펌웨어, 하드웨어 버전을 출력합니다.
- `checkSLLIDARHealth()`
  - 라이다 상태를 검사합니다.
- `publish_scan()`
  - SDK 원시 데이터를 `sensor_msgs/msg/LaserScan` 형식으로 바꿔 퍼블리시합니다.
- `work_loop()`
  - 드라이버 연결, 스캔 시작, 데이터 읽기, 퍼블리시를 반복하는 메인 루프입니다.
- `start_motor`, `stop_motor`
  - 서비스로 모터 제어를 할 수 있게 해 둔 부분입니다.

## 필요한 설치 패키지

아래는 Ubuntu 22.04 + ROS2 Humble 기준 예시입니다. 라즈베리파이에서도 같은 흐름으로 준비하면 됩니다.

### 1) 기본 개발 도구

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3-colcon-common-extensions
```

### 2) ROS2 기본 환경

ROS2 Humble이 아직 없다면 먼저 설치합니다.

```bash
sudo apt install -y ros-humble-ros-base
```

### 3) 이 예제에 필요한 ROS2 패키지

```bash
sudo apt install -y \
  ros-humble-rclcpp \
  ros-humble-sensor-msgs \
  ros-humble-std-srvs \
  ros-humble-tf2-ros \
  ros-humble-ros2launch
```

### 4) 추가로 필요한 외부 코드

이 예제의 `sllidar_node.cpp`는 `sl_lidar.h`와 `sdk/src/*.cpp`를 사용합니다. 이 폴더에는 필요한 SDK가 이미 포함되어 있습니다.

정리하면:

- ROS2 패키지들 설치
- 이 폴더를 ROS2 워크스페이스에 복사

## 워크스페이스에서 빌드하는 방법

예시 워크스페이스가 `~/ros2_ws`일 때:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
cp -r /path/to/robot_pet_bottle/ros2_sllidar_with_sdk/slampibot_robot .
```

이 버전은 SDK가 이미 들어 있으므로 바로 빌드하면 됩니다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select slampibot_robot
source install/setup.bash
```

## 실행 방법

기본 실행:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch slampibot_robot slam_robot.launch.py
```

포트가 다르면 launch 인자로 바꿔서 실행합니다.

```bash
ros2 launch slampibot_robot slam_robot.launch.py serial_port:=/dev/ttyUSB0
```

## 실행 후 확인할 것

### `/scan` 토픽 확인

```bash
ros2 topic list
ros2 topic echo /scan --once
```

### TF 확인

```bash
ros2 run tf2_ros tf2_echo base_link laser
```

### 서비스 확인

```bash
ros2 service list | grep motor
```

## 주의할 점

- `/dev/ttyUSB0`, `/dev/ttyUSB1`는 연결 순서에 따라 바뀔 수 있습니다.
- 현재 launch 파일의 기본 포트는 실제 실행 기준에 맞춰 `/dev/ttyUSB1`로 맞춰 두었습니다.
- 권한 문제로 시리얼 장치가 안 열리면 사용자를 `dialout` 그룹에 추가하거나 udev rule을 설정해야 할 수 있습니다.
- 이 예제는 "드라이버 + TF" 단계까지입니다. 실제 지도 작성까지 하려면 `slam_toolbox` 또는 `cartographer`를 추가로 연결해야 합니다.

## 추천 확장

다음 단계로 이어 가려면 아래 구성이 자연스럽습니다.

1. `rviz2`로 `/scan` 시각화
2. `slam_toolbox` 연결
3. 바퀴 오도메트리와 `base_link` 프레임 추가
4. 정적 TF를 URDF 또는 robot_state_publisher로 대체
