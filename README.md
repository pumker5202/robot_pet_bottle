# robot_pet_bottle

라즈베리파이 기반 개인 로봇 프로젝트 저장소입니다. 기존 하드웨어, 아두이노, 오도메트리 자료는 그대로 유지하고, `robot_pet_bottle/` 폴더에는 비전 추론 서버와 ROS2 SLLidar 예제를 함께 정리했습니다.

## 주요 폴더

- `robot_pet_bottle/`
  - CNN + YOLO 기반 웹캠 추론 프로젝트
  - ROS2 SLLidar 참고 자료와 SDK 포함 예제 추가
- `PET_ardino/`, `aduino_motor/`, `아두이노_코드/`
  - 아두이노/모터 관련 기존 자료
- `odom/`, `topic/`
  - ROS 토픽, 오도메트리 관련 기존 자료

## 새로 추가한 내용

`robot_pet_bottle/` 아래에 다음 폴더를 추가했습니다.

- `ros2_sllidar_reference/`
  - 문서와 예제 파일 위주로 정리한 참고용 폴더
- `ros2_sllidar_with_sdk/`
  - SLLidar SDK 전체까지 포함한 업로드용 폴더

실행 기준 명령은 아래와 같습니다.

```bash
ros2 launch slampibot_robot slam_robot.launch.py
```

자세한 설명은 `robot_pet_bottle/README.md`와 각 폴더의 `README.md`를 보면 됩니다.
