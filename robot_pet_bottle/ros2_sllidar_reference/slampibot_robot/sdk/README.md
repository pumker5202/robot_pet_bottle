# SLLidar SDK 위치

이 패키지는 `sllidar_node.cpp`에서 `sl_lidar.h`와 SDK 소스 파일들을 직접 사용합니다. 그래서 이 폴더 아래에 SLLidar SDK가 있어야 빌드됩니다.

필요한 최소 구조는 아래와 같습니다.

```text
sdk/
├── include/
│   ├── sl_lidar.h
│   └── ...
└── src/
    ├── sl_lidar_driver.cpp
    ├── arch/linux/*.cpp
    ├── hal/*.cpp
    └── dataunpacker/**/*.cpp
```

`CMakeLists.txt`는 위 구조를 기준으로 아래 경로를 읽습니다.

- `sdk/include`
- `sdk/src`
- `sdk/src/arch/linux/*.cpp`
- `sdk/src/hal/*.cpp`
- `sdk/src/dataunpacker/*.cpp`
- `sdk/src/dataunpacker/unpacker/*.cpp`

즉, 이 예제 폴더만 단독으로 복사해도 바로 빌드되지는 않으며, 공식 SLLidar SDK 또는 기존에 사용하던 `sdk/` 폴더를 함께 넣어야 합니다.
