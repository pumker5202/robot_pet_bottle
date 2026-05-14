# SLLidar SDK 포함 폴더

이 폴더는 `sllidar_node.cpp`에서 사용하는 SLLidar SDK 실제 소스가 이미 포함된 상태입니다. 즉, 별도로 SDK를 추가 복사하지 않아도 됩니다.

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

현재 이 폴더 안에는 아래 자료가 함께 들어 있습니다.

- `include/`
- `src/`
- `Makefile`

각 소스 파일의 상단 저작권 및 라이선스 표기는 원본 상태를 유지했습니다.
