# PathFollowing

ROS2 기반 PathFollowing + MPPI 유도제어 패키지입니다.

## 1) What This Package Does

PathFollowing 패키지는 `node`, `core`, `utils` 계층으로 구성됩니다.

### `node` (ROS I/O, 브릿지)
- `node_pathfollowing`: 센서/웨이포인트/상위 상태 토픽을 받아 PathFollowing 제어 명령 생성
- `node_mppi`: MPPI를 수행하고 PathFollowing 입력 파라미터(`u0`, `u1`)를 반환

### `core` (알고리즘 계산)
- `core_pathfollowing`: PathFollowing 핵심 알고리즘
- `core_mppi` + `mppi_kernel.cu`: MPPI + CUDA rollout 커널 (pycuda)
- `gpr`: NDO 기반 외란 추정 신호를 이용한 GPR 예측 모듈

### `utils` (분석/시각화)
- core에서 공통으로 사용하는 계산/수식 보조 모듈
- 로그 저장, 리포트 생성, XY 리플레이 등 오프라인 분석 도구

## 2) Configuration Guide

### sim.yaml (운용/알고리즘 튜닝)
- `vehicle_type`: 1(quad), 2(octo)
- `guid_type`: 0, 1, 2
- `wp_type`: waypoint 생성 타입
- `path_following`: 추종/전환 관련 파라미터
- `mppi`: 샘플 수, horizon, 비용/노이즈, 초기 입력 파라미터
- `ndo`, `gpr`: 외란 추정/예측 관련 파라미터

### quad.yaml / octo.yaml (기체/모델 파라미터)
- `model`: 질량, 관성, 중력
- `actuator`: 모터 수, 최대 추력, 로터 위치/회전 방향
- `px4_model`: 기울기/가속 제한, 시간상수, rate gain

## 3) Version Notes

### r0.0.1
- `node`/`core` 계층 분리 및 구조 리팩터링
- 파라미터 YAML 구성 및 항목 정리
- MPPI PyCUDA 커널을 별도 파일(`mppi_kernel.cu`)로 분리

---

## Appendix) Reference

아래 항목은 참고용입니다.

## A) Logging and Flight Report


실행 위치(프로젝트 루트):
```bash
cd ~/Documents/A4VAI-SITL/ROS2/ros2_ws/src/pathfollowing
```

리포트 생성 (최신 CSV 자동 선택, 같은 폴더에 그래프 저장):
```bash
python3 -m pathfollowing.utils.pf_flight_report
```


## B) Safety Policy

이 패키지는 비행 시작 전 설정 불일치를 즉시 드러내도록 구성합니다.
- 필수 YAML/파라미터 누락 시 `RuntimeError`로 즉시 중단
- 허용 범위 밖 `guid_type`, `vehicle_type`, `wp_type` 즉시 중단
- 필수 모델/튜닝 상수 누락 시 기본값으로 대체하지 않고 중단
- 추후 즉시 중단이 아닌, 안전 모드로 전환하는 알고리즘 추가 예정


<!-- ## C) Run Core-Only Tests

```bash
python test/run_core_smoke.py
python -m pytest -q test/test_core_runtime.py
```

검증 포인트 예시:
- PathFollowing core 단독 루프 출력 생성
- PathFollowing ↔ MPPI 브릿지 reset flag 전이
- waypoint 재생성/`guid_type` 전환 런타임 시나리오
- yaw wrapping 경계(`-pi/pi`) 연속성
- 최종 waypoint 완료 판정 -->
