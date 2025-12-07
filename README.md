# 어드벤처 게임 Rain World 강화학습 프로그램

## 프로젝트에서 사용한 RainWorld 설명

Rain World는 https://store.steampowered.com/app/312520/Rain_World/ 에서 구매할 수 있는 리듬게임이다. 아레나 모드를 활용해, 두 적과 플레이어가 존재하는 환경을 구축해 플레이어가 최대한 오래 생존하는 것을 강화학습의 목적으로 하였다. Rain World를 실행해, 강화학습 코드와 직접 상호작용하며 학습되고 시뮬레이션된다.

## 폴더별 역할 정리

- [main.py](http://main.py/)
    - 강화학습 프로그램 실행 시작점
    - random number generator seed 변경을 통한 실험 및 신뢰구간 작성
    - 게임과 연결한 후, 인게임 속도와 학습할 알고리즘 설정
    - 강화학습 알고리즘 실행
- rainworld_connector.py
    - 게임과의 통신을 담당 (state 받기, action 보내기)
- modelloader.py
    - 학습중인 모델을 임시 저장하고, 학습 중 기록된 episode 점수를 저장
- visualize.ipynb
    - 학습된 알고리즘들의 성능을 통합, 요약, 시각화하여 최종 비교 분석을 수행
- lib/
    - 게임 모드 개발에 필요한 라이브러리
- mod/
    - 게임 모드 설정
- src/
    - 게임에 직접 적용되는 C# 프로젝트
    - 게임 오버 시 자동으로 재시작, 플레이어의 위치 초기화
    - 게임에서 0.1초가 흐를 때마다 state를 python 프로세스에 전송
    - python 프로세스에서 받은 action을 실행
- models/
    - 학습된 모델이 저장됨
- scores/
    - 기록된 점수가 저장됨

## 프로그램 구동 방법

### 프롬프트

```bash
# 기본 실행
pip main.py
#hyper parameter 사용시
pip main.py --algo PPO --lr 0.0003 --gamma 0.95 --render --socket ****
```

### 게임

1. 게임 홈 경로에서 BepInex/plugins 폴더로 이동 (혹은 `C:\Program Files (x86)\Steam\steamapps\common\Rain World\BepInEx\plugins` 경로로 이동)
2. 해당 폴더에 `RLProject.dll` 파일을 복사
3. 게임 실행
4. 아레나의 샌드박스 모드에서, 적을 둘 배치한 환경을 구축 후 플레이한다.

## 설치 및 요구사항

이 프로젝트를 실행하기 위해서는 Python 3.x 환경과 typer, random 등의 라이브러리가 필요하다.

## 라이브러리 설치

프로젝트 루트 디렉토리에서 다음 명령어를 사용하여 필수 패키지를 설치한다.

`pip install -r requirements.txt`

## 강화학습 환경 설정

- Observation Space: (6, )
    1. ← 키 입력
    2. → 키 입력
    3. ↓ 키 입력
    4. ↑ 키 입력
    5. 키 입력 없음
    6. 점프
- Action Space : (8, )
    1. 플레이어의 x좌표 
    2. 플레이어의 y좌표
    3. 적1의 x좌표
    4. 적2의 y좌표
    5. 적1의 x좌표
    6. 적2의 y좌표
    7. 플레이어가 봉을 잡고 오를 수 있는 상태인가? (1 또는 0)
    8. 플레이어가 파이프를 탈 수 있는 상태인가? (1 또는 0)
       
    *모든 좌표는 0~1 사이로 정규화함
- reward
    - 매 step 마다 1의 보상
    - 60초 생존 성공 시 10의 보상 (및 에피소드 정지)
    - 적에게 잡힐 시 -10의 보상

## 성능 평가 및 비교 (Evaluation 함수 내 구현)

- 주요 지표: 각 알고리즘별 평균 점수와 95% 신뢰 구간을 계산함
- 시각화: `plots/` 폴더에 CI를 포함한 막대 차트 및 학습 곡선 생성
- 실행 환경 특성 상 시간 소모가 커 세 개의 시드만 사용
