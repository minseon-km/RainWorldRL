# 리듬게임 osu! 강화학습 프로그램 

## 프로젝트에서 사용한 osu! 설명

osu!는 https://osu.ppy.sh/ 에서 다운받을 수 있는 리듬게임으로 https://osu.ppy.sh/beatmaps/packs 에서 osu!에서 사용하는 비트맵 팩을 다운받을 수 있다. osu!에서는 .osu 파일로 리듬게임을 할 수 있는데 여러 개의 .osu 맵 파일들을 .osz 파일 형태로 묶어 놓은 자료 모음이 비트맵 팩이다. 해당 프로젝트에서는 osu! 게임과 직접 연동하지는 않고 배포된 비트맵 팩의 osu! 파일을 기반으로 강화학습을 진행하였다. 

## 폴더별 역할 정리 

- main.py
    - 프로그램 실행 시작점
    - random number generator seed 변경을 통한 실험 및 신뢰구간 작성
    - 노래 선택 함수 불러온 후 알고리즘 선택하게 함
    - 알고리즘 함수 불러와서 실행

- beatmap.py
    - .osu 파일을 읽고 노트·타이밍 정보를 파싱

- environment.py
    - gym.Env 클래스로 RL 환경 구현

- evaluate.py
    - 학습된 알고리즘들의 성능을 통합, 요약, 시각화하여 최종 비교 분석을 수행

- song_manager.py
    - data 폴더 내 .osz 파일들을 자동으로 번호 매겨 출력
    - 사용자 입력으로 곡 선택
    - 선택한 .osz 압축 해제 후 .osu 파일을 찾아 반환
    - 난이도에 따라 나뉜 .osu 파일 중에 사용자 입력으로 곡 선택
 
- modelloader.py
    - 모델의 가중치(state_dict)를 .pth 파일로 저장 및 로드
    - 에피소드 점수 리스트를 .npy 파일로 저장 및 로드
 
- visualize.ipynb
    - scores 폴더에 저장해둔 점수 데이터를 알고리즘별로 이동 평균 처리하여 학습 추이를 시각화
    - 전체 데이터를 기반으로 평균 보상과 95% 신뢰 구간을 계산하여 최종 성능 시각화

- models/
    - 학습된 모델이 저장됨
 
- scores/
    - modelloader에서 업로드한 알고리즘별 점수가 저장됨 

- beatmaps/
    - 학습용 맵(.osu) 파일들

- data/
    - 원본 .osz 자료

## 프로그램 구동 방법

```
python3 main.py
```
- 곡 선택, 알고리즘 선택 등이 모두 차례로 수동으로 이루어진다. 원래 프로그램의 개발 목표는 이쪽이다.

```
python main.py train --algo PPO --lr 0.0003 --gamma 0.95 --song 2 --render
```
- 서로 다른 hyperparameter에서의 결과 비교가 가능하도록 하기 위해서 이 옵션을 typer를 통해서 추가하였다. 알고리즘 선택과 학습률과 할인율 지정, 곡 선택이 가능하다. 마지막 --render는 GUI visualization을 enable한다는 뜻이다. 


## 설치 및 요구사항

이 프로젝트를 실행하기 위해서는 Python 3.x 환경과 typer, random 등의 라이브러리가 필요하다. 

## 라이브러리 설치

프로젝트 루트 디렉토리에서 다음 명령어를 사용하여 필수 패키지를 설치한다. 

```bash
pip install gymnasium numpy pandas torch typer seaborn
```

## 강화학습 환경 설정

- Action Space
    - `0`: 대기 (No Action)
    - `1`: 노트 입력 (클릭)
- Observation Space
    - [다음 노트까지의 시간, x좌표, y좌표, combo(정규화)] 형식의 4차원 벡터
- Reward
    - 노트 입력 타이밍이 정확할수록 0.0에서 최대 1.0의 보상
    - 노트 입력 타이밍을 놓치거나 오차가 클 경우 벌점 -0.5를 받으며 콤보가 초기화됨
- 악곡 전체를 사용하기에는 노래가 길어서 처음 200개 노트만 사용함

## 성능 평가 및 비교 

- 다중 시드: [0, 23, 147, 575, 2768]를 random number generator seed로 만들어서 편향을 줄임
- 주요 지표: 각 알고리즘별 평균 점수와 95% 신뢰 구간을 계산함
- 시각화: \plots 폴더에 CI를 포함한 막대 차트 및 학습 곡선 생성

## 결과 요약

- 세 모델 중 PPO가 가장 성능이 좋았다. DQN은 낮은 성능을 보였고, A2C는 학습률을 낮게 잡았을 때 성능이 높았다. 
