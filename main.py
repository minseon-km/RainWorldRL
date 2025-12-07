from song_manager import select_song
from beatmap import extract_osz, parse_osu_file
from environment import RhythmEnv
from evaluate import main as evaluate_main

# libraries
import gymnasium as gym
import collections
import random
import numpy as np
import os
import typer
import glob
import torch

from models.ppo_model import *
from models.dqn_models import *
from models.a2c_model import *
from modelloader import save_score

app = typer.Typer(help="Osu! RL training")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_with_seeds(alg_name, render, selected_osu, lr=1e-3, gamma=0.99):
    seeds = [0, 23, 147, 575, 2768]
    results = []
    scores_per_seeds = []

    for seed in seeds:
        print(f"\n Running {alg_name} (seed={seed})")
        set_seed(seed)
        if alg_name == "DQN" or alg_name == "Double_DQN" or alg_name == "Dueling_DQN":
            q, q_target, scores, episodes = run_experiment_dqn(algorithm_type=alg_name, render=render, selected_osu=selected_osu, lr=lr, gamma=gamma)
        elif alg_name == "PPO":
            scores, episodes = run_experiment_ppo(algorithm_type=alg_name, render=render, selected_osu=selected_osu, lr=lr, gamma=gamma)
        elif alg_name == "A2C":
            scores, episodes = run_experiment_a2c(algorithm_type=alg_name, render=render, selected_osu=selected_osu, lr=lr, gamma=gamma)

        # 마지막 20개 에피소드 리워드의 평균
        score = np.mean(scores[-20:])
        scores_per_seeds.append(scores)
        if score is not None:
                results.append(score)

    if not results:
        print("no results recorded.")
        return

    # 기본 통계 계산
    mean_score = np.mean(results)
    std_score = np.std(results)
    sem = std_score / np.sqrt(len(results))
    ci = 1.96 * sem  # 95% 신뢰구간
    print(f"\n===== {alg_name} summary =====")
    print(f" Seeds: {seeds}")
    print(f" Mean score: {mean_score:.2f}")
    print(f" Std: {std_score:.2f}")
    print(f" 95% CI: [{mean_score - ci:.2f}, {mean_score + ci:.2f}]")

    return scores_per_seeds

def main():
    selected_osu = select_song("data")  
    if not selected_osu:
        print("A song is not selected.\n")
        return

    """Run experiments for all three algorithms"""
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN", "PPO", "A2C"]
    seeds = [0, 23, 147, 575, 2768]

    print("Choose algorithm to run:")
    print("1. DQN")
    print("2. Double DQN")
    print("3. Dueling DQN")
    print("4. PPO")
    print("5. A2C")
    print("6. Run all algorithms")

    choice = input("Enter your choice (1-6): ")

    # Ask about rendering
    render_choice = input("Enable GUI visualization? (y/n): ").lower()
    render = render_choice in ['y', 'yes']

    # lr과 gamma 기본값 설정
    lr = 1e-3
    gamma = 0.99

    if choice == "1":
        scores = run_with_seeds("DQN", render, selected_osu, lr, gamma)
        save_score(scores, "./scores", "DQN")
    elif choice == "2":
        scores = run_with_seeds("Double_DQN", render, selected_osu, lr, gamma)
        save_score(scores, "./scores", "Double_DQN")
    elif choice == "3":
        scores = run_with_seeds("Dueling_DQN", render, selected_osu, lr, gamma)
        save_score(scores, "./scores", "Dueling_DQN")
    elif choice == "4":
        scores = run_with_seeds("PPO", render, selected_osu, lr, gamma)
        save_score(scores, "./scores", "PPO")
    elif choice == "5":
        scores = run_with_seeds("A2C", render, selected_osu, lr, gamma)
        save_score(scores, "./scores", "A2C")
    elif choice == "6":
        for alg in algorithms:
            scores = run_with_seeds(alg, render, selected_osu, lr, gamma)
            save_score(scores, "./scores", alg)
    else:
        print("Invalid choice, running DQN by default.")
        scores = run_with_seeds("DQN", render, selected_osu, lr, gamma)
        save_score(scores, "./scores", "DQN")

@app.command()
def train(
    algo: str = typer.Option(default="PPO", help="Algorithm"),
    lr: float = typer.Option(default=1e-3, help="Learning rate"),
    gamma: float = typer.Option(default=0.99, help="Discount factor"),
    song: str = typer.Option(default=None, help="Song number or path"),
    render: bool = typer.Option(default=False, help="Enable GUI"),
):

    if song is None:
        selected_osu = select_song("data")
    else:
        try:
            # 숫자로 입력된 경우 data 폴더 내 순서에 따라 파일 선택
            idx = int(song) - 1
            osz_files = sorted(glob.glob("data/*.osz"))
            out_dir = extract_osz(osz_files[idx])
            osu_files = sorted(glob.glob(out_dir + "/*.osu"))
            selected_osu = osu_files[0]
        except Exception:
            selected_osu = song  # 직접 경로 입력 경우

    scores = run_with_seeds(algo, render, selected_osu, lr, gamma)
    save_score(scores, "./scores", algo)

@app.command()
def evaluate_all():
    evaluate_main() 

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        app()   # argument가 있으면 typer CLI 사용
    else:
        main()  # argument 없으면 기존 메뉴모드 사용