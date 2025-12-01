import rainworld_connector as rc
import modelloader as ml

from models.dqn_model import *
from models.a2c_model import *
from models.ppo_model import *

import typer
import sys
import numpy as np
import os

rspeed = 1
client_socket = None

app = typer.Typer(help="RainWorld RL training")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_with_seeds(alg_name, render, socket, lr, gamma):
    seeds = [0]
    results = []

    for seed in seeds:
        print(f"\n Running {alg_name} (seed={seed})")
        set_seed(seed)

        if alg_name == "DQN" or alg_name == "Double_DQN" or alg_name == "Dueling_DQN":
            q, q_target, scores, episodes = run_experiment_dqn(alg_name, render, socket, lr=lr, gamma=gamma)
            ml.save_model(q, './models', 'q')
            ml.save_model(q_target, './models', 'q_target')
        elif alg_name == "PPO":
            scores, episodes = run_experiment_ppo(socket, lr=lr, gamma=gamma)
        elif alg_name == "A2C":
            scores, episodes = run_experiment_a2c(socket, lr=lr, gamma=gamma)

        # 마지막 20개 에피소드 리워드의 평균
        score = np.mean(scores[-20:])

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

def main(socket):
    """Run experiments for all algorithms"""
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN", "PPO", "A2C"]
    seeds = [0, 23, 147, 575, 2768]

    print("Choose algorithm to run:")
    print("1. DQN")
    print("2. Double DQN")
    print("3. Dueling DQN")
    print("4. PPO")
    print("5. A2C")
    print("6. Run all algorithms")

    choice = input("Enter your choice (1-4): ")

    # Ask about rendering
    #render_choice = input("Enable GUI visualization? (y/n): ").lower()
    #render = render_choice in ['y', 'yes']
    render = False

    try :
        rspeed = int(input("Running speed? (xN times faster) : "))
    except :
        rspeed = 1
    print(f"Running speed : {rspeed}")
    rc.send_data(socket, rspeed)

    # lr과 gamma 기본값 설정
    lr = 1e-3
    gamma = 0.99

    if choice == "1":
        # q, q_target, scores, episodes = run_experiment_dqn("DQN", render, socket)
        scores, _ = run_with_seeds("DQN", render, socket, lr, gamma)
        ml.save_score(scores, './scores', "DQN")
    elif choice == "2":
        # q, q_target, scores, episodes = run_experiment_dqn("Double_DQN", render, socket)
        scores, _ = run_with_seeds("Double_DQN", render, socket, lr, gamma)
        ml.save_score(scores, './scores', "Double_DQN")
    elif choice == "3":
        # q, q_targe, scores, episodes = run_experiment_dqn("Dueling_DQN", render, socket)
        scores, _ = run_with_seeds("Dueling_DQN", render, socket, lr, gamma)
        ml.save_score(scores, './scores', "Dueling_DQN")
    elif choice == "4":
        # scores, episodes = run_experiment_ppo(socket)
        scores, _ = run_with_seeds("PPO", render, socket, lr, gamma)
        ml.save_score(scores, './scores', "PPO")
    elif choice == "5":
        # scores, episodes = run_experiment_a2c(socket)
        scores, _ = run_with_seeds("A2C", render, socket, lr, gamma)
        ml.save_score(scores, './scores', "A2C")
    elif choice == "6":
        # q, q_target, scores, episodes = run_experiment_dqn("DQN", render, socket)
        # q, q_target, scores, episodes = run_experiment_dqn("Double_DQN", render, socket)
        # q, q_target, scores, episodes = run_experiment_dqn("Dueling_DQN", render, socket)
        # scores, episodes = run_experiment_ppo(socket)
        # scores, episodes = run_experiment_a2c(socket)
        for alg in algorithms:
            scores, _ = run_with_seeds(alg, render, socket, lr, gamma)
            ml.save_score(scores, './scores', alg)
    else:
        print("Invalid choice, running DQN by default")
        # q, q_target, scores, episodes = run_experiment_dqn("DQN", render, socket)
        run_with_seeds("DQN", render, socket, lr, gamma)

@app.command()
def train(
    algo: str = typer.Option(default="PPO", help="Choose algorithm (DQN, PPO, etc.)"),
    lr: float = typer.Option(default=1e-3, help="Learning rate"),
    gamma: float = typer.Option(default=0.99, help="Discount factor"),
    render: bool = typer.Option(default=False, help="Enable GUI visualization"),
    socket: bool = typer.Option(default=False, help="Enable socket communication")
):
    run_with_seeds(algo, render, socket, lr, gamma)

@app.command()
def evaluate_all():
    evaluate_main()

if __name__ == '__main__':
    client_socket = rc.main_connector()
    if client_socket :
        if len(sys.argv) > 1:
            app() # argument가 있으면 typer CLI 사용
        else :
            main(client_socket) # argument 없으면 기존 메뉴모드 사용