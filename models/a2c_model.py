from song_manager import select_song
from beatmap import extract_osz, parse_osu_file
from environment import RhythmEnv
from evaluate import main as evaluate_main

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from collections import deque
import random
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_interval = 20

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
    def selection_action(self, state):
        with torch.no_grad():
            prob = self.forward(state)
            b = Bernoulli(prob)
            action = b.sample()
        return action.item()

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Memory:
    def __init__(self, memory_size:int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)
    
    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer): 
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand+batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]
        
    def clear(self):
        self.buffer.clear()

        
def run_experiment_a2c(algorithm_type="PPO", render=False, selected_osu=None, lr=1e-3, gamma=0.99):
    """Run experiment with specified algorithm type"""
    print(f"\n=== Running {algorithm_type} Experiment ===")
    print(f"Algorithm Type: {algorithm_type}")
    print(f"Render: {render}")
    print(f"Selected Song: {selected_osu}")
    print(f"Learning Rate: {lr}")
    print(f"Gamma: {gamma}")

    # selected_osu가 None이면 자동으로 선택하게 함 
    if selected_osu is None:
        from song_manager import select_song
        selected_osu = select_song("data")
        if not selected_osu:
            print("No song selected. Exiting experiment.")
            return

    notes = parse_osu_file(selected_osu)
    print(f"Loaded beatmap: {selected_osu} | Notes: {len(notes)}")
    env = RhythmEnv(notes)

    # if render:
    #     notes = parse_osu_file(selected_osu)
    #     print(f"Loaded beatmap: {selected_osu} | Notes: {len(notes)}")
    #     env = RhythmEnv(notes)

    policy = PolicyNetwork().to(device)
    value = ValueNetwork().to(device)
    # optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    # value_optim = torch.optim.Adam(value.parameters(), lr=3e-4)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optim = torch.optim.Adam(value.parameters(), lr=lr)
    #gamma = 0.99
    memory = Memory(200)
    batch_size = 32
    k = 0

    scores = []
    episodes = []

    epoch = 0
    while epoch < 3000 :
        state, _ = env.reset()
        episode_reward = 0

        while True:
            k += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy.selection_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            done = (terminated or truncated)
            episode_reward += reward
            #memory.add((state, next_state, action, reward, done))
            memory.add((state, next_state, action, reward/100.0, done))

            if k == batch_size:
                k = 0
                experiences = memory.sample(batch_size)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*experiences)
                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
                with torch.no_grad():
                    value_target = batch_reward + gamma * (1 - batch_done) * value(batch_next_state)
                    advantage = value_target - value(batch_state)
                prob = policy(batch_state)
                b = Bernoulli(prob)
                log_prob = b.log_prob(batch_action)
                loss = - log_prob * advantage
                loss = loss.mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
                value_loss = F.mse_loss(value_target, value(batch_state))
                value_optim.zero_grad()
                value_loss.backward()
                value_optim.step()
                
                memory.clear()

                

            if done:
                break
            state = next_state

        if episode_reward == 0 :
            continue
        epoch += 1
        scores.append(episode_reward)
        episodes.append(epoch)
        
        if epoch % 20 == 0:
            print("n_episode :{}, score : {:.1f}".format(epoch, episode_reward))
    return scores, episodes