import gymnasium as gym
from collections import deque
import random
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical # action 2개일 경우 Bernoulli, 3개 이상일 경우 Categorical
import torch.nn.functional as F

import rainworld_connector as rc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#gamma = 0.99
batch_size = 32
terminal_step = 600

#x, y, position of all four creatures
state_length = 8 #to implement, adjust this as 8
#jump, or move to four direction
action_length = 6 #same as above, adjust this as 6

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_length, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_length)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1) # action이 6개이므로 softmax 사용해 확률 출력
        return x

    def selection_action(self, state):
        with torch.no_grad():
            prob = self.forward(state)
            m = Categorical(prob) # Categorical 분포 생성
            action = m.sample()
        return action.item(), m.log_prob(action) # 행동과 로그 확률 반환

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_length, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1) # 출력 크기: 1 (가치)

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


def run_experiment_a2c(client_socket, lr=1e-3, gamma=0.99):
    policy = PolicyNetwork().to(device)
    value = ValueNetwork().to(device)
    # optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    # value_optim = torch.optim.Adam(value.parameters(), lr=3e-4)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optim = torch.optim.Adam(value.parameters(), lr=lr)
    memory = Memory(200)
    k = 0
    scores = []
    episodes = []

    epoch = 0
    while epoch < 1500 :
        # state, _ = env.reset()
        state = rc.receive_data(client_socket)
        episode_reward = 0
        timestamp = 0

        while True:
            k += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, log_prob = policy.selection_action(state_tensor) # action과 log_prob을 함께 받음
            rc.send_data(client_socket, action)
            # next_state, reward, terminated, truncated, _ = env.step(int(action))
            next_state = rc.receive_data(client_socket)
            #done = (terminated or truncated)
            done = True if next_state[0] < 0 else False
            terminate = True if timestamp > terminal_step else False
            if done:
                reward = -10
            elif terminate:
                reward = 10
            else:
                reward = 1
            episode_reward += reward
            memory.add((state, next_state, action, reward/100.0, done, log_prob.item())) # log_prob을 메모리에 함께 저장

            timestamp += 1

            if k == batch_size:
                k = 0
                experiences = memory.sample(batch_size)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done, batch_log_prob = zip(*experiences)
                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
                with torch.no_grad():
                    value_target = batch_reward + gamma * (1 - batch_done) * value(batch_next_state)
                    advantage = value_target - value(batch_state)
                probs = policy(batch_state)
                current_dist = Categorical(probs)
                #batch_log_prob = torch.FloatTensor(batch_log_prob).unsqueeze(1).to(device)
                batch_log_prob = current_dist.log_prob(batch_action.squeeze(1)).unsqueeze(1)
                loss = - batch_log_prob * advantage
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
                rc.send_data(client_socket, -1)
                break
            if terminate:
                print(epoch, "success!")
                rc.send_data(client_socket, -1)
                break
            state = next_state
        if episode_reward == -10 :
            continue
        scores.append(episode_reward)
        episodes.append(epoch)

        if epoch % 10 == 0:
            print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
        epoch += 1

    return scores, episodes