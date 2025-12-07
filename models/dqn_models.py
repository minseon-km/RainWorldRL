from song_manager import select_song
from beatmap import extract_osz, parse_osu_file
from environment import RhythmEnv
from evaluate import main as evaluate_main

# libraries
import collections
import random
import numpy as np

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# hyperparameters
#learning_rate = 0.0005
#gamma = 0.90
buffer_limit = 50000        # size of replay buffer
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)    # double-ended queue

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()

class DuelingQnet(nn.Module):
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = F.relu(self.fc_value(x))
        a = F.relu(self.fc_adv(x))
        v = self.value(v)
        a = self.adv(a)
        a_avg = torch.mean(a, dim=1, keepdim=True)  # Fixed dimension issue
        q = v + a - a_avg
        return q

    def sample_action(self, obs, epsilon):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else:
            return out.argmax().item()

def train_dqn(q, q_target, memory, optimizer, gamma):
    """Standard DQN training"""
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1,a)

        # DQN
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_double_dqn(q, q_target, memory, optimizer, gamma):
    """Double DQN training"""
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)

        # Double DQN
        argmax_Q = q(s_prime).max(1)[1].unsqueeze(1)
        max_q_prime = q_target(s_prime).gather(1, argmax_Q)

        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def run_experiment_dqn(algorithm_type="DQN", render=False, selected_osu=None, lr=1e-3, gamma=0.99):
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

    if algorithm_type == "Dueling_DQN":
        q = DuelingQnet()
        q_target = DuelingQnet()
        train_fn = train_double_dqn  # Dueling uses Double DQN training
    else:
        q = Qnet()
        q_target = Qnet()
        train_fn = train_dqn if algorithm_type == "DQN" else train_double_dqn

    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=lr)

    scores = []
    episodes = []

    for n_epi in range(3000):
        epsilon = max(0.01, 0.1 * np.exp(-n_epi / 300))
        s, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, terminated, truncated, info = env.step(a)
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            episode_reward += r
            if done:
               break

        scores.append(episode_reward)
        episodes.append(n_epi)
            
        if memory.size()>2000:
            train_fn(q, q_target, memory, optimizer, gamma)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                                n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    env.close()
    return q, q_target, scores, episodes