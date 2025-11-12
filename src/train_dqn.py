# libraries
import gymnasium as gym
import collections
import random
import numpy as np

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import rainworld_connector as rc

# hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000        # size of replay buffer
batch_size = 32
#x, y, position of all four creatures
state_length = 8 #to implement, adjust this as 8
#jump, or move to four direction
action_length = 5 #same as above, adjust this as 5

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
        self.fc1 = nn.Linear(state_length, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_length-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,action_length-1)
        else:
            return out.argmax().item()

class DuelingQnet(nn.Module):
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.fc1 = nn.Linear(state_length, 128)
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, action_length)

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
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,action_length)
        else:
            return out.argmax().item()

def train_dqn(q, q_target, memory, optimizer):
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

def train_double_dqn(q, q_target, memory, optimizer):
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

def run_experiment(socket, algorithm_type="DQN", render=False):
    """Run experiment with specified algorithm type"""
    print(f"\n=== Running {algorithm_type} Experiment ===")

    # Initialization
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

    print_interval = 3
    score = 0.0
    r = 1 #reward per timestamp
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    # for 3000 episodes
    for n_epi in range(3000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = rc.receive_data(socket)
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            print(a)
            rc.send_data(socket, a)
            s_prime = rc.receive_data(socket)
            print(s_prime)
            done = True if s_prime[0] < 0 else False
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                print(n_epi, "episode done!")
                rc.send_data(socket, -1)
                break

        if memory.size()>2000:
            train_fn(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    return q, q_target

def main(socket):
    """Run experiments for all three algorithms"""
    algorithms = ["DQN", "Double_DQN", "Dueling_DQN"]

    print("Choose algorithm to run:")
    print("1. DQN")
    print("2. Double DQN")
    print("3. Dueling DQN")
    print("4. Run all algorithms")

    #choice = input("Enter your choice (1-4): ")

    choice = 1

    # Ask about rendering
    #render_choice = input("Enable GUI visualization? (y/n): ").lower()
    #render = render_choice in ['y', 'yes']
    render = False

    if choice == "1":
        run_experiment(socket, "DQN", render)
    elif choice == "2":
        run_experiment(socket, "Double_DQN", render)
    elif choice == "3":
        run_experiment(socket, "Dueling_DQN", render)
    elif choice == "4":
        for alg in algorithms:
            run_experiment(socket, alg, render)
    else:
        print("Invalid choice, running DQN by default")
        run_experiment(socket, "DQN", render)

if __name__ == '__main__':
    client_socket = rc.main_connector()
    main(client_socket)
