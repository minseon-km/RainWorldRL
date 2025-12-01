import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import rainworld_connector as rc

#Hyperparameters
#learning_rate = 0.0005
#gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
terminal_step = 600

#x, y, position of all four creatures
state_length = 8 #to implement, adjust this as 8
#jump, or move to four direction
action_length = 6 #same as above, adjust this as 6

class PPO(nn.Module):
    def __init__(self, lr):
        super(PPO, self).__init__()
        self.data = []
        self.fc1   = nn.Linear(state_length, 256)
        self.fc_pi = nn.Linear(256, action_length)
        self.fc_v  = nn.Linear(256,1)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        a = torch.tensor(np.array(a_lst))
        r = torch.tensor(np.array(r_lst))
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float)
        prob_a = torch.tensor(np.array(prob_a_lst))
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self, gamma):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def run_experiment_ppo(client_socket, lr=1e-3, gamma=0.99):
    model = PPO(lr)
    score = 0.0
    print_interval = 20
    scores = []
    episodes = []

    for n_epi in range(1000):
        s = rc.receive_data(client_socket)
        done = False
        episode_reward = 0.0
        timestamp = 0
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                rc.send_data(client_socket, a)
                s_prime = rc.receive_data(client_socket)
                done = True if s_prime[0] < 0 else False
                terminate = True if timestamp > terminal_step else False
                done_mask = 0.0 if done else 1.0
                if done :
                    r = 0
                elif terminate :
                    r = 100
                else :
                    r = 1
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                timestamp += 1
                episode_reward += r
                if done:
                    rc.send_data(client_socket, -1)
                    break
                if terminate :
                    print(n_epi, "success!")
                    rc.send_data(client_socket, -1)
                    break

            model.train_net(gamma)

        scores.append(episode_reward)
        episodes.append(n_epi)

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    return scores, episodes