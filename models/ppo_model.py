from song_manager import select_song
from beatmap import extract_osz, parse_osu_file
from environment import RhythmEnv
from evaluate import main as evaluate_main

# libraries
import numpy as np

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

#Hyperparameters
#learning_rate = 0.0005
#gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, learning_rate=1e-3):
        super(PPO, self).__init__()
        self.data = []
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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

def run_experiment_ppo(algorithm_type="PPO", render=False, selected_osu=None, lr=1e-3, gamma=0.99):
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

    model = PPO(lr)
    score = 0.0
    print_interval = 20

    scores = []
    episodes = []

    for n_epi in range(3000):
        s, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                episode_reward += r
                if done:
                    break

            model.train_net(gamma)
        scores.append(episode_reward)
        episodes.append(n_epi)

        if n_epi%print_interval==0 and n_epi!=0:
            print("n_episode :{}, score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()
    return scores, episodes