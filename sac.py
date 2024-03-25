import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Hyperparameters
lr_pi           = 0.0003
lr_q            = 0.0003
init_alpha      = 0.01
gamma           = 0.99
batch_size      = 256
buffer_limit    = 1000000
tau             = 0.005 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        s_list = torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
        a_list = torch.tensor(np.array(a_lst), dtype=torch.float).to(device)
        r_list = torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
        s_prime_list = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
        done_mask_list = torch.tensor(np.array(done_mask_lst), dtype=torch.float).to(device)
        return s_list, a_list, r_list, s_prime_list, done_mask_list

    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate, state_space, action_space):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc_mu = nn.Linear(256,action_space)
        self.fc_std  = nn.Linear(256,action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        real_log_prob = real_log_prob.sum(-1, keepdim=True)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate, state_space, action_space):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_space, 128)
        self.fc_a = nn.Linear(action_space,128)
        self.fc_cat = nn.Linear(256,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target
    
def main():
    env = gym.make('Ant-v4')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    memory = ReplayBuffer()
    q1 = QNet(lr_q, state_space, action_space).to(device)
    q2 = QNet(lr_q, state_space, action_space).to(device)
    q1_target = QNet(lr_q, state_space, action_space).to(device)
    q2_target = QNet(lr_q, state_space, action_space).to(device)
    pi = PolicyNet(lr_pi, state_space, action_space).to(device)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 10
    total_steps = 0

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            a, log_prob= pi(torch.from_numpy(s).float().to(device))
            s_prime, r, done, truncated, info = env.step(a.detach().cpu().numpy())
            memory.put((s, a.detach().cpu().numpy(), r, s_prime, done))
            score +=r
            s = s_prime
            total_steps += 1

            if r < min_reward:
                min_reward = r
            if r > max_reward:
                max_reward = r

            if memory.size()>1000:
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)
                
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}, total steps: {:.1f}M".format(n_epi, score/print_interval, pi.log_alpha.exp()))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()