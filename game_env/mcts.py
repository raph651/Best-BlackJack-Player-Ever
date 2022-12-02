import pickle
import qnet
import env

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Dset(Dataset):
    def __init__(self):
        self.data = []
        self.new_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.FloatTensor(self.data[item]['state']),torch.FloatTensor(self.data[item]['Q']),torch.FloatTensor(self.data[item]['P'])

    def add(self, node):
        self.data.append(node)
        self.new_count += 1


class Node:
    def __init__(self, state, parent, prior_action, network, child=None):
        self.state = tuple(state)
        self.parent = parent
        self.prior_action = prior_action
        self.child = child
        # count
        self.N = [0]*3
        # total val
        self.W = [0]*3
        # action val
        self.Q = [0]*3
        # action prob
        # fill by probability returned by neural net, take only prob
        # add another indexing [0] because the output p has shape [1,3]
        # after adding, it has shape [3]
        if state:
            self.P = network(torch.FloatTensor([state]))[0][0]


class MCT:
    def __init__(self, network, optimizer, criterion):
        self.states = {}
        self.env = None
        self.search_amount = 0
        self.explore_constant = 1
        self.dataset = Dset()
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer

    def reset(self, deck_num=6, search_amount=50, explore_constant=1):
        self.env = env.BlackJackEnv(deck_num)
        self.env.reset_env()
        self.env.new_round()
        self.search_amount = search_amount
        self.explore_constant = explore_constant

    def get_child(self, root, state, action):
        if state in self.states:
            root.child = self.states[state]
            root.child.parent = root
        else:
            root.child = Node(state, root, action, self.network)
            self.states[state] = root.child
            self.search(root.child)
        return root.child

    def get_good_action(self, root):
        PUCT_alg = lambda action: self.explore_constant * root.P[action] * (sum(root.N))**0.5 / (1 + root.N[action])

        def metric(action):
            return PUCT_alg(action) + self.network(torch.FloatTensor([root.state]))[1][0].item()

        return max(self.env.player.actions, key=metric)

    def search(self, root):
        # print('search', self.env.state.input())
        temp_env = self.env
        for _ in range(self.search_amount):
            self.env = deepcopy(temp_env)
            cur = root
            while True:
                action = self.get_good_action(cur)
                # print(tuple(self.env.state.input()), action)
                reward, done = self.env.step(action)
                if done:
                    # symbolic child just for backtrack purpose
                    cur.child = Node(state=[], parent=cur, prior_action=action, network=None)
                    cur = cur.child
                    # print("reward: ", reward)
                    self.backtrack(cur, reward, root, train=0)
                    break
                cur = self.get_child(cur, tuple(self.env.state.input()), action)
            # input("end run")

        self.env = temp_env
        self.dataset.add({'state':root.state,'Q':root.Q,'P':root.P})

    def run(self, root):
        if root.state and root.state not in self.states:
            self.search(root)

        cur = root
        while True:
            action = self.get_good_action(cur)
            print(self.env.state.input(), action)
            reward, done = self.env.step(action)
            if done:
                self.backtrack(cur, reward, root, train=1)
                return reward
            cur = self.get_child(cur, tuple(self.env.state.input()), action)

    def backtrack(self, cur, reward, root, train):
        while cur.state != root.state:
            parent = cur.parent
            prior_action = cur.prior_action
            parent.W[prior_action] += reward
            parent.N[prior_action] += 1
            parent.Q[prior_action] = parent.W[prior_action]/parent.N[prior_action]
            parent.P = [n/sum(parent.N) for n in parent.N]

            cur = cur.parent
            if train:
                self.dataset.add({'state':cur.state,'Q':cur.Q,'P':cur.P})
            if cur.state != root.state:
                break

    def train(self):
        loss_list = []
        #trim old data and restart counting
        self.dataset.data = self.dataset.data[-max(1000, self.dataset.new_count+500):]
        self.dataset.new_count = 0

        loader = DataLoader(self.dataset, batch_size=20, shuffle=True)
        for epoch in range(5):
            for state,q,pi in loader:
                p,v = self.network(state)
                self.optimizer.zero_grad()
                # maybe we kindda require only using s  tate as input for this criterion to work
                loss = self.criterion(p,v,q.detach(),pi.detach())
                loss.backward()
                if np.isnan(loss.item()):
                    print("p: ", p)
                    print("v: ", v)
                    print("q: ", q)
                    print("pi: ", pi)
                    input("nan")
                self.optimizer.step()
                # print("loss: ", loss.item())
                loss_list.append(loss.item())
        return loss_list

def plot(reward, mv_reward, losses):
    # plot
    plt.title("Line graph")
    plt.xlabel("Iteration")
    plt.ylabel("reward")
    plt.figure()
    plt.plot(reward, label="reward")
    plt.plot(mv_reward, label="mv_reward")
    plt.figure()
    plt.plot(losses, label='losses')
    plt.legend()
    plt.show()

def simulate(new_model, training_itr, deck_num, search_amount, explore_constant):
    def criterion(p, v, q, pi):
        loss = (v-torch.sum(q*pi, dim=-1).unsqueeze(-1))**2 - torch.sum(pi * torch.log(p+0.01), dim=-1).unsqueeze(-1)
        return torch.sum(loss)

    network = qnet.QNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01, weight_decay=0.01)
    tree = MCT(network, optimizer, criterion)
    tree.reset(deck_num, search_amount, explore_constant)

    PATH = "param.pth"
    if not new_model:
        checkpoint = torch.load(PATH)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        with open('data.pkl', 'rb') as file:
            tree.states, tree.dataset = pickle.load(file)

    losses = []
    rewards = []
    mv_rewards = [0]
    exp_factor = 0.1
    for _ in range(training_itr):
        print("itr: ", _)
        root = Node(state=tree.env.state.input(), parent=None, prior_action=None, network=tree.network)

        reward = tree.run(root)
        rewards.append(reward)
        mv_reward = reward*exp_factor + mv_rewards[-1]*(1-exp_factor)
        mv_rewards.append(mv_reward)
        # print("cardleft: ", tree.env.state.input())
        print("reward: ", reward, "  mv_reward", mv_reward)

        print(len(tree.dataset.data), ' dataset length')
        if tree.dataset.new_count >= 200 and len(tree.dataset) >= 1000:
            loss_list = tree.train()
            losses.extend(loss_list)
    plot(rewards, mv_rewards, losses)
            # print(network.state_dict())


    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, PATH)
    with open('data.pkl', 'wb') as file:
        pickle.dump((tree.states, tree.dataset), file)

if __name__ == '__main__':
    simulate(new_model=1, training_itr=1000, deck_num=6, search_amount=300, explore_constant=0.1)