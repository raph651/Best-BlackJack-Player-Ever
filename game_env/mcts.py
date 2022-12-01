import pickle

import env

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Dset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def add(self, node):
        self.data.append(node)


class Node:
    def __init__(self, state, parent, prior_action, network, child=None):
        self.state = state
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
        self.P = network(state)[1]


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
        self.search_amount = search_amount
        self.explore_constant = explore_constant

    def get_child(self, root, state, action):
        if state in self.states:
            root.child = self.states[state]
            root.child.parent = root
        else:
            root.child = Node(state, root, action, self.network)
            self.states[state] = root.child
            self.search(root.child, self.search_amount)
        return root.child

    def get_good_action(self, root):
        PUCT_alg = lambda action: self.explore_constant * root.P[action] * (sum(root.N))**0.5 / (1 + root.N[action])
        metric = lambda action: PUCT_alg(action) + self.network(root.state, action)[0][0]#take value from the (value, prob) pair]
        return max(self.env.player.actions, key=metric)

    def search(self, root, search_amount):
        cur = root
        temp_env = self.env.copy()
        for _ in range(search_amount):
            # reset random seed here?
            # code to add maybe?
            action = self.get_good_action(cur)
            done, reward = temp_env.step(action)
            while not done:
                # expand the search until done
                cur = self.get_child(cur, temp_env.state.input, action)
                action = self.get_good_action(cur)
                done, reward = temp_env.step(action)

            if done:
                self.backtrack(cur, reward, root)
        self.dataset.add(root)

    def backtrack(self, cur, reward, root):
        while cur.state != root.state:
            parent = cur.parent
            prior_action = cur.prior_action
            parent.W[prior_action] += reward
            parent.N[prior_action] += 1
            parent.Q[prior_action] = parent.W[prior_action]/parent.N[prior_action]
            parent.P = [n/sum(parent.N) for n in parent.N]
            cur = parent

    def train(self):
        loader = DataLoader(self.dataset, batch_size=10, shuffle=True)
        for node in loader:
            # value prediction and prob prediction
            v, p = self.network(torch.FloatTensor(node.state))
            q, pi = node.Q, node.P

            self.optimizer.zero_grad()
            # maybe we kindda require only using state as input for this criterion to work
            loss = self.criterion(v.numpy(), np.array(p), q.numpy(), np.array(pi))
            loss.backward()
            self.optimizer.step()


def simulate(new_model, training_itr, deck_num, search_amount, explore_constant):
    def criterion(v, p, q, pi):
        loss = (v-q)**2 + pi * np.log(p)
        return loss

    optimizer = torch.optim.Adam(lr=0.01, weight_decay=0.01)
    network =
    tree = MCT(network, optimizer, criterion)
    tree.reset(deck_num, search_amount, explore_constant)

    PATH = "param.pth"
    if new_model:
        checkpoint = torch.load(PATH)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        with open('data.pkl', 'rb') as file:
            tree.dataset.data = pickle.load(file)

    for _ in range(training_itr):
        tree.env.new_round()
        root = Node(state=tree.env.state.input, parent=None, prior_action=None, network=tree.network)
        tree.search(root, 1)

    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, PATH + str(j))
    with open('data.pkl', 'wb') as file:
        pickle.dump(tree.dataset.data, file)

if __name__ == '__main__':
    simulate(new_model=1, training_itr=1000, deck_num=6, search_amount=100, explore_constant=1)