import env

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class dset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def add(self, node):
        self.data.append(node)

class MCT:
    def __init__(self, network, optimizer, criterion):
        self.states = {}
        self.env = None
        self.search_amount = 0
        self.explore_constant = 1
        self.dataset = dset()
        self.network = env.network
        self.criterion = criterion
        self.optimizer = optimizer

    def reset(self, search_amount=50, explore_constant=1):
        self.env = env.BlackJackEnv()
        self.search_amount = search_amount
        self.explore_constant = explore_constant

    class node:
        def __init__(self, state, parent, prior_action, child=None):
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
            self.P = [self.network(parent.state, action)[1] for action in self.env.player.actions]


    def get_child(self, root, state, action):
        if state in self.states:
            root.child = self.states[state]
            root.child.parent = root
        else:
            root.child = self.node(state, root, action)
            self.states[state] = root.child
            self.search(root.child)
        return root.child

    def get_good_action(self, root):
        PUCT_alg = lambda action: self.explore_constant * root.P[action] * (sum(root.N))**0.5 / (1 + root.N[action])
        metric = lambda action: PUCT_alg(action) + self.network(root.state, action)[1]#take value from the (value, prob) pair]
        return max(self.env.player.actions, key=metric)


    def search(self, root):
        cur = root
        temp_env = self.env.copy()
        for action in self.env.player.actions:
            for _ in range(self.search_amount):
                # reset random seed here?
                # code to add maybe?

                done, reward = temp_env.step(action)
                while not done:
                    # expand the search until done
                    cur = self.get_child(cur, env.player.state, action)
                    next_action = self.get_good_action(cur)
                    done, reward = temp_env.step(next_action)

                if done:
                    self.backtrack(cur, reward, root)

    def backtrack(self, cur, reward, root):
        while cur.state != root.state:
            parent = cur.parent
            prior_action = cur.prior_action
            parent.W[prior_action] += reward
            parent.N[prior_action] += 1
            parent.Q[prior_action] = parent.W[prior_action]/parent.N[prior_action]
            parent.P = [n/sum(parent.N) for n in parent.N]
            cur = parent
        self.dataset.add(cur)

    def train(self):
        loader = DataLoader(self.dataset, batch_size=10, shuffle=True)
        for node in loader:
            input = torch.FloatTensor(node.state)
            # value prediction and prob prediction
            v, p = zip([self.network(input, action) for action in [0,1,2]])
            q, pi = tuple(node.Q), tuple(node.P)

            self.optimizer.zero_grad()
            # maybe we kindda require only using state as input for this criterion to work
            loss = self.criterion(v, p, q, pi)
            loss.backward()
            self.optimizer.step()



def simulate():
    def criterion(v, p, q, pi):
        v, p, q, pi = np.array(v), np.array(p), np.array(q), np.array(pi)
        loss = np.sum((v-p)**2) + pi * np.log(q)
        return loss

    optimizer = torch.optim.Adam(lr=0.01, weight_decay=0.01)
    tree = MCT(network, optimizer, criterion)

    training_itr = 100
    for _ in range(training_itr):
        # train_somehow
        pass