import pickle
import qnet
import env

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
        return self.data[item]

    def add(self, node):
        self.data.append(node)


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
        def metric(action):
            return PUCT_alg(action) + self.network(torch.FloatTensor([root.state]))[1][0][0]
        #metric = lambda action: PUCT_alg(action) + self.network(torch.FloatTensor([root.state]))[1][0][0]#take value from the (value, prob) pair]

        return max(self.env.player.actions, key=metric)

    def search(self, root, search_amount):
        cur = root
        temp_env = self.env
        for _ in range(search_amount):
            # reset random seed here?
            # code to add maybe?
            self.env = deepcopy(temp_env)
            while True:
                action = self.get_good_action(cur)
                print(tuple(self.env.state.input()))
                done, reward = self.env.step(action)
                if done:
                    print('hey')
                    self.backtrack(cur, reward, root)
                    break
                cur = self.get_child(cur, tuple(self.env.state.input()), action)

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
        #trim old data and restart counting
        self.dataset.data = self.dataset.data[-1000:]
        self.dataset.new_count = 0

        loader = DataLoader(self.dataset, batch_size=20, shuffle=True)
        for epoch in range(5):
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

    network = qnet.QNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01, weight_decay=0.01)
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
        print(_)
        tree.env.new_round()
        root = Node(state=tree.env.state.input(), parent=None, prior_action=None, network=tree.network)
        if root.state not in tree.states:
            tree.search(root, 40)
        tree.search(root, 1)
        print("ga")
        input()
        if tree.dataset.new_count >= 200 and len(tree.dataset) >= 400:
            tree.train()

    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, PATH)
    with open('data.pkl', 'wb') as file:
        pickle.dump(tree.dataset.data, file)

if __name__ == '__main__':
    simulate(new_model=0, training_itr=1000, deck_num=6, search_amount=100, explore_constant=1)