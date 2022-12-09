import pickle

#from matplotlib.font_manager import _Weight
import qnet
import env

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Dset(Dataset):
    def __init__(self,maxlength,sample_length):
        self.maxlength = maxlength
        self.data = deque(maxlen=maxlength)
        self.new_count = 0
        self.sample_length = sample_length
        self.samples=[]

    def __len__(self):
        return self.sample_length
    
    def sample_data(self):
        '''call this at the begining of training, sample specific number of data randomly
        '''
        self.samples = random.sample(self.data, k =self.sample_length)

    def __getitem__(self, item):
        return (
            torch.FloatTensor(self.samples[item]["state"]),
            torch.FloatTensor(self.samples[item]["q"]),
            torch.FloatTensor(self.samples[item]["pi"]),
        )

    def add(self, node):
        self.data.append(node)
        self.new_count += 1


class Node:
    def __init__(self, parent, prior_action):
        self.parent = parent
        self.prior_action = prior_action
        self.children = []
        # count
        self.N = [0] * 3
        # total val
        self.W = [0] * 3
        # action val
        self.Q = [0] * 3
        # if state:
        #    self.P = network(torch.FloatTensor([state]))[0][0]


class MCT:
    def __init__(self, network, optimizer, criterion):
        self.root = None
        self.env = None
        self.search_amount = 0
        self.explore_constant = 1
        # define the maxlength of dataset, and the random sample_length
        self.dataset = Dset(maxlength=600,sample_length=256)
        self.loader= None
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer

    def reset(self, deck_num=6, search_amount=50, explore_constant=1,default_new=True):
        self.env = env.BlackJackEnv(deck_num)
        self.env.reset_env()
        if default_new:
            self.env.new_round()
        self.search_amount = search_amount
        self.explore_constant = explore_constant

    def expand_child(self, cur):
        cur.children = [Node(cur,0), Node(cur,1), Node(cur,2)]

    '''def get_good_action(self, node,state):
        # exploration 
        n_explore = [i for i in range(3) if node.N[i]<self.search_amount*0.02]
        if node==self.root and n_explore:
            return random.choice(n_explore)
        p = self.network(torch.FloatTensor([state]))[0][0].tolist()
        metric =[p[i] + node.W[i]-0.05*node.N[i] for i in range(3)]
        maxs = [i for i in range(3) if metric[i]==max(metric)]
        return random.choice(maxs)'''    
    
    def get_good_action(self, node,state):
        # exploration 
        n_explore = [i for i in range(3) if node.N[i]<self.search_amount*0.04]
        if node==self.root and n_explore:
            return random.choice(n_explore)
        p = self.network(torch.FloatTensor([state]))[0][0].tolist()
        PUCT_alg = (
            lambda action: self.explore_constant
            * p[action]
            * (sum(node.N))**0.5
            / (1 + node.N[action]**1.1)
        )
        metric =[PUCT_alg(action) + node.Q[action] for action in range(3)]
        maxs = [i for i in range(3) if metric[i]==max(metric)]
        return random.choice(maxs)
    
    def search(self):
        for _ in range(self.search_amount):
            temp_env = deepcopy(self.env)
            cur = self.root
            done = False
            # until leaf node
            while cur.children and not done:
                action = self.get_good_action(cur,temp_env.state.input())
                # print(tuple(self.env.state.input()), action)
                reward, done = temp_env.step(action)
                cur = cur.children[action]

            # if game doesn't end, expand this node
            if not done:
                self.expand_child(cur)
            # if game end, last action is double or stand, backtrack
            if done:
                self.backtrack(cur, reward, 0)

    def sample_action(self, pi, temperature):
        if temperature == 1:
            rand = random.random()
            return 0 if rand < pi[0] else 1 if rand < pi[0] + pi[1] else 2
        return max(range(3), key=lambda x: pi[x])

    def run(self):
        cur = self.root = Node(None, None)
        done = False
        temperature = 1

        while not done:
            self.search()
            pi = [n / sum(cur.N) for n in cur.N]
            if not cur.parent and not any([i==0 for i in pi]):
                dir = np.random.dirichlet(cur.N)
                pi = [0.75*pi[i]+0.25*dir[i] for i in range(3)]
            action = self.sample_action(pi, temperature)

            # print progress only
            state = self.env.state.input()
            p,v = self.network(torch.FloatTensor([state]))
            print(state, action, p[0].tolist(), v[0].item())
            print(cur.N,cur.Q)
            #

            self.dataset.add({"state": self.env.state.input(), "pi": pi})
            #temperature -= 1

            reward, done = self.env.step(action)

            newMove  = Node(cur,action)
            cur= self.root= newMove

        self.backtrack(cur, reward, train=1)
        self.root = None
        return reward

    def backtrack(self, cur, reward, train):
        if train:
            last_idx = -1
            while cur.parent:
                self.dataset.data[last_idx]['z']=reward
                self.dataset.data[last_idx]['q']=cur.parent.Q
                cur=cur.parent
                last_idx-=1
        else:
            while cur != self.root:
                parent = cur.parent
                prior_action = cur.prior_action
                parent.W[prior_action] += reward
                parent.N[prior_action] += 1
                parent.Q[prior_action] = parent.W[prior_action] / parent.N[prior_action]
                cur = cur.parent

    def train(self):
        self.network.train()
        loss_list = []
        # trim old data and restart counting
        self.dataset.new_count = 0
        self.dataset.sample_data()
        for epoch in range(2):
            for state, q, pi in self.loader:
                p, v = self.network(state)
                self.optimizer.zero_grad()
                # maybe we kindda require only using s  tate as input for this criterion to work
                loss = self.criterion(p, v, q, pi)
                loss.backward()

                self.optimizer.step()
                # print("loss: ", loss.item())
                loss_list.append(loss.item())
        self.network.eval()
        return loss_list, self.test()

    def test(self, times=100, iter=50):
        test_times = range(times)
        test_iter = range(iter)
        test_env = env.BlackJackEnv()
        test_env.reset_env()
        test_env.new_round()

        earned = [0] * times
        for t in test_times:
            for _ in test_iter:
                done = False
                while not done:
                    inp = test_env.state.input()
                    p, v = self.network(torch.FloatTensor([inp]))
                    action = torch.argmax(p[0]).item()
                    # print(inp,action.item())
                    reward, done = test_env.step(action)
                if done:
                    earned[t] += reward * 50

        print(f"\nFinal avg earned: {sum(earned)/times}")
        print("=" * 40)
        return earned


def plot(reward, mv_reward, losses, test_earned):
    # plot
    plt.plot(reward, label="reward")
    plt.plot(mv_reward, label="mv_reward")
    plt.title("Line graph")
    plt.xlabel("Iteration")
    plt.ylabel("reward")
    plt.figure()
    plt.plot(losses, label="losses")
    plt.legend()
    plt.figure()
    plt.hist(test_earned, bins=50)
    plt.show()


def simulate(new_model, training_itr, deck_num, search_amount, explore_constant,generate_plot):
    def criterion(p, v, q, pi):
        loss = (v-torch.sum(q*pi, dim=-1).unsqueeze(-1))**2 - torch.sum(pi * torch.log(p), dim=-1).unsqueeze(-1)
        return torch.sum(loss)

    network = qnet.QNet()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 24, gamma=0.8)


    tree = MCT(network, optimizer, criterion)
    tree.reset(deck_num, search_amount, explore_constant)

    PATH = "modified_param.pth"
    newData = "modified_data.pkl"
    newPlotData = "modified_plot.pkl"

    last_test=None
    if not new_model:
        checkpoint = torch.load(PATH)
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # with open(newData, 'rb') as file:
        #    tree.states, tree.dataset,last_test = pickle.load(file)
        with open(newData, "rb") as file:
            tree.dataset, last_test = pickle.load(file)
        with open(newPlotData,'rb') as file:
            plots=pickle.load(file)
            losses,rewards,test_earned,mv_rewards = plots['losses'],plots['rewards'],plots['test_earned'],plots['mv_rewards']
    else:
        losses = []
        rewards = []
        test_earned = []
        mv_rewards = [0]

    tree.loader = DataLoader(tree.dataset, batch_size=32, shuffle=True)

    exp_factor = 0.1
    for _ in range(training_itr):
        print("itr: ", _)

        reward = tree.run()
        rewards.append(reward)
        mv_reward = reward * exp_factor + mv_rewards[-1] * (1 - exp_factor)
        mv_rewards.append(mv_reward)
        print("reward: ", reward, "  mv_reward", mv_reward)
        print('='*4)
        print(
            len(tree.dataset.data),
            " dataset length  ",
            tree.dataset.new_count,
            " new count ",
        )
        if tree.dataset.new_count >= tree.dataset.sample_length and len(tree.dataset.data) == tree.dataset.maxlength:
            print('='*50,'\nTraining')
            loss_list, earned = tree.train()
            losses.extend(loss_list)
            test_earned.extend(earned)
            scheduler.step()
    cur_test = sum(tree.test(training_itr, 50)) / training_itr
    print("=" * 40)
    if not last_test or cur_test > last_test-10:
        print("saved...")
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict":scheduler.state_dict(),
            },
            PATH,
        )
        with open(newData, "wb") as file:
            pickle.dump((tree.dataset, cur_test), file)
        with open(newPlotData, "wb") as file:
            pickle.dump(
                {
                    "losses": losses,
                    "test_earned": test_earned,
                    "rewards": rewards,
                    "mv_rewards": mv_rewards,
                },
                file,
            )
    else:
        print("not saved...")

    if generate_plot:
         plot(rewards, mv_rewards, losses,test_earned)


if __name__ == "__main__":
    for _ in range(5):
        simulate(
            new_model=0,
            training_itr=2000,
            deck_num=6,
            search_amount=360,
            explore_constant=0.8,
            generate_plot=False,
        )

