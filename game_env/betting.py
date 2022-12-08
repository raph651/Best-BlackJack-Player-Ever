from copy import deepcopy

import env
import modified_mcts as mct
import qnet1 as qnet
import torch
import pyswarms as ps
import numpy as np
import random
import pickle

def gen_network():
    PATH = "modified_param2.pth"
    network = qnet.QNet()
    checkpoint = torch.load(PATH)
    network.load_state_dict(checkpoint['model_state_dict'])
    return network

def run_til_end(tree):
    root = mct.Node(parent=None, prior_action=None)
    while True:
        prob_network, _v = tree.network(torch.FloatTensor([tree.env.state.input()]))
        action = torch.argmax(prob_network).item()
        reward, done = tree.env.step(action)
        if done:
            break
        root.child = mct.Node(parent=root, prior_action=action)
        root = root.child
    return reward, tree.env

def gen_statistics(tree, sample_count):
    rewards = []
    temp_env = tree.env
    # get rewards data
    for _ in range(sample_count):
        tree.env = deepcopy(temp_env)
        reward, env = run_til_end(tree)
        rewards.append(reward)
    tree.env = temp_env

    #gen reward statistis
    return np.mean(rewards), np.var(rewards)

def gen_data(itr):
    data = []

    network = gen_network()
    tree = mct.MCT(network=network, optimizer=None, criterion=None)
    tree.reset(deck_num=6)
    for _ in range(itr):
        mean, var = gen_statistics(tree, sample_count=100)
        state = tree.env.state.bet_input + [mean, var]

        reward, env = run_til_end(tree)
        data.append((state, reward))

    return data

def betting_function(param, layers):
    #reconstruct nets
    start,end = 0,layers[0]*layers[1]
    W1 = param[start:end].reshape((layers[0], layers[1]))
    W1.astype(float)

    start, end = end, end + layers[1]
    b1 = param[start:end].reshape((layers[1],))
    b1.astype(float)

    start, end = end, end + layers[1]*layers[2]
    W2 = param[start:end].reshape((layers[1], layers[2]))
    W2.astype(float)

    start, end = end, end + layers[2]
    b2 = param[start:end].reshape((layers[2],))
    b2.astype(float)

    return [W1, b1, W2, b2]

def forward(param, x):
    #forward prop
    x = np.array(x)
    x.astype(float)
    z1 = x.dot(param[0]) + param[1]
    z1 = np.tanh(z1)
    z1 = z1.dot(param[2]) + param[3]
    z1 = np.exp(z1) / np.sum(np.exp(z1), axis=0)
    return np.argmax(z1)

def objective(p, data, money, layers):
    choice = [50, 75, 100, 125, 150, 175, 200]
    f_bet = betting_function(p, layers)
    count_turns = 0
    for state, reward in data:
        if money >= 50:
            count_turns += 1
            bet = forward(f_bet, state+[money])
            print("state: ", state+[money], " bet: ", choice[bet])
            # multiplier is 2 since double returns +-1 and hit/stand returns +-0.5
            money += 2*reward*choice[bet]
        else:
            break

    return -money, count_turns*5

def store_data(num, itr):
    try:
        with open('bet_data.pkl', 'rb') as file:
            data = pickle.load(file)
    except:
        data = []
    data.extend([gen_data(num) for i in range(itr)])
    with open('bet_data.pkl', 'wb') as file:
        pickle.dump(data, file)

def read_data():
    with open('bet_data.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

class data_generator:
    def __init__(self, num):
        self.count = 0
        self.data_num = num
        self.data = read_data()
        print(len(self.data))
        print(len(self.data[0]))
        # self.process_data()

    def process_data(self):
        print(self.data)
        self.data = [self.data[i:i+2] for i in range(0, len(self.data), 2)]
        print(self.data)

    def particle_result(self, p, money, layers):
        # if not self.data or self.count >= 50:
        #     self.data = gen_data(self.data_itr)
        #     self.count = 0
        # self.count += 1
        ans = np.array([sum(objective(ele, random.choice(self.data), money, layers)) for ele in p])
        ans.reshape(len(ans),)
        return ans


def PSO(money, data_num, training_itr, layers=[16,16,7]):
    d = data_generator(data_num)
    return
    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # Call instance of PSO
    dimensions = layers[0]*layers[1] + layers[1]*layers[2] + layers[1] + layers[2]
    print(dimensions)
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(objective_func=lambda p: d.particle_result(p, money, layers), iters=training_itr)

    data = gen_data(1000)
    print(objective(pos, data, money, layers))

store_data(1000, 50)
PSO(money=2000, data_num=1000, training_itr=10000, layers=[16,16,7])