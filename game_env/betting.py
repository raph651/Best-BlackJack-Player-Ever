from copy import deepcopy

import env
import mcts2 as mct
import qnet
import torch
import pyswarms as ps
import numpy as np
from matplotlib import pyplot as plt


def gen_network():
    PATH = "param.pth"
    network = qnet.QNet()
    checkpoint = torch.load(PATH)
    network.load_state_dict(checkpoint['model_state_dict'])
    return network

def run_til_end(tree):
    root = mct.Node(state=tree.env.state.input(), parent=None, prior_action=None, network=tree.network)
    while True:
        prob_network, _v = tree.network(torch.FloatTensor([root.state]))
        action = torch.argmax(prob_network).item()
        reward, done = tree.env.step(action)
        if done:
            break
        root.child = mct.Node(state=tree.env.state.input(), parent=root, prior_action=action, network=tree.network)
        root = root.child
    return reward, tree.env

def gen_statistics(tree, sample_count):
    rewards = []
    temp_env = tree.env
    # get rewards data
    for _ in range(sample_count):
        tree.env = deepcopy(temp_env)
        reward, env = run_til_end(tree)
        rewards.append(rewards)
    tree.env = temp_env

    #gen reward statistis
    return np.mean(rewards), np.var(rewards)

def gen_data(itr):
    data = []

    network = gen_network()
    tree = mct.MCT(network=network, optimizer=None, criterion=None)
    for _ in range(itr):
        mean, var = gen_statistics(tree, sample_count=100)
        state = tree.env.state.bet_input + [mean, var]

        reward, env = run_til_end(tree)
        data.append((state, reward))

    return data

def betting_function(param, layers=[16, 16, 1]):
    #reconstruct nets
    start,end = 0,layers[0]*layers[1]
    W1 = param[start:end].reshape((layers[0], layers[1]))

    start, end = end, end + layers[1]
    b1 = param[start:end].reshape((layers[1],))

    start, end = end, end + layers[1]*layers[2]
    W2 = param[start:end].reshape((layers[1], layers[2]))

    start, end = end, end + layers[2]
    b2 = param[start:end].reshape((layers[2],))

    return [W1, b1, W2, b2]

def forward(param, x):
    #forward prop
    z1 = x.dot(param[0]) + param[1]
    z1 = np.tanh(z1)
    z1 = z1.dot(param[2]) + param[3]
    z1 = np.tanh(z1)
    return z1

def objective(p, data, money=1000):
    f_bet = betting_function(p, layers=[16, 16, 1])
    count_turns = 0
    for state, reward in data:
        if money >= 50:
            count_turns += 1
            bet = forward(f_bet, state+[money])

            # multiplier is 2 since double returns +-1 and hit/stand returns +-0.5
            money += 2*reward*bet
        else:
            break

    return money, count_turns

def PSO(money, data_itr, layers=[16,16,1]):
    data = gen_data(data_itr)

    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # Call instance of PSO
    dimensions = layers[0]*layers[1] + layers[1]*layers[2] + layers[1] + layers[2]
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(objective_func=lambda p: objective(p, data, money), iters=1000)

PSO(2000, 10000, layers=[16,16,1])