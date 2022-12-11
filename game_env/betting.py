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
    PATH = "modified_param1.pth"
    network = qnet.QNet()
    checkpoint = torch.load(PATH)
    network.load_state_dict(checkpoint['model_state_dict'])
    return network

def run_til_end(tree):
    root = mct.Node(parent=None, prior_action=None)
    tree.env.new_round()
    while True:
        prob_network, _v = tree.network(torch.FloatTensor([tree.env.state.input()]))
        action = torch.argmax(prob_network).item()
        reward, done= tree.env.step(action,default_new=False)
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
    tree.reset(deck_num=6,default_new=False)
    for _ in range(itr):
        mean, var = gen_statistics(tree, sample_count=100)
        state = tree.env.state.cardleft + [mean, var]
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
    # print("state: ", x)
    # print("prob: ", z1)
    return np.argmax(z1)

def objective(p, data, money, layers):
    choice = [50, 75, 100, 125, 150, 175, 200]
    f_bet = betting_function(p, layers)
    count_turns = 0
    for state, reward in data:
        if money >= 50:
            count_turns += 1
            bet = forward(f_bet, state+[money])
            # print(choice[bet])
            # multiplier is 2 since double returns +-1 and hit/stand returns +-0.5
            money += 2*reward*choice[bet]
        else:
            break

    return -money, -count_turns*5

def kelly_criterion(state, money, restricted):
    p = state[-2]+0.5
    bet = (2*p-1)*money
    def find_close(bet):
        if bet < 50: return 50
        if bet > 200: return 200
        choice = [50, 75, 100, 125, 150, 175, 200]
        idx = 0
        while bet > choice[idx]:
            idx += 1
        if bet - choice[idx-1] > choice[idx] - bet:
            return choice[idx]
        else:
            return choice[idx-1]
    if not restricted:
        return bet if bet > 0 else 0
    else:
        return find_close(bet)
def perfect(restricted):
    d = data_generator()
    dataset = [random.choice(d.data) for _ in range(50)]
    result = []
    wallet = 20000
    for data in dataset:
        money = wallet
        count_turns = 0
        for state, reward in data:
            if money >= 50:
                count_turns += 1
                bet = kelly_criterion(state, money, restricted)
                # multiplier is 2 since double returns +-1 and hit/stand returns +-0.5
                money += 2*reward*bet
            else:
                break
        result.append(money + 5*count_turns)
    print(result)
    avg = sum(result)/len(result)
    print('money: ', avg)
    print('edge: ', (avg/wallet)**(1/1000)-1)

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

def dataset_stats():
    d = data_generator(0)
    win, draw, loss, profit = 0, 0, 0, 0
    for l in d.data:
        for state, res in l:
            if res > 0:
                win += 1
            elif res == 0:
                draw += 1
            else:
                loss += 1
            profit += res
    print(win, draw, loss, profit)

class data_generator:
    def __init__(self):
        self.count = 0
        self.data = read_data()
        print(len(self.data))
        print(len(self.data[0]))


    def particle_result(self, p, money, layers):
        ans = np.array([sum(objective(ele, random.choice(self.data), money, layers)) for ele in p])
        ans.reshape(len(ans),)
        print(ans)
        return ans


def PSO(money, training_itr, layers=[16,16,7], Global = True):
    d = data_generator()
    if Global:
        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.7}

        # Call instance of PSO
        dimensions = layers[0]*layers[1] + layers[1]*layers[2] + layers[1] + layers[2]
        print(dimensions)
        optimizer = ps.single.GlobalBestPSO(n_particles=2000, dimensions=dimensions, options=options)
    else:
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.7, 'k': 5, 'p': 2}
        dimensions = layers[0]*layers[1] + layers[1]*layers[2] + layers[1] + layers[2]
        optimizer = ps.single.LocalBestPSO(n_particles=2000, dimensions=dimensions, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(objective_func=lambda p: d.particle_result(p, money, layers), iters=training_itr)

    data = [random.choice(d.data) for _ in range(10)]
    result = [objective(pos, data[i], money, layers) for i in range(10)]
    print(result)
    money_left = [-r[0] for r in result]
    round_played = [(-r[1]/5) for r in result]
    print("money_left: ", money_left, sum(money_left)/len(money_left))
    print("round_played: ", round_played, sum(round_played)/len(round_played))

# for _ in range(10):
#     print(_)
#     store_data(1000, 100)

# PSO(money=20000, training_itr=300, layers=[16,16,7], Global=False)
perfect(restricted=False)