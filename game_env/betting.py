from copy import deepcopy

import env
import modified_mcts as mct
import qnet as qnet
import torch
import pyswarms as ps
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

def gen_network():
    PATH = "modified_param_60_biggernet.pth"
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

def cross_entropy(money=20000, training_itr=300, layers=[16,16,7], p_elite =0.2):
    '''Searching over parameter space using the proportional cross entropy method (PCEM)
    money (int): initial money
    training_itr (int): number of iterations
    layers (list): number of neurons per layer
    p_elite (float): top p_elite are kept in each iteration
    '''

    data = read_data()
    print(len(data))
    print(len(data[0]))


    dimensions = layers[0]*layers[1] + layers[1]*layers[2] + layers[1] + layers[2]
    print(dimensions)

    # sample multivariate parameter samples from normal distribution
    gen_params = lambda n_samples,mean,cov: np.random.multivariate_normal(mean,cov,n_samples)

    eps = 1e-5 # avoid division by 0
    noise = 5e-2 # noise added to covariance
    '''Algorithm
    1.  Evaluate each sample
    2.  Keep top p_elite samples, 
    3.  Estimate parameters of new distribution from elite set 
      - mean_new = sum of elite params/elite_size , cov_new = (param-mean_old)(param-mean_old)^T/elite_size
    4.  Resample
    '''
    means=[]
    covs=[]
    for iter in range(training_itr):
        n_samples = int(np.round(1000*np.exp(-0.005*iter))) # decaying number of samples
        elite_size = int(np.round(p_elite*n_samples)) # number of elite samples to keep
        print('='*5)
        print(f'iter :{iter}', n_samples, elite_size)
        # generate parameters based on new mean and covariance
        params = gen_params(n_samples,mean,cov) if iter!=0 else np.random.uniform(low=0.0,high=1.0,size=(n_samples,dimensions))
        rand_set = np.random.choice(len(data)) # a random set from data
        objectives = np.array([objective(p,data[rand_set],money,layers) for p in params])
        sum_obj = objectives.sum(axis=1)
        print(objectives.shape, objectives[:,0].max(), objectives[:,0].min(), objectives[:,1].max()/5,objectives[:,1].min()/5)
        elite_idxs = sum_obj.argsort()[:elite_size]
        elite_samples = params[elite_idxs]
        #objectives = objectives[elite_idxs]
        #print(objectives.shape,' obj')
        #M = np.max(objectives)
        #m = np.min(objectives)
        # proportional objectives
        #objectives =  ( objectives - m)/( M - m + eps)
        #elite_idxs = objectives.argsort()[:elite_size]
        #elite_samples = params[elite_idxs]
        mean = elite_samples.mean(axis=0)
        cov = ((elite_samples-mean).T@(elite_samples-mean))/elite_size+ noise*np.exp(-0.005*iter)
        means.append(mean)
        covs.append(cov)
        print(f'elite objectives, mean money left: {-objectives[elite_idxs,0].mean()}, std money left: {objectives[elite_idxs,0].std()}')
        print(f'elite objectives, mean round played: {-objectives[elite_idxs,1].mean()/5}, std round played: {objectives[elite_idxs,1].std()/25}')

        '''stream parameters, and use heap for storing the 
        top p_elite samples, since the sample size is huge
        elite_samples = []
        for n in range(n_samples):
            param = gen_params(mean,cov)
            res = objective(p,data,money,layers)
            if n<elite_size:
                elite_samples.append((res,n))
                # heapify the first elite_size samples
                if n==elite_size-1:
                    heapq.heapify(elite_samples)
                    print('heapified')
            # if res larger than the smallest element in elite_samples ( elite_samples[0][0] )
            # replace it
            elif res>elite_samples[0][0]:
                heapq.heapreplace(elite_samples,(res,n))
        '''
    return np.array(means), np.array(covs)

def test(param, layers=[16,16,7],money=20000,times=1000, iter=50):
    '''test a param/particle using the action model and bet model
    param: parameter or particle to test
    layers: the given layers, default to [16,16,7]
    '''
    test_times = range(times)
    test_iter = range(iter)

    action_net = gen_network()
    tree = mct.MCT(network=action_net, optimizer=None, criterion=None)

    bet_net = betting_function(param,layers)
    choice = [50, 75, 100, 125, 150, 175, 200]

    money_left = [money] * times
    count_turns = [0] * times

    for t in test_times:
        tree.reset(deck_num=6,default_new=False)
        for _ in test_iter:
            mean, var = gen_statistics(tree, sample_count=100)
            state = tree.env.state.cardleft + [mean, var]
            bet = choice[forward(bet_net,state+[money])]
            tree.env.new_round()            
            done = False
            while not done:
                inp = tree.env.state.input()
                p, v = action_net(torch.FloatTensor([inp]))
                action = torch.argmax(p[0]).item()
                reward, done = tree.env.step(action,default_new=False)
            money_left[t] += 2*reward * bet
            count_turns[t]+=1
            if money_left[t]<50:
                break

    print(f"\nFinal avg money left: {sum(money_left)/times}, avg turns played: {sum(count_turns)/times}")
    print("=" * 40)
    return money_left,count_turns

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
#perfect(restricted=False)

if __name__ =='__main__':
    _train_new_ =False
    _test_ = True
    
    if _train_new_:
        means,covs = cross_entropy(money=20000, training_itr=10, layers=[16,16,7])
        CEM_data_path ='./cem_data.pkl'
        with open(CEM_data_path, 'wb') as f:
            pickle.dump((means,covs),f)

    if _test_:
        CEM_data_path ='./cem_data.pkl'
        with open(CEM_data_path,'rb') as f:
            means,covs = pickle.load(f)
    # generate few parameters based on the last mean and covariance,
    # and use the mean of them for testing
        n_samples =1000
        last_mean , last_cov = means[-1,:],covs[-1,:,:]
        
        params = np.random.multivariate_normal(last_mean,last_cov,n_samples)
        param = params.mean(axis=0)
        
        money_left, count_turns = test(param)

        plt.plot(money_left,label='money left')
        plt.title('money left')
        plt.legend()
        plt.figure()
        plt.plot(count_turns, label='count turns')
        plt.title('count turns')
        plt.legend()
        plt.show()