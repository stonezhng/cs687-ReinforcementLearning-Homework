import time

# from gridworld import Grid
# from gridworld import GridEpisode
from functools import partial

from grid_plus import Grid
from grid_plus import GridEpisode
from cartpole import CartPole
from cartpole import CartPoleEpisode
# from minigrid import Grid
import numpy as np
import multiprocessing
from multiprocessing import Queue
from multiprocessing import Pool
import matplotlib.pyplot as plt
import assignment_helper as ah

cp_q = Queue()

CORE_NUM = 4


def cartpole_sampling(theta, cm, K, Ke, N, epsilon):
    theta_list = np.random.multivariate_normal(theta, cm, K)
    result_list = []

    for x in range(K):
        avg = cartpole_evaluate(theta_list[x], N)
        result_list.append((theta_list[x], avg))

    # print(sorted(result_list, key=lambda n: n[-1], reverse=True))
    elite_list = sorted(result_list, key=lambda n: n[-1], reverse=True)[: Ke]
    # print(elite_list)
    theta_final = np.zeros(8)
    cm_final = epsilon * np.identity(8)
    J_final = 0
    for t in elite_list:
        theta_final += t[0]
        cm_final += np.array([t[0] - theta]).T.dot(np.array([t[0] - theta]))
        J_final += t[1]
    theta_final /= Ke
    cm_final /= (epsilon + Ke)
    # print(cm_final)
    J_final /= Ke
    return theta_final, cm_final, J_final


def cartpole_evaluate(t, N):
    reward_l = []
    for i in range(N):
        cartpole = CartPole()
        # print(i)
        cartpole.pi_params = t.reshape(4, 2)
        epi = CartPoleEpisode(cartpole)
        reward_l.append(epi.run_all_steps())

    return sum(reward_l) / N


def multi_cartpole_episode(table, l):
    for i in l:
        cartpole = CartPole()
        # print(i)
        cartpole.pi_params = table
        epi = CartPoleEpisode(cartpole)
        cp_q.put(epi.run_all_steps())
    return 0


def cartpole_trail(bound=None):
    theta = np.ones(8) * 0.25
    cm = np.identity(8) * 0.01
    count = 1
    reward = 10
    reward_list = []
    if bound is None:
        while True:
            params = cartpole_sampling(theta, cm, 20, 10, 60, 0.001)
            print('k = ', count, ': reward: ', params[2])
            reward_list.append(params[2])

            theta = params[0]
            cm = params[1]
            count += 1
            if np.abs(reward - params[2]) <= 0.001:
                break
            reward = params[2]
    else:
        for x in range(bound):
            params = cartpole_sampling(theta, cm, 20, 10, 60, 0.5)
            # grid = Grid()
            # grid.pi_table = params[0].reshape(23, 4)
            # episode = Episode(grid)
            # episode.run_all_steps()
            print('k = ', count, ': reward: ', params[2])
            reward_list.append(params[2])

            theta = params[0]
            cm = params[1]
            count += 1

    return theta, cm, reward_list


def execute_cartpole(trail_num, converge_count):
    tic = time.time()
    # trail_num = 3
    # converge_count = 250
    reward_plt_data = np.zeros((trail_num, converge_count))
    for x in range(trail_num):
        reward_plt_data[x] = np.array(cartpole_trail(converge_count)[2])
    reward_std = reward_plt_data.std(0)
    reward_avg = reward_plt_data.mean(0)

    fig, ax = plt.subplots()
    plt.xlabel('episode')
    plt.ylabel('reward')
    ax.errorbar(np.array(range(converge_count)), reward_avg, yerr=reward_std, fmt='o')
    plt.savefig('cartpole_ce.png', dpi=200)

    plt.show()

    toc = time.time()
    print('running time: ', (toc-tic) / 60, ' mins')
    return reward_avg, reward_std


rewards, err = execute_cartpole(20, 100)
# rewards_t = np.zeros(100) + 1010
# err_t = np.zeros(100)
# rewards_t[: 10] = rewards
# err_t[: 10] = err
ah.save_cp_csvdata(rewards, err, 'ce_cartpole.csv')
