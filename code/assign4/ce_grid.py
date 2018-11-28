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


def grid_sampling(theta, cm, K, Ke, N, epsilon):
    theta_list = np.random.multivariate_normal(theta, cm, K)

    result_list = []

    for x in range(K):
        avg = grid_evaluate(theta_list[x], N)
        result_list.append((theta_list[x], avg))

    elite_list = sorted(result_list, key=lambda n: n[-1], reverse=True)[: Ke]
    # print(elite_list)
    theta_final = np.zeros(92)
    cm_final = epsilon * np.identity(92)
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


def grid_evaluate(t, N):
    reward_l = []
    table = t.reshape(23, 4)

    for i in range(N):
        # concurrent_eval(theta_list, x, result_list, N)
        grid = Grid()
        # print(i)
        grid.pi_params = table
        grid.softmax()
        epi = GridEpisode(grid)
        reward_l.append(epi.run_all_steps())
    return sum(reward_l) / N


def grid_trail(bound=None):
    theta = np.ones(92) * 0.25
    cm = np.identity(92) * 0.01
    count = 1
    reward = 10
    reward_diff = []
    reward_list = []
    if bound is None:
        while True:
            params = grid_sampling(theta, cm, 20, 10, 60, 0.5)
        # grid = Grid()
        # grid.pi_table = params[0].reshape(23, 4)
        # episode = Episode(grid)
        # episode.run_all_steps()
            print('k = ', count, ': reward: ', params[2])
            reward_list.append(params[2])

            theta = params[0]
            cm = params[1]
            count += 1

            if not reward_diff:
                reward_diff.append(np.abs(reward - params[2]))
            elif len(reward_diff) < 3:
                reward_diff.append(np.abs(reward - params[2]))
            else:
                reward_diff.pop(0)
                reward_diff.append(np.abs(reward - params[2]))
        # print(sum((reward_diff)))
            if sum(reward_diff) <= 0.01:
                break
            reward = params[2]
    else:
        for x in range(bound):
            params = grid_sampling(theta, cm, 20, 10, 60, 0.5)
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


def execute_grid(trail_num, converge_count):
    tic = time.time()
    # trail_num = 3
    # converge_count = 250
    reward_plt_data = np.zeros((trail_num, converge_count))
    for x in range(trail_num):
        reward_plt_data[x] = np.array(grid_trail(converge_count)[2])
    reward_std = reward_plt_data.std(0)
    reward_avg = reward_plt_data.mean(0)

    fig, ax = plt.subplots()

    plt.xlabel('episode')
    plt.ylabel('reward')
    ax.errorbar(np.array(range(converge_count)), reward_avg, yerr=reward_std, fmt='o')
    plt.savefig('grid_ce.png', dpi=200)

    plt.show()

    toc = time.time()
    print('running time: ', (toc-tic) / 60, ' mins')
    return reward_avg, reward_std


rewards, err = execute_grid(20, 100)
ah.save_cp_csvdata(rewards, err, 'ce_grid.csv')

# print('optimized theta: ', grid.pi_params)

# theta, cm = cartpole_trail()
# print('optimized reward: ', cartpole_evaluate(theta.reshape(4, 2), 50))
# print('optimized theta: ', theta.reshape(4, 2))

# pool = ThreadPoolExecutor(5)
# futures = []
# for x in range(5):
#     futures.append(pool.submit(trail, x))
