import time

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

cp_q = Queue()

CORE_NUM = 4


def cartpole_sampling(theta, sigma, N, bound=None):
    reward = cartpole_evaluate(theta.reshape(4, 2), N)
    theta_final = None
    theta_temp = theta
    count = 1
    step_bound = 0
    reward_list = []
    if bound is None:
        while True:
            theta_prime = np.random.multivariate_normal(theta_temp, sigma * np.identity(8))
            reward_prime = cartpole_evaluate(theta_prime.reshape(4, 2), N)
            if np.abs(reward_prime - reward) < 0.0001:
                theta_final = theta_prime
                break
            if reward_prime > reward:
                theta_temp = theta_prime
                reward = reward_prime
                step_bound = 0
            print('round = ', count, ': reward: ', reward)
            reward_list.append(reward)
            count += 1
            step_bound += 1
            if step_bound > 500:
                theta_final = theta_prime
                break
    else:
        for x in range(bound):
            theta_prime = np.random.multivariate_normal(theta_temp, sigma * np.identity(8))
            reward_prime = cartpole_evaluate(theta_prime.reshape(4, 2), N)
            if reward_prime > reward:
                theta_temp = theta_prime
                reward = reward_prime
            print('round = ', count, ': reward: ', reward)
            reward_list.append(reward)
            count += 1

    return theta_final, reward_list


def cartpole_evaluate(t, N):
    reward_l = []
    sublen = N // CORE_NUM
    multi_indice = (range(N)[i:i + sublen] for i in range(0, N, sublen))

    table = t.reshape(4, 2)

    processes = []

    for sublist in multi_indice:
        # concurrent_eval(theta_list, x, result_list, N)
        t = multiprocessing.Process(target=multi_cartpole_episode, args=(table, sublist))
        processes.append(t)
        t.start()

    for one_process in processes:
        one_process.join()

    while not cp_q.empty():
        reward_l.append(cp_q.get())
    return sum(reward_l) / N


def multi_cartpole_episode(table, l):
    for i in l:
        cartpole = CartPole()
        # print(i)
        cartpole.pi_params = table
        epi = CartPoleEpisode(cartpole)
        cp_q.put(epi.run_all_steps())
    return 0


def execute_cartpole(trail_num, converge_count):
    tic = time.time()
    # trail_num = 3
    # converge_count = 250
    theta = np.ones(8) * 0.25
    reward_plt_data = np.zeros((trail_num, converge_count))
    for x in range(trail_num):
        reward_plt_data[x] = np.array(cartpole_sampling(theta, 0.5, 200, bound=converge_count)[1])
    reward_std = reward_plt_data.std(0)
    reward_avg = reward_plt_data.mean(0)

    fig, ax = plt.subplots()
    plt.xlabel('episode')
    plt.ylabel('reward')
    ax.errorbar(np.array(range(converge_count)), reward_avg, yerr=reward_std, fmt='o')
    plt.savefig('cartpole_fchc.png', dpi=200)

    plt.show()

    toc = time.time()
    print('running time: ', (toc-tic) / 60, ' mins')


execute_cartpole(500, 10)
