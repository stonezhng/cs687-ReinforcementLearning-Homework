import time

# from gridworld import Grid
# from gridworld import GridEpisode
from grid_plus import Grid
from grid_plus import GridEpisode
from cartpole import CartPole
from cartpole import CartPoleEpisode
# from minigrid import Grid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, as_completed


# def softmax(m):
#     m = np.exp(m)
#     # print(self.pi_table)
#     exp_sum = np.sum(m, axis=1)
#     m /= np.array([exp_sum]).T
#     return m
#
#
# # generate theta, use theta to run N times, regenerate a new theta. repeat k times
# # select the best ke rewards and their corresponding thetas
# def grid_param_sampling(theta, cm, K, Ke, N, epsilon):
#     theta_list = np.random.multivariate_normal(theta, cm, K)
#     for x in range(K):
#         t = theta_list[x].reshape(2, 4)
#         t = softmax(t)
#         theta_list[x] = t.reshape(92)
#
#     # theta_list = []
#     # for x in range(K):
#     #     t = np.random.multivariate_normal(theta, cm)
#     #     t = t.reshape(23, 4)
#     #     t = softmax(t)
#     #     theta_list.append(t.reshape(92))
#     # print(theta_list)
#     result_list = []
#     for x in range(K):
#         # concurrent_eval(theta_list, x, result_list, N)
#         avg_reward = 0
#         for i in range(N):
#             g = Grid()
#             g.pi_table = theta_list[x].reshape(23, 4)
#             epi = GridEpisode(g)
#             epi.run_all_steps()
#             avg_reward += epi.get_discount_reward()
#         result_list.append((theta_list[x], avg_reward / N))
#
#     # print(sorted(result_list, key=lambda n: n[-1], reverse=True))
#     elite_list = sorted(result_list, key=lambda n: n[-1], reverse=True)[: Ke]
#     # print(elite_list)
#     theta_final = np.zeros(92)
#     cm_final = epsilon * np.identity(92)
#     J_final = 0
#     for t in elite_list:
#         theta_final += t[0]
#         cm_final += np.array([t[0] - theta]).T.dot(np.array([t[0] - theta]))
#         J_final += t[1]
#     theta_final /= Ke
#     cm_final /= (epsilon + Ke)
#     # print(cm_final)
#     J_final /= Ke
#     return theta_final, cm_final, J_final
#
#
# def concurrent_eval(theta_list, idx, result_list, N):
#     tk = theta_list[idx]
#     jk = grid_evaluate(tk.reshape(23, 4), N)
#     result_list.append((tk, jk))
#     # print(tk, jk)
#
#
# # run the grid using pi_parameter theta for N times and calculate the average value
# def grid_evaluate(table, N):
#     avg_reward = 0
#     for i in range(N):
#         g = Grid()
#         g.pi_table = table
#         epi = GridEpisode(g)
#         epi.run_all_steps()
#         avg_reward += epi.get_discount_reward()
#     return avg_reward / N
#
#
# def grid_trail():
#     theta = np.ones(92) * 0.25
#     cm = np.identity(92) * 0.01
#     count = 1
#     reward = 10
#     while True:
#         params = grid_param_sampling(theta, cm, 20, 10, 10, 0.001)
#         # grid = Grid()
#         # grid.pi_table = params[0].reshape(23, 4)
#         # episode = Episode(grid)
#         # episode.run_all_steps()
#         print('k = ', count, ': reward: ', params[2])
#
#         theta = params[0]
#         cm = params[1]
#         count += 1
#         if np.abs(reward - params[2]) <= 0.0001:
#             break
#         reward = params[2]
#
#     return theta, cm\

# def grid_sampling(theta, cm, K, Ke, N, epsilon):
#     theta_list = np.random.multivariate_normal(theta, cm, K)
#     result_list = []
#     for x in range(K):
#         # concurrent_eval(theta_list, x, result_list, N)
#         avg_reward = 0
#         for i in range(N):
#             grid = Grid()
#             grid.pi_params = theta_list[x].reshape(2, 4)
#             epi = GridEpisode(grid)
#             avg_reward += epi.run_all_steps()
#         result_list.append((theta_list[x], avg_reward / N))
#
#     # print(sorted(result_list, key=lambda n: n[-1], reverse=True))
#     elite_list = sorted(result_list, key=lambda n: n[-1], reverse=True)[: Ke]
#     # print(elite_list)
#     theta_final = np.zeros(8)
#     cm_final = epsilon * np.identity(8)
#     J_final = 0
#     for t in elite_list:
#         theta_final += t[0]
#         cm_final += np.array([t[0] - theta]).T.dot(np.array([t[0] - theta]))
#         J_final += t[1]
#     theta_final /= Ke
#     cm_final /= (epsilon + Ke)
#     # print(cm_final)
#     J_final /= Ke
#     return theta_final, cm_final, J_final
#
#
# def grid_evaluate(table, N):
#     avg_reward = 0
#     for i in range(N):
#         grid = Grid()
#         grid.pi_params = table
#         epi = GridEpisode(grid)
#         avg_reward += epi.run_all_steps()
#     return avg_reward / N
#
#
# def grid_trail():
#     theta = np.ones(8) * 0.25
#     cm = np.identity(8) * 0.01
#     count = 1
#     reward = 10
#     while True:
#         params = grid_sampling(theta, cm, 20, 10, 50, 0.001)
#         # grid = Grid()
#         # grid.pi_table = params[0].reshape(23, 4)
#         # episode = Episode(grid)
#         # episode.run_all_steps()
#         print('k = ', count, ': reward: ', params[2])
#
#         theta = params[0]
#         cm = params[1]
#         count += 1
#         if np.abs(reward - params[2]) <= 0.001:
#             break
#         reward = params[2]
#
#     return theta, cm

def grid_sampling(theta, cm, K, Ke, N, epsilon):
    theta_list = np.random.multivariate_normal(theta, cm, K)
    result_list = []
    for x in range(K):
        # concurrent_eval(theta_list, x, result_list, N)
        avg_reward = 0
        for i in range(N):
            grid = Grid()
            grid.pi_params = theta_list[x].reshape(23, 4)
            grid.softmax()
            epi = GridEpisode(grid)
            avg_reward += epi.run_all_steps()
        result_list.append((theta_list[x], avg_reward / N))

    # print(sorted(result_list, key=lambda n: n[-1], reverse=True))
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


def grid_evaluate(table, N):
    avg_reward = 0
    for i in range(N):
        grid = Grid()
        grid.pi_params = table
        grid.softmax()
        epi = GridEpisode(grid)
        avg_reward += epi.run_all_steps()
    return avg_reward / N


def grid_trail():
    theta = np.ones(92) * 0.25
    cm = np.identity(92) * 0.01
    count = 1
    reward = 10
    while True:
        params = grid_sampling(theta, cm, 20, 10, 50, 0.5)
        # grid = Grid()
        # grid.pi_table = params[0].reshape(23, 4)
        # episode = Episode(grid)
        # episode.run_all_steps()
        print('k = ', count, ': reward: ', params[2])

        theta = params[0]
        cm = params[1]
        count += 1
        if np.abs(reward - params[2]) <= 0.0001:
            break
        reward = params[2]

    return theta, cm


def cartpole_sampling(theta, cm, K, Ke, N, epsilon):
    theta_list = np.random.multivariate_normal(theta, cm, K)
    result_list = []
    for x in range(K):
        # concurrent_eval(theta_list, x, result_list, N)
        avg_reward = 0
        for i in range(N):
            cartpole = CartPole()
            cartpole.pi_params = theta_list[x].reshape(4, 2)
            epi = CartPoleEpisode(cartpole)
            avg_reward += epi.run_all_steps()
        result_list.append((theta_list[x], avg_reward / N))

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


def cartpole_evaluate(table, N):
    avg_reward = 0
    for i in range(N):
        cartpole = CartPole()
        cartpole.pi_params = table
        epi = CartPoleEpisode(cartpole)
        avg_reward += epi.run_all_steps()
    return avg_reward / N


def cartpole_trail():
    theta = np.ones(8) * 0.25
    cm = np.identity(8) * 0.01
    count = 1
    reward = 10
    while True:
        params = cartpole_sampling(theta, cm, 20, 10, 50, 0.001)
        # grid = Grid()
        # grid.pi_table = params[0].reshape(23, 4)
        # episode = Episode(grid)
        # episode.run_all_steps()
        print('k = ', count, ': reward: ', params[2])

        theta = params[0]
        cm = params[1]
        count += 1
        if np.abs(reward - params[2]) <= 0.001:
            break
        reward = params[2]

    return theta, cm


tic = time.time()

# theta, cm = grid_trail()
# print('optimized reward: ', grid_evaluate(theta.reshape(23, 4), 50))
# grid = Grid()
# grid.pi_params = theta.reshape(23, 4)
# grid.softmax()
# print('optimized theta: ', grid.pi_params)

theta, cm = cartpole_trail()
print('optimized reward: ', cartpole_evaluate(theta.reshape(4, 2), 50))
# print('optimized theta: ', theta.reshape(4, 2))

# pool = ThreadPoolExecutor(5)
# futures = []
# for x in range(5):
#     futures.append(pool.submit(trail, x))
toc = time.time()
print('running time: ', (toc-tic) / 60, ' mins')

