import time

from grid_plus import Grid
from grid_plus import GridEpisode
from cartpole import CartPole
from cartpole import CartPoleEpisode
# from minigrid import Grid
import numpy as np


def softmax(m):
    m = np.exp(m)
    # print(self.pi_table)
    exp_sum = np.sum(m, axis=1)
    m /= np.array([exp_sum]).T
    return m


def grid_param_sampling(theta, sigma, N):
    reward = grid_evaluate(theta.reshape(23, 4), N)
    theta_final = None
    theta_temp = theta
    count = 1
    step_bound = 0
    while True:
        theta_prime = np.random.multivariate_normal(theta_temp, sigma * np.identity(92))
        reward_prime = grid_evaluate(theta_prime.reshape(23, 4), N)
        if np.abs(reward_prime - reward) < 0.0001:
            theta_final = theta_prime
            break
        if reward_prime > reward:
            theta_temp = theta_prime
            reward = reward_prime
            step_bound = 0
        print('round = ', count, ': reward: ', reward)
        count += 1
        step_bound += 1
        if step_bound > 500:
            theta_final = theta_prime
            break
    return theta_final


# run the grid using pi_parameter theta for N times and calculate the average value
def grid_evaluate(table, N):
    avg_reward = 0
    for i in range(N):
        g = Grid()
        g.pi_params = table
        g.softmax()
        epi = GridEpisode(g)
        avg_reward += epi.run_all_steps()
    return avg_reward / N


def cartpole_sampling(theta, sigma, N):
    reward = cartpole_evaluate(theta.reshape(4, 2), N)
    theta_final = None
    theta_temp = theta
    count = 1
    while True:
        theta_prime = np.random.multivariate_normal(theta_temp, sigma * np.identity(8))
        reward_prime = cartpole_evaluate(theta_prime.reshape(4, 2), N)
        if np.abs(reward_prime - reward) < 0.0001:
            theta_final = theta_prime
            break
        if reward_prime > reward:
            theta_temp = theta_prime
            reward = reward_prime
        print('round = ', count, ': reward: ', reward)
        count += 1
    return theta_final


def cartpole_evaluate(table, N):
    avg_reward = 0
    for i in range(N):
        cartpole = CartPole()
        cartpole.pi_params = table
        epi = CartPoleEpisode(cartpole)
        avg_reward += epi.run_all_steps()
    return avg_reward / N


tic = time.time()

theta = np.ones(92) * 0.25
theta_f = grid_param_sampling(theta, 0.5, 200)
grid = Grid()
grid.pi_params = theta_f.reshape(23, 4)
grid.softmax()
episode = GridEpisode(grid)

print('optimized reward: ', episode.run_all_steps())
print('optimized theta: ', theta_f.reshape(23, 4))

# theta = np.ones(8) * 0.25
# theta_f = cartpole_sampling(theta, 0.5, 500)
# cartpole = CartPole()
# cartpole.pi_params = theta_f.reshape(4, 2)
# episode = CartPoleEpisode(cartpole)

# print('optimized reward: ', episode.run_all_steps())
# print('optimized theta: ', theta_f.reshape(4, 2))

toc = time.time()
print('running time: ', (toc - tic) / 60, ' mins')

