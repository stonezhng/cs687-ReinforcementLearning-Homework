from grid_plus import Grid
from grid_plus import GridEpisode
from mountaincar import MountainCar
from mountaincar import MountainCarEpisode
import function_approximation as fa
import assignment_helper as ah
import estimation

import numpy as np
import matplotlib.pyplot as plt


# qlearning for one trial
# use decaying eps
def reinforce_grid(lr, eps, epoch=100, searchbound=400):
    estimated_rewards = np.zeros(epoch)

    # theta is a representation of policy
    theta = np.zeros((23, 4))
    grid = Grid()
    actions = grid.action
    # print(epoch)

    # for each episode:
    for x in range(epoch):
        # s ∼ d0
        s = grid.d_zero()
        count = 0
        hist_s = []
        hist_a = []
        hist_r = []
        grid.pi_params = estimation.softmax(theta, eps(x))
        # for each time step, until s is the terminal absorbing state do
        while s != [5, 5] and count < 1000:
            hist_s.append(s)
            a = grid.pi(s)
            hist_a.append(a)
            new_s, r = grid.P_and_R(s, a)
            hist_r.append(r)
            s = new_s
            count += 1

        # delta_j = 0
        decay = 1
        for i in range(len(hist_s)):
            g = 0
            gd = 1
            for j in range(i, len(hist_s)):
                g += gd * hist_r[j]
                gd *= grid.gamma
            theta[grid.get_index(hist_s[i]), actions.index(hist_a[i])] += lr * decay * g
            decay *= grid.gamma

        grid.pi_params = estimation.softmax(theta, eps(x))
        # grid.softmax()
        grid_epi = GridEpisode(grid, step_bound=searchbound)
        # print('episode: ', x, ', pi: ', grid.pi_params)
        estimated_rewards[x] = grid_epi.run_all_steps()
        if x == epoch-1:
            print('episode: ', x, ', reward: ', estimated_rewards[x])
        # decay *= decay_rate

    return estimated_rewards


# run qlearning in several trails and get plot data
def reinforce_grid_trail(lr, eps, epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = reinforce_grid(lr, eps, epoch=epoch)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def reinforce_mc(alpha, beta, l, baseparams, eps, epoch=100, base='fourier'):
    mc = MountainCar()
    estimated_rewards = np.zeros(epoch)
    actions = mc.actions
    theta = None
    order = 0

    if base == 'fourier':
        order = baseparams['order']
        s = mc.d_zero()
        theta = np.zeros((1, len(actions) * (order + 1) ** len(s)))
        w = np.zeros((1, (order + 1) ** len(s)))
        # theta = np.zeros((len(s), 3))

    for x in range(epoch):
        s = mc.d_zero()
        e = np.zeros(w.shape)

        hist_s = []
        hist_a = []
        hist_r = []
        hist_pi = []

        count = 0
        dj = np.zeros(theta.shape)

        # for each time step, until s is the terminal absorbing state do
        while s[0] < mc.right_bound and count < 1000:

            pi_temp = estimation.softmax(fa.qw(theta, s, actions, base, baseparams), eps(x))
            a = np.random.choice(actions, 1, p=pi_temp)[0]

            new_s, r = mc.P_and_R(s, a)

            hist_a.append(a)
            hist_s.append(s)
            hist_r.append(r)
            hist_pi.append(pi_temp)

            s = new_s
            count += 1

        for i in range(len(hist_a)):
            g = 0
            for j in range(i, len(hist_s)):
                g += hist_r[j]
            v, dv = fa.vw(w, hist_s[i], base, baseparams)
            dj += (g - v) * dsoftmax(hist_s[i], hist_a[i], order, actions, hist_pi[i])
            e = l * e + dv

            if i == len(hist_s) - 1:
                delta = hist_r[i] + 0 - \
                        fa.vw(w, hist_s[i], base, baseparams)[0]
            else:
                delta = hist_r[i] + fa.vw(w, hist_s[i + 1], base, baseparams)[0] - \
                        fa.vw(w, hist_s[i], base, baseparams)[0]

            w += alpha * delta * e
        theta += beta * dj

        epi = MountainCarEpisode(mc)
        # print(theta)
        estimated_rewards[x] = epi.run_with_w_softmax(theta, eps(x), base, baseparams)
        print('episode: ', x, ', reward: ', estimated_rewards[x])
    return estimated_rewards


def reinforce_mc_trail(alpha, beta, l,  baseparams, eps, base='fourier', epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = reinforce_mc(alpha, beta, l, baseparams, eps, epoch=epoch, base=base)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def dsoftmax(s, a, order, actions, pi):
    dtheta = np.zeros((1, len(actions) * (order + 1) ** len(s)))

    for idx in range(len(actions)):
        phi = fa.fourier_phi_mc(s, order).T
        if actions[idx] == a:
            # print('target')
            dtheta[:, idx * phi.shape[1]: (idx + 1) * phi.shape[1]] = (1 - pi[idx]) * phi
        else:
            dtheta[:, idx * phi.shape[1]: (idx + 1) * phi.shape[1]] = -pi[idx] * phi
    return dtheta


def draw_plot(data, error, epoch=100, filename='tests.png'):
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(np.array(range(epoch)), data)
    plt.fill_between(range(epoch), data - error, data + error, alpha=0.3)
    plt.savefig(filename, dpi=200)

    plt.show()


rewards, error = reinforce_grid_trail(0.047266, lambda x: 2, epoch=200)  # 0.097866
ah.save_cp_csvdata(rewards, error, 'rf_grid_alt.csv')
draw_plot(rewards, error, filename='rf_grid.png', epoch=200)

rewards, err = reinforce_mc_trail(1.675643e-3, 2.124e-3, 0.8, {'order': 7}, lambda x: 0.5, trail=100)
ah.save_cp_csvdata(rewards, err, 'rf_mc.csv')
draw_plot(rewards, err, filename='rf_mc.png')
