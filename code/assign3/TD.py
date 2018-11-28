from grid_plus import Grid
from grid_plus import GridEpisode
from cartpole import CartPole
from cartpole import CartPoleEpisode
import numpy as np
import matplotlib.pyplot as plt
import itertools


# td is a policy evaluation method.
# given a policy pi or a\ policy parameter omega

# Create plots where the horizontal axis is the step size (using a logarithmic scale,
# and any reasonable range that includes 0.1, 0.01, and 0.001),
# and the vertical axis is the mean squared TD-error of the value function output by TD after 100 episodes.

# To estimate this mean squared TD-error, stop updating the weights after 100 episodes,
# and run an additional 100 episodes.
# During these extra 100 episodes, compute the TD error at each time step, square it to obtain the squared TD error,
# and report the average value of these TD errors.

# V is commonly approximated as a weighted sum of a set of basis functions
# weight is updated by a 100 loops method
# for gridworld, phi(s) works on s, which is a scalar or a two dim vector

def td_grid(lrs):
    tabular = np.zeros(23 * 4)

    grid = Grid()
    grid.pi_params = tabular.reshape(23, 4)
    grid.softmax()

    print('gridworld td')

    alpha_result = []
    for alpha in lrs:
        estimated_v = np.zeros(23)
        print('alpha = ', alpha)
        # update tabular in 100 loops
        for x in range(100):
            s = grid.d_zero()
            count = 0
            while s != [5, 5] and count < 500:
                a = grid.pi(s)
                new_s, r = grid.P_and_R(s, a)
                i = grid.get_index(s)
                new_i = grid.get_index(new_s)
                estimated_v[i] += alpha * (r + grid.gamma * estimated_v[new_i] - estimated_v[i])
                s = new_s
                count += 1

        # calculate td in another 100 loops
        td_list = []
        for x in range(100):
            s = grid.d_zero()
            count = 0
            while s != [5, 5] and count < 500:
                a = grid.pi(s)
                new_s, r = grid.P_and_R(s, a)
                i = grid.get_index(s)
                new_i = grid.get_index(new_s)
                td_list.append((r + grid.gamma * estimated_v[new_i] - estimated_v[i]) ** 2)
                s = new_s
                count += 1
            td_list.append(0)

        print('square td = ', np.mean(np.array(td_list)))
        alpha_result.append(np.mean(np.array(td_list)))

    print('##########################')
    return alpha_result


def td_cp(lrs, f_order):
    d = 4

    alpha_result = []
    cartpole = CartPole()

    print('cartpole ', f_order, ' td')

    # kth order Fourier Basis is defined as:
    for alpha in lrs:
        weight = np.zeros((1, (f_order + 1) ** d))
        # update weight in 100 loops
        print('alpha = ', alpha)
        for x in range(100):
            s = cartpole.d_zero()
            count = 0
            while np.abs(s[0]) < cartpole.edge and np.abs(s[1]) < cartpole.fail_angle and count < 1010:
                a = cartpole.pi(s)
                new_s, r = cartpole.P_and_R(s, a)
                weight += alpha * (r + vw(weight, new_s, f_order) - vw(weight, s, f_order)) * dvwdw(weight, s,
                                                                                                      f_order).T
                s = new_s
                count += 1
        # print(weight)

        # calculate td in another 100 loops
        td_list = []
        for x in range(100):
            s = cartpole.d_zero()
            count = 0
            while np.abs(s[0]) < cartpole.edge and np.abs(s[1]) < cartpole.fail_angle and count < 1010:
                a = cartpole.pi(s)
                new_s, r = cartpole.P_and_R(s, a)
                td_list.append((r + vw(weight, new_s, f_order) - vw(weight, s, f_order)) ** 2)
                s = new_s
                count += 1
            td_list.append(0)

        msv = np.mean(np.array(td_list))
        print('square td = ', msv)
        if np.isnan(msv):
            alpha_result.append(1e100)
        else:
            alpha_result.append(msv)

    print('##########################')
    return alpha_result


def vw(w, s, f_order):
    return w.dot(fourier_phi(s, f_order))[0][0]


def dvwdw(w, s, f_order):
    return fourier_phi(s, f_order)


def fourier_phi(s, f_order):
    normalized_s = np.array([normalize(s)]).T
    iter = itertools.product(range(f_order + 1), repeat=len(s))
    c = np.array([list(map(int, x)) for x in iter])  # ((n+1)^d, d) = (256, 4) if f_order = 3
    return np.cos(np.pi * c.dot(normalized_s))  # ((n+1)^d) = (256, ) if f_order = 3


def normalize(s):
    ns = s.copy()
    ns[0] = (s[0] - (-3)) / (3 - (-3))
    ns[1] = (s[1] - (-10)) / (10 - (-10))
    ns[2] = (s[2] - (-np.pi / 2)) / (np.pi / 2 - (-np.pi / 2))
    ns[3] = (s[3] - (-np.pi)) / (np.pi - (-np.pi))
    return ns


def draw_multi_bar(x, y_map, limbase='nolim', filename='result.png'):
    labels = list(y_map.keys())

    # set threshold
    if limbase != 'nolim' and limbase in labels:
        lim = (y_map[limbase])[x.index(0.1)] * 1.5
        plt.ylim([0, lim])
    elif type(limbase) is int or float:
        plt.ylim([0, limbase])

    plt.xlabel('step size')
    plt.ylabel('mean square td')

    plt.xticks([x.index(0.001), x.index(0.01), x.index(0.1)], [0.001, 0.01, 0.1])
    for l in labels:
        plt.plot(range(len(x)), y_map[l], linestyle='-', marker='|', label=l)

    plt.legend(loc='upper left')

    plt.savefig(filename, dpi=200)
    plt.show()


# this is for test only
def td_cp_single(f_order, alpha):
    d = 4

    cartpole = CartPole()

    print('cartpole ', f_order, ' td')

    weight = np.zeros((1, (f_order + 1) ** d))
    # update weight in 100 loops
    print('alpha = ', alpha)
    for x in range(100):
        s = cartpole.d_zero()
        count = 0
        while np.abs(s[0]) < cartpole.edge and np.abs(s[1]) < cartpole.fail_angle and count < 1010:
            a = cartpole.pi(s)
            new_s, r = cartpole.P_and_R(s, a)
            weight += alpha * (r + vw(weight, new_s, f_order) - vw(weight, s, f_order)) * dvwdw(weight, s,
                                                                                                f_order).T
            s = new_s
            print(weight)
            count += 1
    # calculate td in another 100 loops
    td_list = []
    for x in range(100):
        s = cartpole.d_zero()
        count = 0
        while np.abs(s[0]) < cartpole.edge and np.abs(s[1]) < cartpole.fail_angle and count < 1010:
            a = cartpole.pi(s)
            new_s, r = cartpole.P_and_R(s, a)
            td_list.append((r + vw(weight, new_s, f_order) - vw(weight, s, f_order)) ** 2)
            s = new_s
            count += 1
        td_list.append(0)
    print('square td = ', np.mean(np.array(td_list)))


# td_cp_single(3, 1e-4)


lrs = []
for i in np.arange(-5, 0.25, 0.25):
    lrs.append(10 ** i)

result = {'gridworld': td_grid(lrs), 'cartpole3rd': td_cp(lrs, 3), 'cartpole5th': td_cp(lrs, 5)}

# result = {'grid': td_grid(lrs), 'cartpole3': td_cp(lrs, 3)}
# result = {'cartpole5': td_cp(lrs, 5), 'cartpole3': td_cp(lrs, 3)}
# result = {'cartpole3': td_cp(lrs, 3)}
# result = {'grid': td_grid(lrs)}

draw_multi_bar(lrs, result, limbase=600, filename='result_F1.png')
