import numpy as np


def epsilon_greedy(q, actions, eps):
    pi = np.zeros(q.shape) + eps / len(actions)
    if len(q.shape) == 2:
        a_star_indices = np.argmax(q, axis=1)
        for si in range(q.shape[0]):
            pi[si][a_star_indices[si]] += 1 - eps
    elif len(q.shape) == 1:
        a_star_indices = np.argmax(q)
        pi[a_star_indices] += 1 - eps
    return pi


def softmax(q, sigma):
    # pass
    # print(q)
    # pi = np.zeros(q.shape)
    # nq = q / np.sum(q)
    q_exp = np.exp(sigma * q)
    # print(q_exp)
    if len(q.shape) == 2:
        pi = q_exp / np.array([np.sum(q_exp, axis=1)]).T
    elif len(q.shape) == 1:
        pi = q_exp / np.sum(q_exp)
    return pi


def dsoftmax(q, dydtheta_list, a_idx, sigma):
    q_exp = np.exp(sigma * q)
    q_exp /= q_exp[a_idx]
    # print('dsoftmax: ', q_exp)
    exp_sum = np.sum(q_exp)
    # dout = (-q_exp / exp_sum ** 2) * q_exp[a_idx]
    dout = - sigma * (exp_sum - 1) / exp_sum ** 2
    dtheta = np.zeros(dydtheta_list[a_idx].shape)
    for dt in dydtheta_list:
        dtheta += np.array([dout]).dot(dt - dydtheta_list[a_idx])
    dtheta /= (q_exp[a_idx] / exp_sum)
    return dtheta
