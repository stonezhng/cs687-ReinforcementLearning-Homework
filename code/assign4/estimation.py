import itertools
import numpy as np
import function_approximation as fa


def qw_ele(w, s, a, actions, base, baseparams):
    aphi = None
    phi = None

    if base == 'fourier':
        order = baseparams['order']
        aphi = fa.fourier_phi(s, order)  # aphi.shape = ((n+1)^d, 1)
        phi = np.zeros((len(actions) * aphi.shape[0], aphi.shape[1]))  # phi.shape = (num_of_actions * (n+1)^d, 1)
        i = actions.index(a)
        phi[i * aphi.shape[0]: (i + 1) * aphi.shape[0], :] = aphi

    elif base == 'tile':
        num_tilings, tiles_per_tiling = baseparams['num_tilings'], baseparams['tiles_per_tiling']
        # phi = fa.tile_phi(s, num_tilings, tiles_per_tiling, a, actions)
        phi = fa.a_tile_phi(s, num_tilings, tiles_per_tiling, a, actions)
        # phi = fa.tile_phi(s, lambda x: x+0.2**(x+3), num_tilings, tiles_per_tiling, a, actions)

    elif base == 'rbf':
        order = baseparams['order']
        aphi = fa.rbf_phi(s, order)  # aphi.shape = (n^d, 1)
        phi = np.zeros((len(actions) * aphi.shape[0], aphi.shape[1]))  # phi.shape = (num_of_actions * n^d, 1)
        i = actions.index(a)
        phi[i * aphi.shape[0]: (i + 1) * aphi.shape[0], :] = aphi
    # print(phi)
    return w.dot(phi)[0][0], phi.T


# give a list of results corresponding to s
def qw(w, s, actions, base, baseparams):
    q = np.zeros(len(actions))
    for i in range(len(actions)):
        q[i] = qw_ele(w, s, actions[i], actions, base, baseparams)[0]
    return q


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


def softmax(q, actions, sigma):
    # pass
    pi = np.zeros(q.shape)
    q_exp = np.exp(sigma * q)
    if len(q.shape) == 2:
        pi = q_exp / np.array([np.sum(q_exp, axis=1)]).T
    elif len(q.shape) == 1:
        pi = q_exp / np.sum(q_exp)
    return pi