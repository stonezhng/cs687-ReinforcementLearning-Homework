import itertools
import numpy as np


def fourier_phi_mc(s, order):
    ns = s.copy()
    ns[0] = (s[0] + 1.2) / (0.5 + 1.2)
    ns[1] = (s[1] + 0.07) / (0.07 + 0.07)

    ns = np.array([ns]).T
    iter = itertools.product(range(order + 1), repeat=len(s))
    c = np.array([list(map(int, x)) for x in iter])  # ((n+1)^d, d) = (256, 4) if f_order = 3
    return np.cos(np.pi * c.dot(ns))  # ((n+1)^d, 1) = (256, 1) if f_order = 3


def qw_ele(w, s, a, actions, base, baseparams):
    aphi = None
    phi = None

    if base == 'fourier':
        order = baseparams['order']
        aphi = fourier_phi_mc(s, order)  # aphi.shape = ((n+1)^d, 1)
        # print(aphi)
        phi = np.zeros((len(actions) * aphi.shape[0], aphi.shape[1]))  # phi.shape = (num_of_actions * (n+1)^d, 1)
        i = actions.index(a)
        phi[i * aphi.shape[0]: (i + 1) * aphi.shape[0], :] = aphi

    return w.dot(phi)[0][0], phi.T


# give a list of results corresponding to s
def qw(w, s, actions, base, baseparams):
    q = np.zeros(len(actions))
    for i in range(len(actions)):
        q[i] = qw_ele(w, s, actions[i], actions, base, baseparams)[0]
    return q


def vw(w, s, base, baseparams):
    if base == 'fourier':
        order = baseparams['order']
        f = fourier_phi_mc(s, order)
        return w.dot(f)[0][0], f.T
