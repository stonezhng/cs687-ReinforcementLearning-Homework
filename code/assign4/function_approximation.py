import itertools
import numpy as np


def fourier_phi(s, f_order):
    normalized_s = np.array([normalize(s)]).T  # (d, 1)
    iter = itertools.product(range(f_order + 1), repeat=len(s))
    c = np.array([list(map(int, x)) for x in iter])  # ((n+1)^d, d) = (256, 4) if f_order = 3
    return np.cos(np.pi * c.dot(normalized_s))  # ((n+1)^d, 1) = (256, 1) if f_order = 3


def tile_phi(s, num_tilings, tiles_per_tiling, a, actions):
    # pass
    dim = len(s)
    s_max = np.array([3, 10, np.pi / 2, np.pi])
    s_min = -s_max
    tile_size = (s_max - s_min) / (tiles_per_tiling - 1)
    num_tiles = tiles_per_tiling**dim * num_tilings

    tile_indices = np.zeros(num_tilings)

    # use matrix to store offsets
    matrix = np.zeros((num_tilings, dim))

    # do the offset
    for tiling in range(num_tilings):
        ns = s - s_min
        for i in range(dim):
            matrix[tiling, i] = int(ns[i] / tile_size[i] + tiling / num_tilings)

    for i in range(1, dim):
        matrix[:, i] *= tiles_per_tiling ** i

    for i in range(num_tilings):
        tile_indices[i] = (i * (tiles_per_tiling ** dim) + sum(matrix[i, :]))

    phi = np.zeros((num_tiles * len(actions), 1))
    for i in tile_indices:
        index = int(i + (num_tiles * actions.index(a)))
        if index < phi.shape[0]:
            phi[index, 0] = 1
    return phi


def a_tile_phi(s, num_tilings, tiles_per_tiling, a, actions):
    dim = len(s)
    s_max = np.array([3, 10, np.pi / 2, np.pi])
    s_min = -s_max
    tile_size = (s_max - s_min) / (tiles_per_tiling - 1)
    num_tiles = tiles_per_tiling ** dim * num_tilings
    tiling_offset = tile_size / num_tilings

    tile_indices = np.zeros(num_tilings)

    for tiling in range(num_tilings):
        ns = (s / tile_size)
        index = tiling * (tiles_per_tiling ** dim)
        for d in range(dim):
            index += int(ns[d]) * (tiles_per_tiling ** d)

        tile_indices[tiling] = index
        s += tiling_offset

    phi = np.zeros((num_tiles * len(actions), 1))

    for i in tile_indices:
        index = int(i + (num_tiles * actions.index(a)))
        if index < phi.shape[0]:
            phi[index, 0] = 1
    return phi


def rbf_phi(s, order):
    iter = itertools.product(range(order), repeat=len(s))
    normalized_s = np.array([normalize(s)]).T  # (d, 1)
    c = (np.array([list(map(int, x)) for x in iter]) + 1) / (order+1) # ((n)^d, d) = (81, 4) if f_order = 3
    newc = c - normalized_s[:, 0]
    b = np.sum(np.square(newc), axis=1)  # (n^d, )
    sigma = 2 / (order - 1)
    phi = np.exp(-b / (2*sigma))
    phi /= np.sqrt(2*np.pi*sigma)
    # print(phi)
    return np.array([phi]).T  # (n^d, 1)


def normalize(s):
    ns = s.copy()
    ns[0] = (s[0] - (-3)) / (3 - (-3))
    ns[1] = (s[1] - (-10)) / (10 - (-10))
    ns[2] = (s[2] - (-np.pi / 2)) / (np.pi / 2 - (-np.pi / 2))
    ns[3] = (s[3] - (-np.pi)) / (np.pi - (-np.pi))
    return ns
