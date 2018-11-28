import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def add():
    b = np.array([[0], [1], [2], [3]])
    # print(b.shape)
    a = np.array([4, 5, 6])
    print(a)
    print(b)
    print(a.shape, b.shape)
    print(a+b)

    c = np.array([[3, 4], [6, 7], [-1, 2]])
    d = np.array([-1, -2])
    print(c+d)


def row_add():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    a = a + np.array([[1, 2]]).reshape(2, 1)
    print(np.array([[1, 2]]).shape)
    print(a)


def switch_zero():
    a = np.array([[-1, 2, 3], [4, -5, -6]])
    a = a.clip(min=0)
    a = (a > 0).astype(int)
    print(a)


def random_choice():
    a = np.array([[-1, 2, 3, 4], [-5, -6, 7, 8]])
    print(a[:, np.random.choice(a.shape[1], size=2)])


def indices_add():
    y = [2, 3]
    x = range(len(y))
    a = np.array([[-1, 2, 3, 4], [-5, -6, 7, 8]])
    print(np.sum(a[x, y]))


def indices_sub():
    W = np.array([[-1, 2, 1], [-3, 4, 2], [-5, 6, 1], [-7, 8, 3]])
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 2, 3], [1, 3, 4, 7], [2, 2, 3, 3]])
    print("Input Matrix: \n")
    print(W)
    print()
    print(X)
    y = [2, 2]
    x = range(len(y))
    W.T[y] += X[x]
    # print(list(x))
    print()
    print("When taking W.T[y] += X[x] operation, W became: ")
    print(W)
    print()
    W = np.array([[-1, 2, 1], [-3, 4, 2], [-5, 6, 1], [-7, 8, 3]])
    W[:, 2] += X[0]
    W[:, 2] += X[1]
    print('But the expected W should be: \n', W)


def random_action():
    action = ['AU', 'AL', 'AD', 'AR']
    prob = [0.50988135, 0.16579113, 0.14744689, 0.17688063]
    return np.random.choice(action, 1, p=prob)[0]


def indices_at_add():
    W = np.array([[-1, 2, 1], [-3, 4, 2], [-5, 6, 1], [-7, 8, 3]])
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 2, 3], [1, 3, 4, 7], [2, 2, 3, 3]])
    y = [2, 2, 1]
    x = range(len(y))
    np.add.at(W.T, y, X[x])
    print(W)
    W = np.array([[-1, 2, 1], [-3, 4, 2], [-5, 6, 1], [-7, 8, 3]])
    W[:, 2] += X[0]
    W[:, 2] += X[1]
    W[:, 1] += X[2]
    print('But the expected W should be: \n', W)


def multipilication():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2])
    print(X * y)


def sorttest():
    result_list = [(1, 3), (2, 4), (5, 2), (7, -1)]
    elite_list = sorted(result_list, key=lambda n: n[-1], reverse=True)[: 2]
    print(elite_list)


def k_fold_test():
    train_x = np.array([1, 2, 3, 4, 5])
    train_y = np.array([6, 7, 8, 9, 10])
    kf = KFold(n_splits=5)
    for train, test in kf.split(train_x):
        # print(train, test)
        print(train_x[train], train_x[test], train_y[train][0], train_y[test])

# pad = 1
# stride = 2
# px = np.pad(np.array([[1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6], [1, 2, 4, 5, 6]]), [(1, 1), (1, 1)], mode='constant', constant_values=0)
# k = 0
# j = 0
# ww = 2
# hh = 2
# w = 5
# h = 4
# print(px)
# for k in range(0, h + 2 * pad + 1 - hh, stride):
#     for j in range(0, w + 2 * pad + 1 - ww, stride):
#         sub_px = px[k: k+hh, j: j+ww]
#         print(sub_px)


# a = np.array([[2, 3, 4, 5], [1, 2, 4, 7]])
# print(np.unravel_index(a.argmax(), (2, 4)))
a = np.zeros(100) + 1010
a[1] = 8
print(np.std(a))
# print(k)
#
# k_fold_test()
# switch_zero()
# print('W'+str(1))
# countAR = 0
# countAD = 0
# countAL = 0
# countAU = 0
# for x in range(100):
#     result = random_action()
#     if result == 'AR':
#         countAR += 1
#     elif result == 'AD':
#         countAD += 1
#     elif result == 'AL':
#         countAL += 1
#     elif result == 'AU':
#         countAU += 1
# print(countAR)
# print(countAD)
# print(countAL)
# print(countAU)

# l = list(range(20))
# print(list(l[i:i+5] for i in range(0, len(l), 5)))
# i = 'user5@organisation2.com20180910'.find('@') + 1
# print('user5@organisation2.com20180910'[i: -8])
# print('20190101'[-8:])
# fig, ax = plt.subplots()
# ax.errorbar(np.array([1, 2, 3, 4]), np.array([1, 2, 2.5, 2.85]), yerr=np.array([0.3, 0.2, 0.1, 0.05]), fmt='o')
# plt.show()


