import numpy as np
import random


class Grid:
    def __init__(self):
        self.state = []
        for x in range(5):
            for y in range(5):
                self.state.append([x+1, y+1])
        self.action = ['AU', 'AL', 'AD', 'AR']
        self.pi_params = None
        self.gamma = 0.9
        self.obstacles = [[3, 3], [4, 3]]
        self.waterpool = [[5, 3]]

    def d_zero(self):
        return [1, 1]

    # def pi(self, s, t=None):
    #     # print(s)
    #     p = (np.array(s)**3).dot(self.pi_params)
    #     p = np.exp(p)
    #     exp_sum = np.sum(p)
    #     p /= exp_sum
    #     # print(s, p)
    #     return np.random.choice(self.action, 1, p=p)[0]

    def softmax(self):
        self.pi_params = np.exp(self.pi_params)
        exp = np.sum(self.pi_params, axis=1)
        self.pi_params /= np.array([exp]).T

    def pi(self, s):
        i = (s[0]-1) * 5 + s[1] - 1
        if 12 < i < 18:
            i -= 1
        elif i > 18:
            i -= 2
        return np.random.choice(self.action, 1, p=self.pi_params[i])[0]

    def dynamic(self, a):
        rand = random.uniform(0, 1)
        new_a = ''
        if rand < 0.8:
            new_a = a
        elif rand < 0.9:
            new_a = ''
        elif rand < 0.95:
            new_a = self.action[self.action.index(a) - 3]
        else:
            new_a = self.action[self.action.index(a) - 1]
        return new_a

    def P_and_R(self, s, a):
        new_action = self.dynamic(a)
        # print(new_action)
        s_next = s.copy()
        if new_action == 'AU':
            s_next = [s[0]-1, s[1]]
            if s_next[0] < 1 or s_next in self.obstacles:
                s_next = s
        elif new_action == 'AR':
            s_next = [s[0], s[1]+1]
            if s_next[1] > 5 or s_next in self.obstacles:
                s_next = s
        elif new_action == 'AD':
            s_next = [s[0]+1, s[1]]
            if s_next[0] > 5 or s_next in self.obstacles:
                s_next = s
        elif new_action == 'AL':
            s_next = [s[0], s[1]-1]
            if s_next[1] < 1 or s_next in self.obstacles:
                s_next = s
        elif new_action == '':
            s_next = s

        if s_next == [5, 5]:
            reward = 10
        elif s_next in self.waterpool:
            reward = -10
        else:
            reward = 0
        # print(s_next, reward)
        return s_next, reward

    def get_index(self, s):
        i = (s[0] - 1) * 5 + s[1] - 1
        if 12 < i < 18:
            i -= 1
        elif i > 18:
            i -= 2
        return i


class GridEpisode:
    def __init__(self, grid, step_bound=1000):
        self.grid = grid
        self.active = 1
        self.step_count = 0
        self.step_bound = step_bound

    def run_all_steps(self):
        reward = 0
        discount = 1
        s = self.grid.d_zero()
        while self.active != 0:
            a = self.grid.pi(s)
            new_s, r = self.grid.P_and_R(s, a)
            reward += discount * r
            discount *= self.grid.gamma
            self.step_count += 1
            if self.step_count > self.step_bound:
                self.active = 0
            if new_s == [5, 5]:
                self.active = 0
            s = new_s
        return reward

    def run_with_s0(self, s0):
        reward = 0
        discount = 1
        s = s0
        while self.active != 0:
            a = self.grid.pi(s)
            new_s, r = self.grid.P_and_R(s, a)
            reward += discount * r
            discount *= self.grid.gamma
            self.step_count += 1
            if self.step_count > self.step_bound:
                self.active = 0
            if new_s == [5, 5]:
                self.active = 0
            s = new_s
        return reward

    def run_with_q(self, q):
        reward = 0
        discount = 1
        s = self.grid.d_zero()
        while self.active != 0:
            a = self.grid.action[np.argmax(q[self.grid.get_index(s)])]
            print(q[self.grid.get_index(s)])
            print(a)
            new_s, r = self.grid.P_and_R(s, a)
            reward += discount * r
            discount *= self.grid.gamma
            self.step_count += 1
            if self.step_count > self.step_bound:
                self.active = 0
            if new_s == [5, 5]:
                self.active = 0
            s = new_s
        return reward


# print(grid.P_and_R([5, 2], 'AR'))


# pi_table = [[0, 0, 0, 1],
#                          [0, 0, 0, 1],
#                          [0, 0, 0, 1],
#                          [0, 0, 1, 0],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1],
#                          [0, 0, 0, 1],
#                          [0, 0, 0, 1],
#                          [0, 0, 1, 0],
#                          [0, 0, 1, 0],
#                          [1, 0, 0, 0],
#                          [1, 0, 0, 0],
#                          [0, 0, 1, 0],
#                          [0, 0, 1, 0],
#                          [1, 0, 0, 0],
#                          [1, 0, 0, 0],
#                          [0, 0, 0, 1],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1],
#                          [1, 0, 0, 0],
#                          [0, 0, 0, 1],
#                          [0, 0, 0, 1],
#             [0, 0, 0, 1]]
# grid = Grid()
# grid.pi_table = pi_table
# # print(grid.state)
# rewards = np.zeros(1000)
# for x in range(1000):
#     episode = Episode(grid)
#     episode.run_all_steps()
#     rewards[x] = episode.get_discount_reward()
# print('################ question 1 ################')
# print('Mean of rewards: ', rewards.mean())
# print('Standard deviation of rewards: ', rewards.std())
# print('Maximum of rewards: ', rewards.max())
# print('Minimum of rewards: ', rewards.min())
# print()

# tabular = np.zeros(23 * 4)
# grid = Grid()
# grid.pi_params = tabular.reshape(23, 4)
# grid.softmax()
# epi = GridEpisode(grid)
# print(epi.run_with_s0([0, 0]))

# pi = [[2.5000e-04, 2.5000e-04, 2.5000e-04, 9.9925e-01],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 9.9925e-01, 2.5000e-04, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 2.5000e-04, 9.9925e-01],
#  [2.5000e-04, 2.5000e-04, 2.5000e-04, 9.9925e-01],
#  [2.5000e-04, 2.5000e-04, 2.5000e-04, 9.9925e-01],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [9.9925e-01, 2.5000e-04, 2.5000e-04, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 9.9925e-01, 2.5000e-04, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [9.9925e-01, 2.5000e-04, 2.5000e-04, 2.5000e-04],
#  [2.5000e-04, 9.9925e-01, 2.5000e-04, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 9.9925e-01, 2.5000e-04],
#  [2.5000e-04, 9.9925e-01, 2.5000e-04, 2.5000e-04],
#  [2.5000e-04, 9.9925e-01, 2.5000e-04, 2.5000e-04],
#  [2.5000e-04, 2.5000e-04, 2.5000e-04, 9.9925e-01]]
# grid = Grid()
# grid.pi_params = pi
# epi = GridEpisode(grid)
# print(epi.run_all_steps())
