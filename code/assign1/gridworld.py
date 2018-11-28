import numpy as np
import random


class Grid:
    def __init__(self):
        self.state = list(range(1, 24))
        self.action = ['AU', 'AL', 'AD', 'AR']
        # self.dynamic = ['stay', 'vl', 'vr', 'normal']
        # self.dynamic_prob = [0.1, 0.05, 0.05, 0.8]
        # self.action_map = {('AU', 'stay'): '', ('AU', 'vl'): 'AL', ('AU', 'vr'): 'AR', ('AU', 'normal'): 'AU',
        #                   ('AD', 'stay'): '', ('AD', 'vl'): 'AR', ('AD', 'vr'): 'AL', ('AD', 'normal'): 'AL',
        #                   ('AL', 'stay'): '', ('AL', 'vl'): 'AD', ('AL', 'vr'): 'AU', ('AL', 'normal'): 'AL',
        #                   ('AR', 'stay'): '', ('AR', 'vl'): 'AU', ('AR', 'vr'): 'AD', ('AR', 'normal'): 'AR'}
        self.pi_table = None
        # print(self.pi_table[0])
        self.gama = 0.9

        self.left_wall = [1, 6, 11, 13, 17, 15, 19]
        self.right_wall = [5, 10, 12, 14, 16, 18, 23]
        self.up_wall = [1, 2, 3, 4, 5, 21]
        self.down_wall = [8, 19, 20, 21, 22, 23]

        # self.K = 20

    def d_zero(self):
        return 1

    def pi(self, s, t=None):
        # print(s)
        return np.random.choice(self.action, 1, p=self.pi_table[s-1])[0]

    def dynamic(self, a):
        rand = random.uniform(0, 1)
        if rand < 0.8:
            return a
        elif rand < 0.9:
            return ''
        elif rand < 0.95:
            return self.action[self.action.index(a) - 3]
        else:
            return self.action[self.action.index(a) - 1]

    def P_and_R(self, s, a):
        new_action = self.dynamic(a)
        s_next = s
        if new_action == 'AU':
            if s in self.up_wall:
                s_next = s
            elif s in [13, 14, 15, 16, 17, 18, 19, 20]:
                s_next = self.state[self.state.index(s) - 4]
            else:
                s_next = self.state[self.state.index(s) - 5]
        elif new_action == 'AL':
            if s in self.left_wall:
                s_next = s
            else:
                s_next = self.state[self.state.index(s) - 1]
        elif new_action == 'AD':
            if s in self.down_wall:
                s_next = s
            elif s in [9, 10, 11, 12, 13, 14, 15, 16]:
                s_next = self.state[self.state.index(s) + 4]
            else:
                s_next = self.state[self.state.index(s) + 5]
        elif new_action == 'AR':
            if s in self.right_wall:
                s_next = s
            else:
                s_next = self.state[self.state.index(s) + 1]
        elif new_action == '':
            s_next = s

        if s_next == 21:
            reward = -10
        elif s_next == 23:
            reward = 10
        else:
            reward = 0
        # print(s_next, reward)
        return s_next, reward, new_action


class GridEpisode:
    def __init__(self, grid, step_bound=400):
        self.grid = grid
        self.history = []
        self.rewards = []
        self.active = 1
        self.step_count = 0
        self.step_bound = step_bound

    def run_first_step(self):
        s = self.grid.d_zero()
        self.history.append(s)

    def run_next_step(self):
        if self.active == 1:
            s = self.history[-1]
            # a = self.grid.pi(s)
            a = self.grid.pi(s)
            # a = self.grid.pi_optimized(s)
            next_step = self.grid.P_and_R(s, a)
            # print(next_step)
            s_next = next_step[0]
            reward = next_step[1]
            action = next_step[2]
            self.history.append(action)
            self.history.append(s_next)
            self.rewards.append(reward)
            if s_next == 23:
                self.active = 0

    def run_all_steps(self):
        self.run_first_step()
        while self.active != 0:
            self.run_next_step()
            self.step_count += 1
            if self.step_count > self.step_bound:
                self.active = 0

    def get_discount_reward(self):
        sum = 0
        discount = 1.0
        for r in self.rewards:
            sum += r * discount
            discount *= self.grid.gama
        # print(self.history)
        return sum


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
