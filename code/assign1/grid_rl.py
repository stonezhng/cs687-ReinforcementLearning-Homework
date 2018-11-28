import random
import numpy as np


########################################
#        begin of Grid class           #
########################################
class Grid:
    def __init__(self):
        self.state = []

        self.state = list(range(1, 24))

        self.action = ['AU', 'AL', 'AD', 'AR']
        self.gama = 0.9

        self.pi_matrix = []

        self.left_wall = [1, 6, 11, 13, 17, 15, 19]
        self.right_wall = [5, 10, 12, 14, 16, 18, 23]
        self.up_wall = [1, 2, 3, 4, 5, 21]
        self.down_wall = [8, 19, 20, 21, 22, 23]

        self.pi_matrix.append(['AR'])  # 1
        self.pi_matrix.append(['AR'])  # 2
        self.pi_matrix.append(['AR'])  # 3
        self.pi_matrix.append(['AD'])  # 4
        self.pi_matrix.append(['AD'])  # 5
        self.pi_matrix.append(['AR'])  # 6
        self.pi_matrix.append(['AR'])  # 7A
        self.pi_matrix.append(['AR'])  # 8
        self.pi_matrix.append(['AD'])  # 9
        self.pi_matrix.append(['AD'])  # 10
        self.pi_matrix.append(['AU'])  # 11
        self.pi_matrix.append(['AU'])  # 12
        self.pi_matrix.append(['AD'])  # 13
        self.pi_matrix.append(['AD'])  # 14
        self.pi_matrix.append(['AU'])  # 15
        self.pi_matrix.append(['AU'])  # 16
        self.pi_matrix.append(['AR'])  # 17
        self.pi_matrix.append(['AD'])  # 18
        self.pi_matrix.append(['AR'])  # 19
        self.pi_matrix.append(['AU'])  # 20
        self.pi_matrix.append(['AR'])  # 21
        self.pi_matrix.append(['AR'])  # 22
        self.pi_matrix.append([''])  # 23

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

    def pi_uniform(self, s):
        return self.action[random.randrange(4)]

    def pi_optimized(self, s):
        return self.pi_matrix[s-1][0]

    def R(self, s, a):
        end_state = self.P(s, a)
        if end_state == 21:
            return -10
        elif end_state == 23:
            return 10
        else:
            return 0

    def dzero(self):
        return 1


class Episode:
    def __init__(self, grid):
        self.grid = grid
        self.history = []
        self.rewards = []
        self.active = 1

    def run_first_step(self):
        s = self.grid.dzero()
        self.history.append(s)

    def run_next_step_randomly(self):
        if self.active == 1:
            s = self.history[-1]
            a = self.grid.pi_uniform(s)
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

    def run_next_step_with_optimization(self):
        if self.active == 1:
            s = self.history[-1]
            # a = self.grid.pi_uniform(s)
            a = self.grid.pi_optimized(s)
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

    def run_all_steps_randomly(self):
        self.run_first_step()
        while self.active != 0:
            self.run_next_step_randomly()

    def run_all_steps_with_optimization(self):
        self.run_first_step()
        while self.active != 0:
            self.run_next_step_with_optimization()

    def get_discount_reward(self):
        sum = 0
        discount = 1.0
        for r in self.rewards:
            sum += r * discount
            discount *= self.grid.gama
        return sum

########################################
#        end of Grid class             #
########################################

##########################################################

########################################
#            question 1                #
########################################
grid = Grid()
# print(grid.state)
rewards = np.zeros(10000)
for x in range(10000):
    episode = Episode(grid)
    episode.run_all_steps_randomly()
    rewards[x] = episode.get_discount_reward()
print('################ question 1 ################')
print('Mean of rewards: ', rewards.mean())
print('Standard deviation of rewards: ', rewards.std())
print('Maximum of rewards: ', rewards.max())
print('Minimum of rewards: ', rewards.min())
print()


########################################
#            question 2                #
########################################
# For example, we replace the action in state 0 with 'AD', and we compare the mean value with the original one
grid = Grid()
grid.pi_matrix[10] = ['AR']
# grid.pi_matrix[6] = ['AU']
rewards = np.zeros(10000)
for x in range(10000):
    episode = Episode(grid)
    episode.run_all_steps_with_optimization()
    rewards[x] = episode.get_discount_reward()
print('################ question 2 ################')
print('Mean of rewards: ', rewards.mean())
print('Standard deviation of rewards: ', rewards.std())
print('Maximum of rewards: ', rewards.max())
print('Minimum of rewards: ', rewards.min())
print()


########################################
#            question 3                #
########################################
grid = Grid()
rewards = np.zeros(10000)
for x in range(10000):
    episode = Episode(grid)
    episode.run_all_steps_with_optimization()
    rewards[x] = episode.get_discount_reward()
print('################ question 3 ################')
print('Mean of rewards: ', rewards.mean())
print('Standard deviation of rewards: ', rewards.std())
print('Maximum of rewards: ', rewards.max())
print('Minimum of rewards: ', rewards.min())
print()


########################################
#            question 4                #
########################################
grid = Grid()
countl = 0
for x in range(100000):
    episode = Episode(grid)
    episode.history.append(18)
    while episode.active != 0:
        episode.run_next_step_randomly()
        if len(episode.history) > 22 and episode.history[22] == 21:
            countl += 1
            break

print('################ question 4 ################')
print('empirical estimation of probability: ', countl * 1.0 / 100000)

