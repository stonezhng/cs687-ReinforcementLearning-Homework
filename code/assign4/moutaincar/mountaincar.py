import numpy as np
import moutaincar.mc_estimation as me


class MountainCar:
    def __init__(self):
        # action size = 3, state dimension = 2
        self.actions = [0, 1, 2]
        self.right_bound = 0.5
        self.left_bound = -1.2
        self.pi_params = np.array([[1, 1], [1, 1], [1, 1]]).T

    def d_zero(self):
        return [-0.5, 0]

    def dynamics(self, s, a):
        new_v = s[1] + 0.001 * a + 0.0025 * np.cos(3 * s[0])
        new_x = s[0] + new_v

        if new_x < self.left_bound:
            new_x = self.left_bound
            new_v = 0
        elif new_x > self.right_bound:
            new_x = self.right_bound
            new_v = 0

        return [new_x, new_v]

    def pi(self, s):
        # softmax process
        p = np.array(s).dot(self.pi_params)
        p = np.exp(p)
        exp_sum = np.sum(p)
        p /= exp_sum

    def P_and_R(self, s, a):
        new_s = self.dynamics(s, a)
        if new_s[0] == self.right_bound:
            return new_s, 0
        else:
            return new_s, -1


class MountainCarEpisode:
    def __init__(self, mountaincar):
        self.mountaincar = mountaincar
        self.active = 1

    def run_all_steps(self):
        s = self.mountaincar.d_zero()
        reward = 0
        while self.active == 1:
            a = self.mountaincar.pi(s)
            s, r = self.mountaincar.P_and_R(s, a)
            # print(s)
            reward += r

            if s[0] == self.mountaincar.right_bound:
                self.active = 0
        return reward

    def run_with_w(self, w, eps, base, baseparams):
        reward = 0
        s = self.mountaincar.d_zero()

        while self.active == 1:
            q = me.qw(w, s, self.mountaincar.actions, base, baseparams)
            pi = me.epsilon_greedy(q, self.mountaincar.actions, eps)
            a = np.random.choice(self.mountaincar.actions, 1, p=pi)[0]
            s, r = self.mountaincar.P_and_R(s, a)
            reward += r
            if s[0] == self.mountaincar.right_bound:
                self.active = 0

        return reward
