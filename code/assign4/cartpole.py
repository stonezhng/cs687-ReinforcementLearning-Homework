import numpy as np
import estimation as pe


class CartPole:
    def __init__(self):
        self.mc = 1.0
        self.mp = 0.1
        self.g = 9.8
        self.fail_angle = np.pi / 2
        self.l = 0.5
        self.delta_t = 0.02
        self.F = 10.0
        self.edge = 3

        self.actions = ['l', 'r']
        # s = (x, v, θ, w ̇)
        self.pi_params = np.array([[1, 1, 1, 1], [1, 1, 1, 1]]).T

    def d_zero(self):
        return [0, 0, 0, 0]

    def dynamics(self, s, a):
        dthetadtdt = None
        dxdtdt = None
        if a == 'r':
            dthetadtdt = self.g * np.sin(s[2]) + np.cos(s[2]) * (
                        -self.F - self.mp * self.l * (s[3]**2) * np.sin(s[2])) / (self.mp + self.mc)
            dxdtdt = self.F + self.mp * self.l * ((s[3]**2) * np.sin(s[2]) - dthetadtdt * np.cos(s[2]))
            dthetadtdt /= self.l * (4.0 / 3 - (self.mp * (np.cos(s[2]) ** 2)) / (self.mp + self.mc))
            dxdtdt /= self.mc + self.mp
        elif a == 'l':
            dthetadtdt = self.g * np.sin(s[2]) + np.cos(s[2]) * (
                        +self.F - self.mp * self.l * (s[3] ** 2) * np.sin(s[2])) / (self.mp + self.mc)
            dxdtdt = -self.F + self.mp * self.l * ((s[3] ** 2) * np.sin(s[2]) - dthetadtdt * np.cos(s[2]))
            dthetadtdt /= self.l * (4.0 / 3 - (self.mp * (np.cos(s[2]) ** 2)) / (self.mp + self.mc))
            dxdtdt /= self.mc + self.mp

        return dthetadtdt, dxdtdt

    def pi(self, s):
        # softmax process
        p = np.array(s).dot(self.pi_params)
        p = np.exp(p)
        exp_sum = np.sum(p)
        p /= exp_sum

        return np.random.choice(self.actions, 1, p=p)[0]

    def P_and_R(self, s, a):
        dthetadtdt, dxdtdt = self.dynamics(s, a)
        new_s = s.copy()
        new_s[0] += dxdtdt * self.delta_t * self.delta_t
        new_s[1] += dxdtdt * self.delta_t
        new_s[2] += dthetadtdt * self.delta_t * self.delta_t
        new_s[3] += dthetadtdt * self.delta_t
        return new_s, 1


class CartPoleEpisode:
    def __init__(self, cartpole):
        self.cartpole = cartpole
        self.maxturn = 1010
        self.active = 1
        self.step_count = 0

    def run_all_steps(self):
        s = self.cartpole.d_zero()
        reward = 0
        while self.active == 1:
            a = self.cartpole.pi(s)
            s, r = self.cartpole.P_and_R(s, a)
            # print(s)
            reward += r

            self.step_count += 1
            if self.step_count > self.maxturn:
                self.active = 0
            if np.abs(s[0]) > self.cartpole.edge:
                self.active = 0
            if np.abs(s[1]) > self.cartpole.fail_angle:
                self.active = 0
        return reward-1

    def run_with_w(self, w, eps, base, baseparams):
        reward = 0
        s = self.cartpole.d_zero()

        while self.active == 1:
            q = pe.qw(w, s, self.cartpole.actions, base, baseparams)
            pi = pe.epsilon_greedy(q, self.cartpole.actions, eps)
            a = np.random.choice(self.cartpole.actions, 1, p=pi)[0]
            s, r = self.cartpole.P_and_R(s, a)
            reward += 1
            self.step_count += 1
            if self.step_count > self.maxturn:
                self.active = 0
            if np.abs(s[0]) > self.cartpole.edge:
                self.active = 0
            if np.abs(s[1]) > self.cartpole.fail_angle:
                self.active = 0

        return reward - 1

    def run_with_w_softmax(self, w, eps, base, baseparams):
        reward = 0

        s = self.cartpole.d_zero()

        while self.active == 1:
            q = pe.qw(w, s, self.cartpole.actions, base, baseparams)
            pi = pe.softmax(q, self.cartpole.actions, eps)
            a = np.random.choice(self.cartpole.actions, 1, p=pi)[0]
            s, r = self.cartpole.P_and_R(s, a)
            reward += 1
            self.step_count += 1
            if self.step_count > self.maxturn:
                self.active = 0
            if np.abs(s[0]) > self.cartpole.edge:
                self.active = 0
            if np.abs(s[1]) > self.cartpole.fail_angle:
                self.active = 0

        return reward - 1

