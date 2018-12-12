import numpy as np
import function_approximation as fa
import estimation
import scipy.stats


class MountainCar:
    def __init__(self):
        # action size = 3, state dimension = 2
        # self.actions = [0, 1, 2]
        self.actions = [-1, 0, 1]
        self.right_bound = 0.5
        self.left_bound = -1.2
        # self.pi_params = np.array([[1, 1], [1, 1], [1, 1]]).T
        self.gamma = 1

    def d_zero(self):
        return [-0.5, 0]

    def dynamics(self, s, a):
        new_v = s[1] + 0.001 * a - 0.0025 * np.cos(3 * s[0])
        new_x = s[0] + new_v

        if new_v < -0.07:
            new_v = -0.07
        elif new_v > 0.07:
            new_v = 0.07

        if new_x < self.left_bound:
            new_x = self.left_bound
            new_v = 0
        elif new_x > self.right_bound:
            new_x = self.right_bound
            new_v = 0

        return [new_x, new_v]

    # def pi(self, s):
    #     # softmax process
    #     p = np.array(s).dot(self.pi_params)
    #     p = np.exp(p)
    #     exp_sum = np.sum(p)
    #     p /= exp_sum
    #     return np.random.choice(self.actions, 1, p=p)[0]

    # def pi_theta(self, s, theta):
    #     p = np.array(s).dot(theta)
    #     p = np.exp(p)
    #     exp_sum = np.sum(p)
    #     p /= exp_sum
    #     # print(p)
    #     a = np.random.choice(self.actions, 1, p=p)[0]
    #
    #     dout = (-p / exp_sum ** 2) * p[self.actions.index(a)]
    #     dout[self.actions.index(a)] = (exp_sum - p[self.actions.index(a)]) * p[self.actions.index(a)] / exp_sum ** 2
    #     # p: (1, 3), s: (1, 2)
    #     dtheta = np.array([s]).T.dot(np.array([dout]))
    #     dtheta /= (p[self.actions.index(a)] / exp_sum)
    #     return a, dtheta

    def gaussian_pi(self, s, theta, order, sigma):
        # sigma = 2 / (order - 1)
        c = theta.T.dot(fa.fourier_phi_mc(s, order))
        # print(c)
        prob = np.zeros(3)
        # for a in self.actions:
        #     prob[self.actions.index(a)] =
        # x = np.arange(0, 3)
        # xU, xL = x + 0.5, x - 0.5
        # prob = p.cdf(xU) - p.cdf(xL)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        return prob

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

    # def run_with_theta(self, theta):
    #     s = self.mountaincar.d_zero()
    #     reward = 0
    #     count = 0
    #     self.mountaincar.pi_param = theta
    #     while self.active == 1:
    #         # prob = self.mountaincar.gaussian_pi(s, theta, order, sigma
    #         # prob = self.mountaincar.pi(s)
    #         # print(prob)
    #         # p = self.mountaincar.pi_theta(s, theta)
    #         # a = np.random.choice(self.mountaincar.actions, 1, p=p)[0]
    #         a = self.mountaincar.pi_theta(s, theta)[0]
    #         s, r = self.mountaincar.P_and_R(s, a)
    #         # print(s)
    #         reward += r
    #         count += 1
    #
    #         if s[0] == self.mountaincar.right_bound:
    #             self.active = 0
    #         if count == 1000:
    #             self.active = 0
    #
    #     return reward

    def run_with_w(self, w, eps, base, baseparams):
        reward = 0
        s = self.mountaincar.d_zero()

        count = 0

        while self.active == 1:
            q = fa.qw(w, s, self.mountaincar.actions, base, baseparams)
            # print(q)
            # pi = estimation.softmax(q, eps)
            pi = estimation.epsilon_greedy(q, self.mountaincar.actions, eps)
            # print(pi)
            a = np.random.choice(self.mountaincar.actions, 1, p=pi)[0]
            # print(a)
            # print(s[0])
            # print(pi, s, a)
            s, r = self.mountaincar.P_and_R(s, a)
            count += 1
            if s[0] == self.mountaincar.right_bound:
                self.active = 0
            else:
                reward += r
            if count >= 1e3:
                self.active = 0

        return reward

    def run_with_w_softmax(self, w, eps, base, baseparams):
        reward = 0
        s = self.mountaincar.d_zero()

        count = 0

        while self.active == 1:
            q = fa.qw(w, s, self.mountaincar.actions, base, baseparams)
            # print(q)
            pi = estimation.softmax(q, eps)
            # pi = estimation.epsilon_greedy(q, self.mountaincar.actions, eps)
            # print('test')
            # print(pi)
            a = np.random.choice(self.mountaincar.actions, 1, p=pi)[0]
            # print(pi, s, a)
            # print(a)
            # print(s[0])
            s, r = self.mountaincar.P_and_R(s, a)
            count += 1
            if s[0] == self.mountaincar.right_bound:
                self.active = 0
            else:
                reward += r
            if count >= 1e3:
                self.active = 0

        return reward


# mc = MountainCar()
# epi = MountainCarEpisode(mc)
# epi.run_all_steps()
