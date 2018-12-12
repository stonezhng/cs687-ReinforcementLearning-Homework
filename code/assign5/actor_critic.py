from grid_plus import Grid
from grid_plus import GridEpisode
from mountaincar import MountainCar
from mountaincar import MountainCarEpisode
import function_approximation as fa
import assignment_helper as ah
import estimation

import numpy as np
import matplotlib.pyplot as plt


# qlearning for one trial
# use decaying eps
def actor_critic_grid(lr, eps, epoch=100, searchbound=400):
    estimated_rewards = np.zeros(epoch)

    # Initialize tabular-v arbitrarily
    v = np.zeros(23)
    # theta is a representation of policy
    theta = np.zeros((23, 4))
    grid = Grid()
    actions = grid.action

    # for each episode:
    for x in range(epoch):
        # s ∼ d0
        s = grid.d_zero()
        count = 0
        # for each time step, until s is the terminal absorbing state do
        while s != [5, 5] and count < 1000:
            # a ∼ π(s, ·);
            grid.pi_params = estimation.softmax(theta, eps(x))
            a = grid.pi(s)
            # Take action a and observe r and s′;
            new_s, r = grid.P_and_R(s, a)

            # Critic update using TD(λ)
            # e ← γλe + ∂qw(s,a)/∂qw;
            delta = r + grid.gamma * v[grid.get_index(new_s)] - v[grid.get_index(s)]
            # w←w+αδev;
            v[grid.get_index(s)] += lr * delta

            theta[grid.get_index(s), actions.index(a)] += lr * delta
            # print(theta)

            s = new_s
            count += 1
        # using q function to estimate the reward and add it to estimated_reward
        # print('episode: ', x, ', q function: ', q)
        grid.pi_params = estimation.softmax(theta, eps(x))
        # grid.softmax()
        grid_epi = GridEpisode(grid, step_bound=searchbound)
        # print('episode: ', x, ', pi: ', grid.pi_params)
        estimated_rewards[x] = grid_epi.run_all_steps()
        if x == 99:
            print('episode: ', x, ', reward: ', estimated_rewards[x])
        # decay *= decay_rate

    return estimated_rewards


# run qlearning in several trails and get plot data
def actor_critic_grid_trail(lr, eps, epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = actor_critic_grid(lr, eps, epoch=epoch)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def actor_critic_mc(lr, l, baseparams, eps, epoch=100, base='fourier'):
    mc = MountainCar()
    estimated_rewards = np.zeros(epoch)
    actions = mc.actions
    w = None
    theta = None
    order = 0

    if base == 'fourier':
        order = baseparams['order']
        s = mc.d_zero()
        w = np.zeros((1, (order + 1) ** len(s)))
        theta = np.zeros((1, len(actions) * (order + 1) ** len(s)))
        # theta = np.zeros((len(s), 3))

    for x in range(epoch):
        s = mc.d_zero()
        # ev ← 0
        e = np.zeros(w.shape)
        # et ← 0
        # et = np.zeros(theta.shape)

        count = 0

        # for each time step, until s is the terminal absorbing state do
        while s[0] < mc.right_bound and count < 1000:

            pi_temp = estimation.softmax(fa.qw(theta, s, actions, base, baseparams), eps(x))
            a = np.random.choice(actions, 1, p=pi_temp)[0]
            # print(a)

            # print(pi_temp)

            # dydtheta_list = []
            # for na in actions:
            #     dydtheta_list.append(fa.qw_ele(theta, s, na, actions, base, baseparams)[1])
            #
            # dtheta = estimation.dsoftmax(fa.qw(theta, s, actions, base, baseparams), dydtheta_list, actions.index(
            # a), eps(x))

            dtheta = np.zeros((1, len(actions) * (order + 1) ** len(s)))

            for idx in range(len(actions)):
                phi = fa.fourier_phi_mc(s, order).T
                if actions[idx] == a:
                    # print('target')
                    dtheta[:, idx * phi.shape[1]: (idx+1) * phi.shape[1]] = (1-pi_temp[idx]) * phi
                else:
                    dtheta[:, idx * phi.shape[1]: (idx+1) * phi.shape[1]] = -pi_temp[idx] * phi

            # Take action a and observe r and s′;

            new_s, r = mc.P_and_R(s, a)

            # Critic update using TD(λ)
            # ev←γλev+∂vw(s);
            v, dv = fa.vw(w, s, base, baseparams)
            if new_s[0] > mc.right_bound:
                new_v = 0
            else:
                new_v = fa.vw(w, new_s, base, baseparams)[0]

            e = l * mc.gamma * e
            e += dv
            # δ ← r + γvw(s′,a′) − vw(s,a);
            delta = r + mc.gamma * new_v - v
            # w←w+αδev;
            w += lr * delta * e

            # Actor update
            # θ + αγ^tδ ∂ ln(π(s,a,θ))
            theta += lr * delta * dtheta

            s = new_s
            count += 1

        epi = MountainCarEpisode(mc)
        # print(theta)
        estimated_rewards[x] = epi.run_with_w_softmax(theta, eps(x), base, baseparams)
        print('episode: ', x, ', reward: ', estimated_rewards[x])
    return estimated_rewards


def actor_critic_mc_trail(lr, l, baseparams, eps, base='fourier', epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = actor_critic_mc(lr, l, baseparams, eps, epoch=epoch, base=base)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def draw_plot(data, error, epoch=100, filename='tests.png'):
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(np.array(range(epoch)), data, 'k')
    plt.fill_between(range(epoch), data - error, data + error, alpha=0.3)
    plt.savefig(filename, dpi=200)

    plt.show()


rewards, error = actor_critic_grid_trail(0.1, lambda x: 2, trail=100)
ah.save_cp_csvdata(rewards, error, 'ac_grid.csv')
draw_plot(rewards, error, filename='ac_grid.png')

rewards, err = actor_critic_mc_trail(1e-2, 0.8, {'order': 5}, lambda x: 0.25, trail=100)
ah.save_cp_csvdata(rewards, err, 'ac_mc.csv')
draw_plot(rewards, err, filename='ac_mc.png')