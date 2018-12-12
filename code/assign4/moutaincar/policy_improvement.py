from mountaincar import MountainCar
from mountaincar import MountainCarEpisode
import estimation
import function_approximation as fa
import assignment_helper as ah

import numpy as np
import matplotlib.pyplot as plt


def sarsa_mountaincar(lr, baseparams, eps, epoch=100, base='fourier'):
    mc = MountainCar()
    estimated_rewards = np.zeros(epoch)
    actions = mc.actions
    w = None

    if base == 'fourier':
        order = baseparams['order']
        s = mc.d_zero()
        w = np.zeros((1, len(actions) * (order + 1) ** len(s)))

    elif base == 'tile':
        num_tilings, tiles_per_tiling = baseparams['num_tilings'], baseparams['tiles_per_tiling']
        s = mc.d_zero()
        w = np.zeros((1, len(actions) * num_tilings))

    for x in range(epoch):
        s = mc.d_zero()

        # choose a from s using a policy derived from q (e.g., ε-greedy or softmax);
        first_q = estimation.epsilon_greedy(fa.qw(w, s, actions, base, baseparams), actions, eps(x))
        # pi_s = pe.epsilon_greedy(pe.qw(w, s, order, actions, base), actions, eps)
        a = np.random.choice(actions, 1, p=first_q)[0]

        count = 0

        while s[0] < mc.right_bound:
            # Take action a and observe r and s′;
            new_s, r = mc.P_and_R(s, a)

            # Choose a′ from s′ using a policy derived from q;
            pi_temp = estimation.epsilon_greedy(fa.qw(w, new_s, actions, base, baseparams), actions, eps(x))
            new_a = np.random.choice(actions, 1, p=pi_temp)[0]

            # w += lr * (r + pe.qw_fourier_ele(w, new_s, new_a, order, actions) -
            # pe.qw_fourier_ele(w, s, a, order, actions)) * pe.dqwdw_fourier(s, a, order, actions)
            new_q = fa.qw_ele(w, new_s, new_a, actions, base, baseparams)[0]
            q, dqdw = fa.qw_ele(w, s, a, actions, base, baseparams)
            w += lr * (r + new_q - q) * dqdw

            s = new_s
            a = new_a
            count += 1

        epi = MountainCarEpisode(mc)
        estimated_rewards[x] = epi.run_with_w(w, eps(x), base, baseparams)
        print('episode: ', x, ', reward: ', estimated_rewards[x])
    return estimated_rewards


def sarsa_mc_trail(lr, baseparams, eps, base='fourier', epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = sarsa_mountaincar(lr, baseparams, eps, epoch=epoch, base=base)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def qlearning_mountaincar(lr, baseparams, eps, epoch=100, base='fourier'):
    mc = MountainCar()
    estimated_rewards = np.zeros(epoch)
    actions = mc.actions
    w = None

    if base == 'fourier':
        order = baseparams['order']
        s = mc.d_zero()
        w = np.zeros((1, len(actions) * (order + 1) ** len(s)))

    elif base == 'tile':
        num_tilings, tiles_per_tiling = baseparams['num_tilings'], baseparams['tiles_per_tiling']
        s = mc.d_zero()
        w = np.zeros((1, len(actions) * num_tilings))

    for x in range(epoch):
        s = mc.d_zero()

        while s[0] < mc.right_bound:
            # Choose a′ from s′ using a policy derived from q;
            pi_temp = estimation.epsilon_greedy(fa.qw(w, s, actions, base, baseparams), actions, eps(x))
            a = np.random.choice(actions, 1, p=pi_temp)[0]

            # Take action a and observe r and s′;
            new_s, r = mc.P_and_R(s, a)

            # w += lr * (r + pe.qw_fourier_ele(w, new_s, new_a, order, actions) -
            # pe.qw_fourier_ele(w, s, a, order, actions)) * pe.dqwdw_fourier(s, a, order, actions)
            new_q = np.max(fa.qw(w, new_s, actions, base, baseparams))
            q, dqdw = fa.qw_ele(w, s, a, actions, base, baseparams)
            w += lr * (r + new_q - q) * dqdw

            s = new_s

        epi = MountainCarEpisode(mc)
        estimated_rewards[x] = epi.run_with_w(w, eps(x), base, baseparams)
        print('episode: ', x, ', reward: ', estimated_rewards[x])
    return estimated_rewards


def qlearning_mc_trail(lr, baseparams, eps, base='fourier', epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = qlearning_mountaincar(lr, baseparams, eps, epoch=epoch, base=base)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def draw_plot(data, err, epoch=100, filename='testq.png'):
    fig, ax = plt.subplots()
    plt.xlabel('episode')
    plt.ylabel('reward')
    ax.errorbar(np.array(range(epoch)), data, yerr=err, fmt='o')
    plt.savefig(filename, dpi=200)

    plt.show()


rewards, err = sarsa_mc_trail(2e-2, {'order': 5}, trail=100, eps=lambda x: 0.01)
ah.save_cp_csvdata(rewards, err, 'sarsa_mountaincar.csv')
draw_plot(rewards, err, filename='testsarsamc.png')

rewards, err = qlearning_mc_trail(1e-2, {'order': 5}, trail=100, eps=lambda x: 0.2)
ah.save_cp_csvdata(rewards, err, 'qlearning_mountaincar.csv')
draw_plot(rewards, err, filename='testqlearningmc.png')