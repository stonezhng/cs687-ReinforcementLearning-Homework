from grid_plus import Grid
from grid_plus import GridEpisode
from cartpole import CartPole
from cartpole import CartPoleEpisode
from moutaincar.mountaincar import MountainCar
from moutaincar.mountaincar import MountainCarEpisode
import estimation as pe
import assignment_helper as ah

import numpy as np
import matplotlib.pyplot as plt


# sarsa for one trial
# use decaying eps
def qlearning_grid(lr, eps, epoch=100, searchbound=400):
    grid = Grid()
    grid.pi_params = np.zeros((23, 4))
    grid.softmax()
    actions = grid.action
    estimated_rewards = np.zeros(epoch)

    q = np.zeros((23, 4))

    for x in range(epoch):
        s = grid.d_zero()

        while s != [5, 5]:
            # choose new_a from new_s using policy derived from q
            pi_temp = pe.softmax(q[grid.get_index(s)], actions, eps(x))
            a = np.random.choice(actions, 1, p=pi_temp)[0]
            # print(q)
            # Take action a and observe r and s′;
            new_s, r = grid.P_and_R(s, a)
            q[grid.get_index(s), actions.index(a)] += lr * (
                    r + grid.gamma * np.max(q[grid.get_index(new_s)]) - q[grid.get_index(s), actions.index(a)])
            s = new_s
        # using q function to estimate the reward and add it to estimated_reward
        # print('episode: ', x, ', q function: ', q)
        grid.pi_params = pe.softmax(q, actions, eps(x))
        grid_epi = GridEpisode(grid, step_bound=searchbound)
        # print('episode: ', x, ', pi: ', grid.pi_params)
        estimated_rewards[x] = grid_epi.run_all_steps()
        print('episode: ', x, ', reward: ', estimated_rewards[x], 'epsilon: ', eps(x))
        # decay *= decay_rate

    return estimated_rewards


def qlearning_cartpole(lr, baseparams, decaylambda, epoch=100, base='fourier'):
    cartpole = CartPole()
    estimated_rewards = np.zeros(epoch)
    actions = cartpole.actions
    w = None

    if base == 'fourier':
        order = baseparams['order']
        s = cartpole.d_zero()
        w = np.zeros((1, len(actions) * (order + 1) ** len(s)))

    elif base == 'tile':
        num_tilings, tiles_per_tiling = baseparams['num_tilings'], baseparams['tiles_per_tiling']
        s = cartpole.d_zero()
        w = np.zeros((1, len(actions) * num_tilings))

    for x in range(epoch):
        s = cartpole.d_zero()

        count = 0

        while np.abs(s[0]) < cartpole.edge and np.abs(s[1]) < cartpole.fail_angle and count < 1010:
            # Choose a′ from s′ using a policy derived from q;
            pi_temp = pe.epsilon_greedy(pe.qw(w, s, actions, base, baseparams), actions, decaylambda(x))
            a = np.random.choice(actions, 1, p=pi_temp)[0]

            # Take action a and observe r and s′;
            new_s, r = cartpole.P_and_R(s, a)

            # w += lr * (r + pe.qw_fourier_ele(w, new_s, new_a, order, actions) -
            # pe.qw_fourier_ele(w, s, a, order, actions)) * pe.dqwdw_fourier(s, a, order, actions)
            new_q = np.max(pe.qw(w, new_s, actions, base, baseparams))
            q, dqdw = pe.qw_ele(w, s, a, actions, base, baseparams)
            w += lr * (r + new_q - q) * dqdw

            s = new_s
            count += 1

        epi = CartPoleEpisode(cartpole)
        estimated_rewards[x] = epi.run_with_w_softmax(w, decaylambda(x), base, baseparams)
        print('episode: ', x, ', reward: ', estimated_rewards[x])
        # print('episode: ', x, ', w: ', w)

    return estimated_rewards


# run sarsa in several trails and get plot data
def qlearning_grid_trail(lr, eps, epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = qlearning_grid(lr, eps, epoch=epoch)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


# run sarsa in several trails and get plot data
def qlearning_cp_trail(lr, baseparams, decaylambda, base='fourier', epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = qlearning_cartpole(lr, baseparams, decaylambda, epoch=epoch, base=base)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def draw_plot(data, error, epoch=100, filename='tests.png'):
    fig, ax = plt.subplots()
    plt.xlabel('episode')
    plt.ylabel('reward')
    ax.errorbar(np.array(range(epoch)), data, yerr=error, fmt='o')
    plt.savefig(filename, dpi=200)

    plt.show()


rewards, error = qlearning_grid_trail(0.5, lambda x: 5, trail=100)
ah.save_cp_csvdata(rewards, error, 'qlearning_grid_f_1.csv')
draw_plot(rewards, error, filename='testqgridf.png')

# easy to go inf; should be careful when tuning; difficult to converge; chaos at the first several loops
rewards, err = qlearning_cp_trail(2e-3, {'order': 3}, lambda x: 0.1 * (0.8 ** (x-1)) + 0.01, trail=100)
ah.save_cp_csvdata(rewards, err, 'q_cartpole_f.csv')
draw_plot(rewards, err, filename='testqcpf.png')

