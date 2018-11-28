from grid_plus import Grid
from grid_plus import GridEpisode
from cartpole import CartPole
from cartpole import CartPoleEpisode
import estimation as pe
import function_approximation
import assignment_helper as ah

import numpy as np
import matplotlib.pyplot as plt


# q-learning for gridworld
def qlearning_grid(lr, eps, epoch=100, searchbound=400):
    q = np.zeros((23, 4))

    grid = Grid()
    grid.pi_params = np.zeros((23, 4))
    grid.softmax()
    actions = grid.action
    estimated_rewards = np.zeros(epoch)

    for x in range(epoch):
        s = grid.d_zero()
        while s != [5, 5]:
            # Choose a from s using a policy derived from q;
            pi_temp = pe.epsilon_greedy(q[grid.get_index(s)], actions, eps(x))
            a = np.random.choice(actions, 1, p=pi_temp)[0]

            # Take action a and observe r and s′;
            new_s, r = grid.P_and_R(s, a)

            q[grid.get_index(s), actions.index(a)] += lr * (
                        r + grid.gamma * np.max(q[grid.get_index(new_s)]) - q[grid.get_index(s), actions.index(a)])
            s = new_s

        grid.pi_params = pe.epsilon_greedy(q, actions, eps(x))
        grid_epi = GridEpisode(grid, step_bound=searchbound)
        # print('episode: ', x, ', pi: ', grid.pi_params)
        estimated_rewards[x] = grid_epi.run_all_steps()
        print('episode: ', x, ', reward: ', estimated_rewards[x], ', epsilon: ', eps(x))

    return estimated_rewards


# 0.11 -> 0.01, rate = 0.8
def qlearning_cartpole(lr, baseparams, decaylambda, base='fourier', epoch=100):
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
        w = np.zeros((1, len(actions) * num_tilings * (tiles_per_tiling**len(s))))

    elif base == 'rbf':
        order = baseparams['order']
        s = cartpole.d_zero()
        w = np.zeros((1, len(actions) * order ** len(s)))

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
        estimated_rewards[x] = epi.run_with_w(w, decaylambda(x), base, baseparams)
        print('episode: ', x, ', reward: ', estimated_rewards[x], ', epsilon: ', decaylambda(x))
        # print('episode: ', x, ', w: ', w)

    return estimated_rewards


def qlearning_grid_trail(lr, eps, epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = qlearning_grid(lr, eps, epoch=epoch)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def qlearning_cp_trail(lr, baseparams, decaylambda, base='fourier', epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = qlearning_cartpole(lr, baseparams, decaylambda, base=base, epoch=epoch)  # (epoch, )
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


# rewards, err = (qlearning_grid_trail(1e-2, lambda x: 0.3 if x < 80 else 0.01, trail=100))
# ah.save_cp_csvdata(rewards, err, 'qlearning_grid_f_1.csv')
# draw_plot(rewards, err, filename='testgridf.png')
#
# rewards, err = (qlearning_cp_trail(5e-3, {'order': 3}, lambda x: 0.1 * (0.8 ** (x-1)) + 0.01, trail=100))
# ah.save_cp_csvdata(rewards, err, 'qlearning_cartpole_f.csv')
# draw_plot(rewards, err, filename='testqcpf.png')

# rewards, err = (qlearning_cp_trail(1e-2, {'num_tilings': 3, 'tiles_per_tiling': 20},
#                                    lambda x: 0.2, base='tile', trail=100))
# ah.save_cp_csvdata(rewards, err, 'qlearning_cartpole_tile.csv')
# draw_plot(rewards, err, filename='testqcptile1.png')

rewards, err = qlearning_cp_trail(8e-3, {'order': 6}, lambda x: 0.01, base='rbf', trail=10)
ah.save_cp_csvdata(rewards, err, 'sarsa_cartpole_rbf.csv')
draw_plot(rewards, err, filename='testqcprbf.png')

