from grid_plus import Grid
from grid_plus import GridEpisode
from mountaincar import MountainCar
from mountaincar import MountainCarEpisode
import function_approximation as fa
import assignment_helper as ah
import estimation

import numpy as np
import matplotlib.pyplot as plt


# sarsa for one trial
# use decaying eps
def sarsa_lambda_grid(lr, l, eps, epoch=100, searchbound=400):
    grid = Grid()
    grid.pi_params = np.zeros((23, 4))
    grid.softmax()
    actions = grid.action
    estimated_rewards = np.zeros(epoch)

    # Initialize tabular-q arbitrarily
    q = np.zeros((23, 4))

    # for each episode:
    for x in range(epoch):
        # s ∼ d0
        s = grid.d_zero()

        # e ← 0
        e = np.zeros((23, 4))

        # choose a from s using a policy derived from q (e.g., ε-greedy or softmax);
        pi_s = estimation.epsilon_greedy(q[grid.get_index(s)], actions, eps(x))
        a = np.random.choice(actions, 1, p=pi_s)[0]

        # for each time step, until s is the terminal absorbing state do
        while s != [5, 5]:
            # Take action a and observe r and s′;
            new_s, r = grid.P_and_R(s, a)

            # choose new_a from new_s using policy derived from q
            pi_temp = estimation.epsilon_greedy(q[grid.get_index(new_s)], actions, eps(x))
            new_a = np.random.choice(actions, 1, p=pi_temp)[0]

            # e ← γλe + ∂qw(s,a)/∂qw;
            e = l * grid.gamma * e
            e[grid.get_index(s), actions.index(a)] += 1
            # δ ← r + γqw(s′,a′) − qw(s,a);
            delta = r + grid.gamma * q[grid.get_index(new_s), actions.index(new_a)] - q[grid.get_index(s), actions.index(a)]
            # w ← w + αδe;
            q += lr * delta * e

            s = new_s
            a = new_a
        # using q function to estimate the reward and add it to estimated_reward
        # print('episode: ', x, ', q function: ', q)
        grid.pi_params = estimation.epsilon_greedy(q, actions, eps(x))
        grid_epi = GridEpisode(grid, step_bound=searchbound)
        # print('episode: ', x, ', pi: ', grid.pi_params)
        estimated_rewards[x] = grid_epi.run_all_steps()
        print('episode: ', x, ', reward: ', estimated_rewards[x], 'epsilon: ', eps(x))
        # decay *= decay_rate

    return estimated_rewards


# run sarsa in several trails and get plot data
def sarsa_lambda_grid_trail(lr, l, eps, epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = sarsa_lambda_grid(lr, l, eps, epoch=epoch)  # (epoch, )
    std_error = np.std(trail_results, axis=0)
    mean_rewards = np.mean(trail_results, axis=0)
    return mean_rewards, std_error


def sarsa_lambda_mc(lr, l, baseparams, eps, epoch=100, base='fourier'):
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

        # e ← 0
        e = np.zeros(w.shape)

        # choose a from s using a policy derived from q (e.g., ε-greedy or softmax);
        first_q = estimation.epsilon_greedy(fa.qw(w, s, actions, base, baseparams), actions, eps(x))
        # pi_s = pe.epsilon_greedy(pe.qw(w, s, order, actions, base), actions, eps)
        a = np.random.choice(actions, 1, p=first_q)[0]

        count = 0

        while s[0] < mc.right_bound and count < 1e3:
            # Take action a and observe r and s′;
            new_s, r = mc.P_and_R(s, a)

            # Choose a′ from s′ using a policy derived from q;
            pi_temp = estimation.epsilon_greedy(fa.qw(w, new_s, actions, base, baseparams), actions, eps(x))
            new_a = np.random.choice(actions, 1, p=pi_temp)[0]
            if new_s == [0.5, 0]:
                new_q = 0
            else:
                new_q = fa.qw_ele(w, new_s, new_a, actions, base, baseparams)[0]
            q, dqdw = fa.qw_ele(w, s, a, actions, base, baseparams)

            # e←γλe+∂qw(s,a)/∂w;
            e = l * 1 * e + dqdw
            # δ←r+γqw(s′,a′)−qw(s,a);
            delta = r + 1 * new_q - q
            # w←w+αδe;
            w += lr * delta * e

            # print(w)

            s = new_s
            a = new_a
            count += 1
        # print('update end')

        epi = MountainCarEpisode(mc)
        estimated_rewards[x] = epi.run_with_w(w, eps(x), base, baseparams)
        print('episode: ', x, ', reward: ', estimated_rewards[x])
    return estimated_rewards


def sarsa_lambda_mc_trail(lr, l, baseparams, eps, base='fourier', epoch=100, trail=100):
    trail_results = np.zeros((trail, epoch))
    for x in range(trail):
        trail_results[x] = sarsa_lambda_mc(lr, l, baseparams, eps, epoch=epoch, base=base)  # (epoch, )
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


rewards, error = sarsa_lambda_grid_trail(5e-2, 0.95, lambda x: 0.3 if x < 20 else 0.01, trail=100)
ah.save_cp_csvdata(rewards, error, 'sarsa_grid.csv')
draw_plot(rewards, error, filename='sarsa_grid.png')

# rewards, err = sarsa_lambda_mc_trail(1e-2, 0.95, {'order': 3}, trail=100, eps=lambda x: 0.2 if x < 80 else 0.01)
# ah.save_cp_csvdata(rewards, err, 'sarsa_mountaincar_25.csv')
# draw_plot(rewards, err, filename='sarsa_mc_25.png')

rewards, err = sarsa_lambda_mc_trail(1e-2, 0.8, {'order': 7}, trail=100, eps=lambda x: 0.3 if x < 20 else 0.01)
ah.save_cp_csvdata(rewards, err, 'sarsa_mc.csv')
draw_plot(rewards, err, filename='sarsa_mc.png')
