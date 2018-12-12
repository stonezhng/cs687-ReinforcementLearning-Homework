import numpy as np
import matplotlib.pyplot as plt
import csv


def save_cp_csvdata(reward, err, filename):
    with open(filename, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        data_writer.writerow(['epoch', 'reward', 'error'])
        for i in range(reward.shape[0]):
            data_writer.writerow([i, reward[i], err[i]])


def read_cp_csvdata(epoch, filename):
    reward = np.zeros(epoch)
    err = np.zeros(epoch)
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                reward[line_count-1] = row[1]
                err[line_count-1] = row[2]
            line_count += 1
        print(f'Processed {line_count} lines.')
    return reward, err


def draw_plot(data, error, epoch=100, filename='tests.png'):
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(np.array(range(epoch)), data, 'k')
    plt.fill_between(range(epoch), data - error, data + error, alpha=0.3)
    plt.savefig(filename, dpi=200)

    plt.show()


def draw_multi_bar(x, y_map, filename='result.png'):
    labels = list(y_map.keys())

    plt.xlabel('episode')
    plt.ylabel('reward')

    plt.xticks([x.index(0), x.index(49), x.index(99)], [0, 49, 99])

    for l in labels:
        plt.plot(range(len(x)), y_map[l], linestyle='-', label=l)

    plt.legend(loc='lower right')

    plt.savefig(filename, dpi=200)
    plt.show()


def draw_multi_err(x, y_map, filename):
    labels = list(y_map.keys())

    fig, ax = plt.subplots()
    plt.xlabel('episode')
    plt.ylabel('reward')

    # plt.plot(np.array(range(epoch)), data, 'k')
    # plt.fill_between(range(epoch), data - error, data + error, alpha=0.3)
    # plt.savefig(filename, dpi=200)

    for l in labels:
        ax.plot(x, y_map[l][0], label=l)
        ax.fill_between(x, y_map[l][0] - y_map[l][1], y_map[l][0] + y_map[l][1], alpha=0.1)
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=200)

    plt.show()


def draw_plot1():
    grid_map = {
        'Sarsa': read_cp_csvdata(100, 'sarsa_grid_f.csv'),
        'QLearning': read_cp_csvdata(100, 'qlearning_grid_f.csv'),
        'Sarsa_lambda': read_cp_csvdata(100, 'sarsa_grid.csv'),
        'QLearning_lambda': read_cp_csvdata(100, 'q_grid.csv')
    }
    draw_multi_err(range(100), grid_map, 'grid1.png')

    mc_map = {
        'Sarsa': read_cp_csvdata(100, 'sarsa_mountaincar.csv'),
        'QLearning': read_cp_csvdata(100, 'qlearning_mountaincar.csv'),
        'Sarsa_lambda': read_cp_csvdata(100, 'sarsa_mc.csv'),
        'QLearning_lambda': read_cp_csvdata(100, 'q_mc.csv')
    }
    draw_multi_err(range(100), mc_map, 'mc1.png')


def draw_plot2():
    grid_map = {
        'Actor-Critic': read_cp_csvdata(100, 'ac_grid.csv'),
        'Sarsa_lambda': read_cp_csvdata(100, 'sarsa_grid.csv'),
        'QLearning_lambda': read_cp_csvdata(100, 'q_grid.csv')
    }
    draw_multi_err(range(100), grid_map, 'grid2.png')

    mc_map = {
        'Actor-Critic': read_cp_csvdata(100, 'ac_mc.csv'),
        'Sarsa_lambda': read_cp_csvdata(100, 'sarsa_mc.csv'),
        'QLearning_lambda': read_cp_csvdata(100, 'q_mc.csv')
    }
    draw_multi_err(range(100), mc_map, 'mc2.png')


def draw_plot3():
    grid_map = {
        'REINFORCE': read_cp_csvdata(100, 'rf_grid.csv')
    }
    draw_multi_err(range(100), grid_map, 'grid3.png')


if __name__ == '__main__':
    draw_plot2()
