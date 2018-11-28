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
    fig, ax = plt.subplots()
    plt.xlabel('episode')
    plt.ylabel('reward')
    ax.errorbar(np.array(range(epoch)), data, yerr=error, fmt='o')
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
    for l in labels:
        ax.errorbar(np.array(range(x)), y_map[l][0], yerr=y_map[l][1], fmt='o')
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=200)

    plt.show()


def draw_plot1():
    reward, err = read_cp_csvdata(100, 'sarsa_grid_f_1.csv')
    draw_plot(reward, err, filename='sarsa_grid.png')

    reward, err = read_cp_csvdata(100, 'sarsa_cartpole_f_1.csv')
    draw_plot(reward, err, filename='sarsa_cartpole.png')

    reward, err = read_cp_csvdata(100, 'qlearning_grid_f_1.csv')
    draw_plot(reward, err, filename='qlearning_grid.png')

    reward, err = read_cp_csvdata(100, 'qlearning_cartpole_f.csv')
    draw_plot(reward, err, filename='qlearning_cartpole.png')


def draw_plot3():
    grid_map = {}
    cp_map = {}

    grid_map['sarsa'] = read_cp_csvdata(100, 'sarsa_grid_f_1.csv')[0]
    cp_map['sarsa'] = read_cp_csvdata(100, 'sarsa_cartpole_f_1.csv')[0]
    grid_map['qlearning'] = read_cp_csvdata(100, 'qlearning_grid_f_1.csv')[0]
    cp_map['qlearning'] = read_cp_csvdata(100, 'qlearning_cartpole_f.csv')[0]
    grid_map['cem'] = read_cp_csvdata(100, 'ce_grid.csv')[0]
    cp_map['cem'] = read_cp_csvdata(100, 'ce_cartpole.csv')[0]
    draw_multi_bar(range(100), grid_map, filename='grid_comparision.png')
    draw_multi_bar(range(100), cp_map, filename='cartpole_comparision.png')


def draw_plot4():
    sarsagrid_map = {}
    sarsacp_map = {}
    qgrid_map = {}
    qcp_map = {}

    sarsagrid_map['epsilon greedy'] = read_cp_csvdata(100, 'sarsa_grid_f_1.csv')[0]
    sarsagrid_map['softmax'] = read_cp_csvdata(100, 'softmax/sarsa_grid_f_1.csv')[0]

    sarsacp_map['epsilon greedy'] = read_cp_csvdata(100, 'sarsa_cartpole_f_1.csv')[0]
    sarsacp_map['softmax'] = read_cp_csvdata(100, 'softmax/sarsa_cartpole_f_1.csv')[0]

    qgrid_map['epsilon greedy'] = read_cp_csvdata(100, 'qlearning_grid_f_1.csv')[0]
    qgrid_map['softmax'] = read_cp_csvdata(100, 'softmax/qlearning_grid_f_1.csv')[0]

    qcp_map['epsilon greedy'] = read_cp_csvdata(100, 'qlearning_cartpole_f.csv')[0]
    qcp_map['softmax'] = read_cp_csvdata(100, 'softmax/q_cartpole_f.csv')[0]

    draw_multi_bar(range(100), sarsagrid_map, filename='sarsa_grid_se.png')
    draw_multi_bar(range(100), sarsacp_map, filename='sarsa_cp_se.png')
    draw_multi_bar(range(100), qgrid_map, filename='q_grid_se.png')
    draw_multi_bar(range(100), qcp_map, filename='q_cp_se.png')


def draw_plot5():
    pass

# draw_plot1()
