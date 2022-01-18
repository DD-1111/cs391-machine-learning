import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from tqdm import tqdm
from scipy.special import expit


def ica(source, n, learning_rate, epoch, amplifier, W):
    global Y
    # t is the length of the signals
    t = source.shape[1]
    one_matrix = np.full((n, t), 1.0)
    i_matrix = np.eye(n)
    new_w = W
    for x in tqdm(range(epoch)):
        # obtain Y=WX, Y is the current estimate of the source
        Y = new_w.dot(source)
        # obtain Z using the formula, 10iter/s using 2*44000 data
        """Z = np.zeros((n, t))
        for i in range(n):
            for j in range(t):
                exp = np.exp(-Y[i, j])
                Z[i, j] = 1 / (1 + exp)"""
        # more efficient way to do sigmoid, 30iter/s using 2*44000 data
        """Z = np.zeros((n, t))
        vect_expit = np.vectorize(expit)
        Z = vect_expit(Y)"""
        # a new magical way with much faster speed to do sigmoid, 700iter/s using 2*44000 data, using intel 8565U
        Z = sigmoid(Y)
        # obtain delta W
        delta_w = learning_rate * np.dot(i_matrix + np.dot((one_matrix - 2 * Z), np.transpose(Y)), W)
        # update W
        new_w = new_w + delta_w
    """ print('\n')
    print('w =')
    print(delta_w)
    print('\n')"""
    # modify the result by a factor to make the output in the same scale as the source
    return Y * amplifier, new_w


def sigmoid(Y):
    return 1 / (1 + np.exp(-Y))


def plot_result(subtitle, data):
    fig, fig2 = plt.subplots(len(data), sharex=True, sharey=False)
    for i in range(len(data)):
        fig2[i].plot(data[i])
    fig.subplots_adjust(hspace=0.3)
    fig.suptitle(subtitle, fontsize=20)
    plt.xlabel('Time Step', fontsize=15)
    plt.ylabel('Signal Values', fontsize=15)
    plt.show()


def generate_random(dimensionx, dimensiony):
    A = np.zeros((dimensionx, dimensiony))
    for i in range(dimensionx):
        for j in range(dimensiony):
            A[i][j] = np.random.uniform(0, 0.1)
    return A


def choose_cut_source(source, index, length):
    """index: 0,1,2,3,4"""
    size = len(index)
    sounds = np.zeros((size, length))
    for i in range(size):
        for j in range(length):
            sounds[i][j] = source[index[i]][j]
    return sounds


def get_correlation_matrix(source_a, source_b):
    result = np.corrcoef(source_a, source_b)
    size = source_a.shape[0]
    # the calculation result of correlation will be a matrix like filled with 1 on the diagonal
    return result[size:2 * size, 0:size]


def small_set_test():
    sounds = io.loadmat('icaTest.mat')['U']
    np.shape(sounds)
    A = io.loadmat('icaTest.mat')['A']
    # mix data
    X = np.dot(A, sounds)
    W = generate_random(3, 3)
    ica_output, W = ica(X, 3, 0.1, 10000, 1, W)

    plot_result('Original', sounds)
    plot_result('Mixed', X)
    plot_result('Reconstructed', ica_output)
    return get_correlation_matrix(ica_output, sounds)


def main():
    # correlation = [[small_set_test()]]
    # fig, ax = plt.subplots(1, 1)
    # ax.axis('tight')
    # ax.axis('off')
    # ax.table(cellText=correlation, loc="center")
    # plt.show()
    sounds = io.loadmat('sounds.mat')['sounds']
    sound_chose = [0, 3, 4]
    sounds = choose_cut_source(sounds, sound_chose, 44000)
    n = len(sound_chose)
    m = sounds.shape[0]
    W = generate_random(n, m)
    A = generate_random(n, n)
    X = np.dot(A, sounds)
    plot_result('Original', sounds)
    plot_result('Mixed', X)
    convergence_detection = False
    w_plot_y = []
    w_plot_x = []
    epoch_group_size = 200
    epoch_group_number = 17
    # for plotting the change of reconstructed signals
    last_w = np.zeros((n, m))
    for i in range(epoch_group_number):
        ica_output, W = ica(X, n, 0.01, epoch_group_size, 1, W)
        plot_result('Reconstructed after ' + str((i + 1) * epoch_group_size) + 'epochs', ica_output)
        correlation = get_correlation_matrix(ica_output, sounds)
        print('\n')
        print('correlation after ' + str((i + 1) * epoch_group_size) + 'epochs')
        print(correlation)
        # if correlation[0, 1] > 0.9 and correlation[1, 0] > 0.9:
        #    break
        delta_w_norm = abs(np.linalg.norm(last_w) - np.linalg.norm(W))
        if convergence_detection:
            if delta_w_norm < np.linalg.norm(last_w) / 1000:
                break
        last_w = W
        w_plot_y.append(delta_w_norm)
        w_plot_x.append((i + 1) * epoch_group_size)
    l = plt.plot(w_plot_x, w_plot_y)
    plt.title('Delta W changing against epochs')
    plt.xlabel('Epochs (numbers of iteration)')
    plt.ylabel('Norm of Delta W')
    plt.show()


if __name__ == '__main__':
    main()
