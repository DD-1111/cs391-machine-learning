import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from scipy.interpolate import make_interp_spline
from sklearn.metrics import mean_squared_error


def readcsv(name, sensor_number):
    sensor_number = str(sensor_number)
    cols = [sensor_number + '_x', sensor_number + '_y', sensor_number + '_z', sensor_number + '_c']
    all_data = np.array(pd.read_csv(name)[cols])
    print("File_loader:" + name + " loaded.")
    return all_data


def data_loader(path):
    all_file = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file != ".DS_Store":
                all_file.append(os.path.join(root, file))
    return all_file


# fix broken data
def data_smoother(data):
    counter = 0
    for i in range(1030):
        if data[i][3] < 0:
            data[i][:3] = data[i - 1][:3] + 0.5 * (data[i - 1][:3] - data[i - 2][:3])
            counter += 1
    print(counter)
    return data


def plot_one_row_data(data, title, index):
    t = range(0, 1030)
    plt.figure(figsize=(12, 6))
    x = []
    for i in range(1030):
        x.append(data[i][index])
    plt.plot(t, x)
    plt.title(title)
    plt.show()
    #plt.savefig(title)


def plot_gp(mean, cov, X, samples=[], title=None, ):
    X = X.ravel()
    mean = mean.ravel()
    plt.figure(figsize=(12, 5))
    plt.fill_between(X, mean + 1.96 * np.sqrt(cov), mean - 1.96 * np.sqrt(cov), alpha=0.1)
    plt.plot(X, mean, label='Predicted')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


def sklearn_predict(t_s, t_train, y_train, sigma):
    f, l, n = sigma

    gpr = GaussianProcessRegressor(kernel=ConstantKernel(f) * RBF(length_scale=l), alpha=n ** 2,
                                   n_restarts_optimizer=8).fit(t_train, y_train)
    score = gpr.score(t_train, y_train)
    means, cov = gpr.predict(t_s, return_cov=True)

    optimal = [np.sqrt(gpr.kernel_.k1.get_params()['constant_value']), gpr.kernel_.k2.get_params()['length_scale'],
               np.exp(n)]
    return means, cov, optimal, score


def RBF_kernel(X1, X2, l=1.0, f=1.0):
    """source: https://stackoverflow.com/questions/55218154/implementation-of-isotropic-squared-exponential-kernel-with-numpy"""
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


def data_augument(x, y, t, window_size, noise):
    x[(window_size // 2):-(window_size // 2)] /= 2
    y[(window_size // 2):-(window_size // 2)] /= 2
    index = list(range(0, 1030, 25))
    for i in range(1030 // 25):
        noise_fix = (0.5 - np.random.random_sample()) *0.5* noise
        x[index[i]] += noise_fix
    print(x)
    smoother = 0
    counter = 0
    for i in range(1030):
        if i in index[:-1]:
            smoother = x[index[counter + 1]]
            counter += 1
        x[i] += (smoother - x[i]) * (i % 25) / 25
    t = np.array(range(0, 1030, 25))
    x = x[index]
    X_Y_Spline = make_interp_spline(index, x)
    T_ = np.linspace(t.min(), t.max(), 1030)
    x = X_Y_Spline(T_)

    return x, y


def sliding_window(win_size, delta, t, x, samples = None):
    parameters = [0.5, 1.0, 0.01]
    means, cov_maxs = np.zeros(1030), np.zeros(1030)
    movement_num = 1030 - win_size
    result = []
    optimal = []

    for i in range(0, movement_num, delta):
        selected_t, selected_x = t[i:i + win_size], x[i:i + win_size]
        gpr = GaussianProcessRegressor(kernel=ConstantKernel(parameters[0]) * RBF(length_scale=parameters[1]),
                                       alpha=parameters[2] ** 2,
                                       n_restarts_optimizer=8).fit(selected_t, selected_x)
        result.append(gpr.score(selected_t, selected_x))
        mean, temp_cov = gpr.predict(selected_t, return_cov=True)
        means[i:i + win_size] += mean
        cov_maxs[i:i + win_size] += np.diag(temp_cov)

        optimal.append(
            [np.sqrt(gpr.kernel_.k1.get_params()['constant_value']), gpr.kernel_.k2.get_params()['length_scale']/100,
             np.exp(parameters[2])])
    performance = np.average(result)
    print(result)
    means, cov_maxs = data_augument(means, cov_maxs, t, win_size, parameters[2])
    plt.figure(figsize=(12, 6))
    title = f'sliding window with a window size of {win_size}, step = {delta} \n average precision = {round(performance, 4)}'
    plot_gp(means, cov_maxs, t, samples=samples, title=title)
    return means, optimal

if __name__ == '__main__':
    five_data = []
    for file in data_loader("data_GP\AG/"):
        five_data.append(readcsv(file, 8))

    # plot_one_row_data(five_data[0], 'raw AG Block 1, x_0 sensor', 0)
    # smoothed = data_smoother(five_data[0])
    # plot_one_row_data(smoothed, 'smoothed AG Block 1, x_0 sensor', 0)

    t = np.array(range(0, 1030)).reshape(-1, 1)
    smoothed_all_data = []
    for each_sample in five_data:
        smoothed_all_data.append(data_smoother(each_sample))

    mean = np.zeros(t.shape[0]).reshape(-1, 1)
    cov = np.diag(RBF_kernel(t, t))
    data = np.array(smoothed_all_data)

    t = np.array(range(0, 1030)).reshape(-1, 1)
    subset = 2
    index = list(range(0, 1030, 25))
    x = smoothed_all_data[subset][:, 0]
    t_train, x_train = t[index], x[index]
    samples = [data[subset, :, 0]]

    # samples = [data[0,:,0],data[1,:,0],data[2,:,0],data[3,:,0],data[4,:,0]]
    # plot_gp(None, cov, t, samples=samples, title="smoothed AG all 5 sample, y_0 sensor")
    initial_para = [0.5, 1.0, 0.1]
    sigma, log, noise = initial_para
    for epoch in range(5):
        mean, cov_s, opt_hp, score = sklearn_predict(t, t_train, x_train, [sigma, log, noise])
        print(opt_hp[0], opt_hp[1], score)
        title = f'sigma = {sigma}, log = {log}, noise_level = {noise}\n precision = {round(score, 4)}'
        plot_gp(mean, np.diag(cov_s), t, samples = samples, title=title)
        sigma, log, noise = opt_hp[0], opt_hp[1] / 100.0, 0.1
    "global kernel converge at 0.377044563894303 85.29727281700298, with a precision of 0.9839, in AG Block3 8_x"
    global_performance = mean
    t = np.array(range(0, 1030)).reshape(-1, 1)
    window_size = 20
    local_performance, records = sliding_window(100, window_size, t, x, samples = samples)
    mse = mean_squared_error(global_performance, x)
    mse_local = mean_squared_error(local_performance, x)
    print(mse)
    print(mse_local)
    """0.001310634237978369
0.013055935482962848"""
    mse = mean_squared_error(global_performance[0:800], x[0:800])
    mse_local = mean_squared_error(local_performance[0:800], x[0:800])
    print(mse)
    print(mse_local)
    """0.0016483400624844149
0.00010143974320715838"""
    historical_s = []
    historical_l = []
    historical_n = []

    for each_record in records:
        historical_s.append(each_record[0])
        historical_l.append(each_record[1])
        historical_n.append(each_record[2])
    t = np.array(range(0, 1030, window_size))
    plt.figure(figsize=(12, 6))
    plt.plot(t[:len(historical_s)], historical_s, '-g', label='historical_s')
    plt.plot(t[:len(historical_s)], historical_l, ':b', label='historical_l')
    plt.plot(t[:len(historical_s)], historical_n, '--r', label='historical_n')
    title = f' parameters’ values of local kernels at each frame\n data collected under window = 100, delta ={window_size}'
    plt.title(title)
    plt.legend();
    plt.show()


