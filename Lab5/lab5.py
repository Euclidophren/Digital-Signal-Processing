import matplotlib
matplotlib.use('TkAgg')
import matplotlib.plt as plt
from math import exp, floor
from numpy import random, linspace
from scipy import signal

PERIOD = 2
POINT_NUMBER = 250
SIGMA = 1.0


class GaussSignal:
    def __init__(self, sigma, amplitude):
        self.sigma = sigma
        self.amplitude = amplitude

    def value(self, t):
        return self.amplitude * exp(- (t ** 2) / (self.sigma ** 2))


def noise_impulse(signal_point_number, noise_point_number, mult):
    step = floor(signal_point_number / noise_point_number)
    result = [0] * signal_point_number
    middle = round(signal_point_number / 2)
    for i in range(floor(noise_point_number / 2)):
        result[middle + i * step] = mult * (0.5 + random.rand())
        result[middle - i * step] = mult * (0.5 + random.rand())
    return result


def butterworth_filter(D, size, filter):
    x = linspace(- size / 2, size / 2, num=size)
    if filter == 'low':
        x1 = [float(v) / D for v in x]
    elif filter == 'high':
        x1 = [float(D) / v for v in x]
    else:
        raise NotImplementedError
    x2 = [1 + v ** 4 for v in x1]
    x3 = [1.0 / v for v in x2]
    sum_x = sum(x3)
    return [v / sum_x for v in x3]


def gauss_filter(sigma, size, filter):
    x = linspace(- size / 2, size / 2, num=size)
    coef = 2 * sigma ** 2
    x1 = [exp(- v ** 2 / coef) for v in x]
    if filter == 'low':
        x2 = x1
    elif filter == 'high':
        x2 = [1 - v for v in x1]
    else:
        raise NotImplementedError
    sum_x = sum(x2)
    return [v / sum_x for v in x2]


def main():
    border_c = 2.0
    border = PERIOD * border_c

    step = 2 * PERIOD / POINT_NUMBER
    amplitude = 1.0

    s_gauss = GaussSignal(SIGMA, amplitude)
    t_values = [- border + i * step for i in range(int(2 * border / step) + 1)]

    gauss_values = [s_gauss.value(t) for t in t_values]
    gauss_noise_values = random.normal(0, 0.05, len(t_values))
    impulse_noise_values = noise_impulse(len(gauss_values), 7, 0.4)

    values_w_gauss = [val1 + val2 for val1, val2 in zip(gauss_values, gauss_noise_values)]
    values_w_impulse = [val1 + val2 for val1, val2 in zip(gauss_values, impulse_noise_values)]

    butterworth_f = butterworth_filter(6, 20, 'high')
    gauss_f = gauss_filter(4, 20, 'high')

    plt.subplot(2, 2, 1)
    plt.plot(t_values, gauss_values, 'r',
                t_values, values_w_gauss, 'y',
                t_values, values_w_gauss - signal.filtfilt(gauss_f, 1, values_w_gauss), 'g')
    plt.legend(['Исходный график', 'Гауссовы помехи', 'Фильтр Гаусса'])

    plt.subplot(2, 2, 2)
    plt.plot(t_values, gauss_values, 'r',
                t_values, values_w_gauss, 'y',
                t_values, values_w_gauss - signal.filtfilt(butterworth_f, 1, values_w_gauss), 'b')
    plt.legend(['Исходный график', 'Гауссовы помехи', 'Фильтр Баттеруорта'])

    plt.subplot(2, 2, 3)
    plt.plot(t_values, gauss_values, 'r',
                t_values, values_w_impulse, 'y',
                t_values, values_w_impulse - signal.filtfilt(gauss_f, 1, values_w_impulse), 'g')
    plt.legend(['Исходный график', 'Импульсные помехи', 'Фильтр Гаусса'])

    plt.subplot(2, 2, 4)
    plt.plot(t_values, gauss_values, 'r',
                t_values, values_w_impulse, 'y',
                t_values, values_w_impulse - signal.filtfilt(butterworth_f, 1, values_w_impulse), 'b')
    plt.legend(['Исходный график', 'Импульсные помехи', 'Фильтр Баттеруорта'])
    plt.show()


if __name__ == '__main__':
    main()
