import numpy as np
from numpy.fft import *
from numpy.random import normal
import matplotlib.pyplot as plt
from math import *
from random import *

A = 1.0
SIGMA = 0.5
MULT = 5
STEP = 0.005


class Pulse():
    def __init__(self, amplitude, sigma):
        self.amplitude = amplitude
        self.sigma = sigma

    def gauss_pulse(self, x):
        return [self.amplitude * exp(- (xi / self.sigma) ** 2) for xi in x]

    def impnoise(self, imp_size, n, mult):
        imp_step = floor(imp_size / n)
        y = np.zeros(imp_size)
        for i in range(floor(n / 2)):
            y[round(round(imp_size / 2) + i * imp_step)] = mult * (0.5 + random())
            y[round(round(imp_size / 2) - i * imp_step)] = mult * (0.5 + random())
        return y


def wiener_filter(x, n):
    y = [1 - ((n[i] / x[i]) ** 2) for i in range(len(n))]
    return y


def main():
    t = np.arange(-MULT, MULT, STEP)

    pulse = Pulse(A, SIGMA)

    # pulse generation
    x0 = pulse.gauss_pulse(t)

    n1 = normal(0, 0.05, len(x0))
    x1 = [x+y for x, y in zip(x0, n1)]

    # impulsive noise generation
    n2 = pulse.impnoise(len(x0), 7, 0.4)
    x2 = [x+y for x, y in zip(x0, n2)]

    # wiener
    y1 = wiener_filter(fft(x1), fft(n1))
    y2 = wiener_filter(fft(x2), fft(n2))

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot(t, x0, 'g', label='Исходный сигнал')
    ax1.plot(t, x1, 'r', label='Гауссовы помехи')
    ax1.plot(t, ifft([x*y for x, y in zip(fft(x1), y1)]), 'b', label='Восстановленный сигнал')
    ax1.legend()

    ax2.plot(t, x0, 'g', label='Исходный сигнал')
    ax2.plot(t, x2, 'r', label='Импульсные помехи')
    ax2.plot(t, ifft([(x*y) for x, y in zip(fft(x2), y2)]), 'b', label='Восстановленный сигнал')
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    main()
