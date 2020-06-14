import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from numpy import exp, random


ABS_T = 2
POINT_NUMBER = 10
INT_AMPLITUDE = 2
STEP = 0.01
EPS = 0.05


class GaussSignal:
    def __init__(self, sigma, amplitude):
        self.sigma = sigma
        self.amplitude = amplitude

    def value(self, t):
        return self.amplitude * exp(- (t ** 2) / self.sigma)


def med(values, pos):
    imin = pos - 1
    imax = pos + 1
    ir = 0
    if imin < 0:
        ir = values[imax]
    else:
        if imax > len(values)-1:
            ir = values[imin]
        else:
            if values[imax] > values[imin]:
                ir = values[imin]
            else:
                ir = values[imax]
    return ir


def mean(values, pos):
    ir = 0
    imin = pos - 2
    imax = pos + 2
    for j in range(imin, imax+1):
        if -1 < j < len(values):
            ir = ir + values[j]
    ir = ir / 5
    return ir


def main():
    border = 5
    gauss = GaussSignal(2, 1)
    t_values = np.arange(-border, border + STEP / 2, STEP)
    gauss_base = gauss.value(t_values)

    gauss_int = gauss_base
    for x in range(POINT_NUMBER):
        amp = 0.25 * randint(1, INT_AMPLITUDE)
        pos = randint(0, len(t_values))
        gauss_int[pos] += amp

    mean_values = np.copy(gauss_int)
    for i in range(len(mean_values)):
        tmp = mean(mean_values, i)
        if abs(tmp-mean_values[i]) > EPS:
            mean_values[i] = tmp

    med_values = np.copy(gauss_int)
    for i in range(len(med_values)):
        tmp = med(med_values, i)
        if abs(tmp-med_values[i]) > EPS:
            med_values[i] = tmp

    plt.subplot(311)
    plt.plot(t_values, gauss_int)
    plt.title('Сигнал с помехой')
    plt.subplot(312)
    plt.title('med')
    plt.plot(t_values, med_values)
    plt.subplot(313)
    plt.title('mean')
    plt.plot(t_values, mean_values)
    plt.show()


if __name__ == '__main__':
    main()
