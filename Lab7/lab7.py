import matplotlib.pyplot as plt
import scipy.fft as fourier
from math import pi
from numpy import exp, conj, sqrt, amax, arange

STEP = 0.01


class Signal:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def value(self, x, add_error):
        error = amax(x) if add_error is True else 0
        return self.numerator * exp(-(x / self.denominator) ** 2) + 0.1 * error


def residual(delta, eps, n, t, v1, v2, alpha):
    gamma_sum = 0
    beta_sum = 0
    delx = t / n

    while (1):
        for m in range(n):
            temp1 = (abs(v2[m]) ** 2) * (delx ** 2)
            temp2 = abs(v1[m]) ** 2
            temp3 = 1 + (2 * pi * m / t) ** 2
            divider = (temp1 + alpha * temp3) ** 2
            gamma_sum += temp1 * temp2 * temp3 / divider
            beta_sum += alpha ** 2 * temp2 * temp3 / divider
        beta_sum *= delx / n
        gamma_sum *= delx / n
        new_value = beta_sum - (delta + eps * sqrt(gamma_sum)) ** 2

        if new_value > 0:
            alpha -= STEP
        else:
            break
    return alpha


def sampling(delta, eps, n, t, v1, v2, alpha, k):
    delx = t / n
    h_sum = 0
    for m in range(n):
        temp1 = abs(v2[m]) ** 2 * delx ** 2
        temp2 = exp(-2 * pi * k * m * sqrt(-1 + 0j) / n)
        temp3 = 1 + (2 * pi * m / t) ** 2
        divider = (temp1 + alpha * temp3)
        h_sum += temp2 * v1[m] * conj(v2[m]) / divider
    return delx / n * h_sum


def main():
    border = 5

    # init signals u1, u2
    signal1 = Signal(1, 2)
    signal2 = Signal(1.2, 3)

    # get values

    x_values = arange(- border, border, STEP)
    signal1_values = signal1.value(x_values, False)
    signal2_values = signal2.value(x_values, False)

    signal1_values_with_error = signal1.value(x_values, True)
    signal2_values_with_error = signal2.value(x_values, True)

    eps = max(signal1_values) / 10
    delta = eps

    v1 = fourier.fft(signal1_values_with_error)
    v2 = fourier.fft(signal2_values_with_error)
    n = len(v1)

    alpha = residual(delta, eps, n, 2 * border, v1, v2, 1.1)
    sampling_values = [sampling(delta, eps, n, 2 * border, v1, v2, alpha, x) for x in x_values]

    # plot

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(x_values, signal1_values, 'r')
    ax1.plot(x_values, signal2_values, 'b')

    ax2.plot(x_values, sampling_values, 'b')

    plt.show()


if __name__ == '__main__':
    main()

