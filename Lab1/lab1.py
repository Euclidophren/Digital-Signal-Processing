import matplotlib.pyplot as plt
from numpy import exp, arange
from math import pi, sin


PERIOD = 2
SAMPLES = 150
STEP = 0.01
AMPLITUDE = 2.0
SIGMA = 1.0


class Signal:
    def gauss(self, x):
        return AMPLITUDE * exp(- ((x ** 2) / SIGMA) ** 2)

    def rect(self, x):
        return abs(x) <= abs(PERIOD)


def sinc(x):
    return 0 if x == 0 else sin(x) / x


def restore(t, signal_type):
    max_k = SAMPLES - 1
    min_k = - max_k
    aggr = 0
    delta = PERIOD / (SAMPLES - 1)
    signal = Signal()

    for k in range(min_k, max_k + 1):
        if signal_type == 'gauss':
            n = signal.gauss(k * delta) * sinc(pi * (t / delta - k))
        else:
            n = signal.rect(k * delta) * sinc(pi * (t / delta - k))
        aggr += n

    return aggr


def main():

    t = arange(- (PERIOD + 1), PERIOD + 1, STEP)

    signal = Signal()

    gauss_reference = signal.gauss(t)
    rect_reference = signal.rect(t)

    gauss_restored = [restore(i, 'gauss') for i in t]
    rect_restored = [restore(i, 'rect') for i in t]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot(t, gauss_reference, 'r-')
    ax1.plot(t, gauss_restored, 'b')
    ax1.legend()

    ax2.plot(t, rect_reference, color='red')
    ax2.plot(t, rect_restored, color='green')
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    main()