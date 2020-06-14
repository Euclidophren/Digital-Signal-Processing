import matplotlib.pyplot as plt
from numpy.fft import *
from numpy import arange, zeros
from math import exp, floor, ceil

MULT = 5
STEP = 0.05


class Pulse:
    def gauss(self, t, amplitude, sigma):
        return [amplitude * exp(- ((ti ** 2) / (sigma ** 2))) for ti in t]

    def rect(self, t, period, a):
        x = zeros(len(t))
        for i in range(len(t)):
            if abs(t[i]) <= abs(period):
                x[i] = a
            else:
                x[i] = 0
        return x


def convolution(x1, x2, step):
    result = ifft(
        [
            v1 * v2
            for v1, v2 in zip(fft(x1), fft(x2))
        ]
    ) * step
    return result


def main():
    t = arange(-MULT, MULT + STEP, STEP)

    # Pulse generation
    pulse = Pulse()

    rect1 = pulse.rect(t, 2, 2)
    rect2 = pulse.rect(t, 1, 1)
    gauss1 = pulse.gauss(t, 1, 1)
    gauss2 = pulse.gauss(t, 1, 0.5)

    # convolution

    rectrect = convolution(rect1, rect2, STEP)
    rectgauss = convolution(rect1, gauss1, STEP)
    gaussgauss = convolution(gauss1, gauss2, STEP)

    # normalize
    start = (len(rectrect) - len(t)) / 2
    start = floor(start) if start > 0 else ceil(start)

    rectrect = rectrect[start: start + len(t)]
    rectgauss = rectgauss[start: start + len(t)]
    gaussgauss = gaussgauss[start: start + len(t)]

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols = 1)

    print(len(t))
    ax1.plot(t, rect1, color='red', label='Rectangular')
    ax1.plot(t, gauss1, color='blue', label='Gaussian')
    ax1.plot(t, list(rectgauss[101:]) + list(rectgauss[:101]), color='orange', label='Convolution')
    ax1.legend()

    ax2.plot(t, rect1, color='red', label='Rectangular 1')
    ax2.plot(t, rect2, color='blue', label='Rectangular 2')
    ax2.plot(t, list(rectrect[101:]) + list(rectrect[:101]), color='orange', label='Convolution')
    ax2.legend()

    ax3.plot(t, gauss1, color='red', label='Gaussian 1')
    ax3.plot(t, gauss2, color='blue', label='Gaussian 2')
    ax3.plot(t, list(gaussgauss[101:]) + list(gaussgauss[:101]), color='orange', label='Convolution')
    ax3.legend()

    plt.show()


if __name__ == '__main__':
    main()
