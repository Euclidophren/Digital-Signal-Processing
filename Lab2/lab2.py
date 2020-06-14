from numpy import linspace, vectorize, abs
import matplotlib.pyplot as plt
from math import exp, cos, sin, pi


class RectangularPulse:
    def __init__(self, border):
        self.border = border

    def value(self, t):
        dif = abs(t) - abs(self.border)
        result = 0.5 if dif == 0 else 0 if dif > 0 else 1
        return result

    def get_border(self):
        return self.border


class GaussianPulse:
    def __init__(self, sigma):
        self.sigma = sigma

    def value(self, t):
        return exp(-pow(t / self.sigma, 2))

    def get_border(self):
        return self.sigma * 3


def iexp(n):
    return complex(cos(n), sin(n))


def dft(values):
    n = len(values)
    return [sum((values[k] * iexp(-2 * pi * i * k / n) for k in range(n)))
            for i in range(n)]


def is_pow2(n):
    return False if n == 0 else (n == 1 or is_pow2(n >> 1))


def fft_(values, n, start=0, stride=1):
    if n == 1: return [values[start]]
    hn, sd = n // 2, stride * 2
    rs = fft_(values, hn, start, sd) + fft_(values, hn, start + stride, sd)
    for i in range(hn):
        e = iexp(-2 * pi * i / n)
        rs[i], rs[i + hn] = rs[i] + e * rs[i + hn], rs[i] - e * rs[i + hn]
        pass
    return rs


def fft(values):
    assert is_pow2(len(values))
    return fft_(values, len(values))

def main():
    gaussian = GaussianPulse(1)
    rect = RectangularPulse(0.3)
    N = 512

    # rectangular
    rect_right_border = rect.get_border() + 2
    rect_left_border = -rect_right_border
    t_rect = 1 / float(N)
    x_rect = linspace(rect_left_border, rect_right_border, N)
    xf_rect = linspace(0.0, 1.0 / t_rect, N)

    vectorized = vectorize(rect.value)
    y_rect = vectorized(x_rect)
    # БПФ
    ffreq = fft(y_rect)
    dfreq = dft(y_rect)
    normal_ffreq = list(reversed(ffreq[0:N // 2])) + ffreq[0:N // 2]
    normal_dfreq = list(reversed(dfreq[0:N // 2])) + dfreq[0:N // 2]
    plt.figure(1)
    plt.subplot(3, 2, 1)
    plt.plot(x_rect, y_rect, 'yellow')
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.title('БПФ')
    plt.plot(xf_rect, 1.0 / N * abs(normal_ffreq[0:N]), 'red', xf_rect, 1.0 / N * abs(ffreq[0:N]), 'orange')
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.title('ДПФ')
    plt.plot(xf_rect, 1.0 / N * abs(normal_dfreq[0:N]), 'black', xf_rect, 1.0 / N * abs(dfreq[0:N]), 'orange')
    plt.grid()

    # gaussian
    gaus_right_border = gaussian.get_border()
    gaus_left_border = -gaussian.get_border()
    t_gaus = 1.0 / N
    x_gaus = linspace(gaus_left_border, gaus_right_border, N)
    xf_gaus = linspace(0.0, 1.0 / t_rect, N)
    vectorized = vectorize(gaussian.value)
    y_gaus = vectorized(x_gaus)

    ffreq = fft(y_gaus)
    dfreq = dft(y_gaus)
    normal_ffreq = list(reversed(ffreq[0:N // 2])) + ffreq[0:N // 2]
    normal_dfreq = list(reversed(dfreq[0:N // 2])) + dfreq[0:N // 2]

    plt.subplot(3, 2, 2)
    plt.plot(x_gaus, y_gaus, 'grey')
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.title('БПФ')
    plt.plot(xf_gaus, 1.0 / N * abs(normal_ffreq[0:N]), 'blue', xf_gaus, 1.0 / N * abs(ffreq[0:N]), 'orange')
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.title('ДПФ')
    plt.plot(xf_gaus, 1.0 / N * abs(normal_dfreq[0:N]), 'green', xf_gaus, 1.0 / N * abs(dfreq[0:N]), 'orange')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()