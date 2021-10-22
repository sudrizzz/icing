import numpy as np
from math import cos, sin, pi
from numba import jit


@jit(nopython=True)
def fourier(fx, x, n):
    """
    构造傅里叶函数

    :param fx: 原函数
    :param x: 自变量
    :param n: 展开项个数
    :return: constant 常量
    :return: an 正弦项
    :return: bn 余弦项
    """
    T = x[-1] - x[0]  # 周期
    delta_T = T / (len(x) - 1)
    N = len(x)
    an, bn = np.zeros(n), np.zeros(n)
    for i in range(n):
        a, b = 0, 0
        for j in range(N):
            a += fx[j] * cos(2 * pi * (i + 1) * x[j] / T)
            b += fx[j] * sin(2 * pi * (i + 1) * x[j] / T)
        an[i] = 2 * a * delta_T / T
        bn[i] = 2 * b * delta_T / T

    constant = 0
    for i in range(N):
        constant += fx[i]
    constant = 2 * constant * delta_T / T
    return constant, an, bn


@jit(nopython=True)
def inverse_fourier(constant, an, bn, x):
    """
    反傅里叶变换

    :param constant: 常量
    :param an: 余弦项
    :param bn: 正弦项
    :param x: 自变量
    :return: fx 原函数
    """
    T = x[-1] - x[0]
    N = len(x)
    n = len(bn)
    fx = np.zeros(N)
    for i in range(N):
        fxx = 0
        for j in range(n):
            fxx += an[j] * cos(2 * pi * (j + 1) * x[i] / T) + \
                   bn[j] * sin(2 * pi * (j + 1) * x[i] / T)
        fx[i] = constant / 2 + fxx
    return fx
