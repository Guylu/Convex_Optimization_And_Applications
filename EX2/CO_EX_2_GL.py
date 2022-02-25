from inspect import signature
import numdifftools as nd
import numpy as np

eps = 0.00001


def f(x1, x2, x3, x4):
    """
    f as in the exercise
    """
    return (x1 ** 4) * x2 - x3 / ((1 + x2) ** 2) + 100 * x1 * np.exp(x3) + x4 ** 3


def gradient(f, x):
    """
    calc gradient
    """
    movement = np.full(len(signature(f).parameters), eps)
    return f(*(x + movement)) - f(*x) / eps


def hessian(f, x):
    """
    calc hessian
    """
    return nd.Hessian(f)(x)


def run_question():
    """
    maybe run if f?
    """
    return gradient(f, 0.1)
