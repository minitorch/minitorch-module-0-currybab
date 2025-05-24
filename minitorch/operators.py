"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import List

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        x * y
    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        x
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        x + y
    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x: A float.

    Returns:
    -------
        -x
    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        x < y
    """
    return x < y


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        x == y
    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        max(x, y)
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value.

    Args:
    ----
        x: A float.
        y: A float.

    Returns:
    -------
        abs(x - y) < 1e-2
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""Calculates the sigmoid function.

    Args:
    ----
        x: A float.

    Returns:
    -------
        \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function.

    Args:
    ----
        x: A float.

    Returns:
    -------
        max(0, x)
    """
    return max(0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x: A float.

    Returns:
    -------
        math.log(x)
    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential.

    Args:
    ----
        x: A float.

    Returns:
    -------
        math.exp(x)
    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        x: A float.

    Returns:
    -------
        1.0 / x
    """
    return 1.0 / x


def log_back(x: float, grad: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x: A float.
        grad: grad.

    Returns:
    -------
        1.0 / x * grad
    """
    return 1.0 / x * grad


def inv_back(x: float, grad: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
        x: A float.
        grad: grad.

    Returns:
    -------
        -1.0 / (x * x) * grad
    """
    return -1.0 / (x * x) * grad


def relu_back(x: float, grad: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        x: A float.
        grad: grad.

    Returns:
    -------
        grad if x > 0 else 0.0
    """
    return grad if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def addLists(ls1: List[float], ls2: List[float]) -> List[float]:
    """Add two lists together.

    Args:
    ----
        ls1: A list of floats.
        ls2: A list of floats.

    Returns:
    -------
        ls1 + ls2
    """
    return [x + y for x, y in zip(ls1, ls2)]


def negList(ls: List[float]) -> List[float]:
    """Negate a list.

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        -ls
    """
    return [-x for x in ls]


def sum(ls: List[float]) -> float:
    """Sum lists.

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        sum(ls)
    """
    return sum(ls)


def prod(ls: List[float]) -> float:
    """Product of lists.

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        prod(ls)
    """
    return prod(ls)
