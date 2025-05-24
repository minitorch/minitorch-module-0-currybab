"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

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
        abs(x - y) < 1e-4
    """
    return abs(x - y) < 1e-4


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


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable

    Args:
    ----
        fn: A function that takes a float and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats and returns an iterable of floats.
    """

    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes two iterables of floats and returns an iterable of floats.
    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return apply


def reduce(fn: Callable[[float, float], float]) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats and returns a float.
    """

    def apply(ls: Iterable[float]) -> float:
        l = list(ls)
        if len(l) == 0:
            return 0.0
        result = l[0]
        if len(l) > 1:
            for x in l[1:]:
                result = fn(result, x)
        return result

    return apply


# TODO: Implement for Task 0.3.
def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        -ls
    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith

    Args:
    ----
        ls1: An iterable of floats.
        ls2: An iterable of floats.

    Returns:
    -------
        ls1 + ls2
    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        sum(ls)
    """
    return reduce(add)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce

    Args:
    ----
        ls: A list of floats.

    Returns:
    -------
        prod(ls)
    """
    return reduce(mul)(ls)
