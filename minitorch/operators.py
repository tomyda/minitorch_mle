from typing import Callable, Iterable, TypeVar
import math


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Return the input unchanged.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The input number.

    """
    return x


# add - Adds two numbers
def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The negation of x.

    """
    return -x


# lt - Checks if one number is less than another
def lt(x: float, y: float) -> bool:
    """Check if x is less than y.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if x is less than y, 0.0 otherwise.

    """
    return True if x < y else False


# eq - Checks if two numbers are equal
def eq(x: float, y: float) -> bool:
    """Check if x is equal to y.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: 1.0 if x is equal to y, 0.0 otherwise.

    """
    return True if x == y else False


# max - Returns the larger of two numbers
def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is close to y, False otherwise.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid of x.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply the ReLU activation function to x.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The ReLU of x.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculate the natural logarithm of x.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential of x.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal of x.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The reciprocal of x.

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Compute the derivative of log multiplied by d.

    Args:
    ----
        x (float): The input value.
        d (float): The multiplication factor.

    Returns:
    -------
        float: The result of the computation.

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of reciprocal multiplied by d.

    Args:
    ----
        x (float): The input value.
        d (float): The multiplication factor.

    Returns:
    -------
        float: The result of the computation.

    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of ReLU multiplied by d.

    Args:
    ----
        x (float): The input value.
        d (float): The multiplication factor.

    Returns:
    -------
        float: The result of the computation.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def map(fn: Callable[[A], B], iter: Iterable[A]) -> Iterable[B]:
    """Apply a function to each element in an iterable.

    Args:
    ----
        fn (Callable[[A], B]): The function to apply to each element.
        iter (Iterable[A]): The input iterable.

    Returns:
    -------
        Iterable[B]: An iterable containing the results of applying fn to each element in iter.

    """
    return (fn(x) for x in iter)


def zipWith(
    fn: Callable[[A, B], C], iter1: Iterable[A], iter2: Iterable[B]
) -> Iterable[C]:
    """Combine elements from two iterables using a given function.

    Args:
    ----
        fn (Callable[[A, B], C]): The function to combine elements.
        iter1 (Iterable[A]): The first input iterable.
        iter2 (Iterable[B]): The second input iterable.

    Returns:
    -------
        Iterable[C]: An iterable containing the results of applying fn to pairs of elements from iter1 and iter2.

    """
    return (fn(x, y) for x, y in zip(iter1, iter2))


def reduce(fn: Callable[[B, A], B], iter: Iterable[A], start: B) -> B:
    """Reduce an iterable to a single value using a given function.

    Args:
    ----
        fn (Callable[[B, A], B]): The function to combine elements.
        iter (Iterable[A]): The input iterable.
        start (B): The initial value.

    Returns:
    -------
        B: The final reduced value.

    """
    result = start
    for x in iter:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The input list.

    Returns:
    -------
        Iterable[float]: A new list with all elements negated.

    """
    return list(map(lambda x: -x, ls))


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        ls1 (Iterable[float]): The first input list.
        ls2 (Iterable[float]): The second input list.

    Returns:
    -------
        Iterable[float]: A new list with the sum of corresponding elements from ls1 and ls2.

    """
    return list(zipWith(lambda x, y: x + y, ls1, ls2))


def sum(ls: Iterable[float]) -> float:
    """Calculate the sum of all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The input list.

    Returns:
    -------
        float: The sum of all elements in the list.

    """
    return reduce(lambda x, y: x + y, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        ls (Iterable[float]): The input list.

    Returns:
    -------
        float: The product of all elements in the list.

    """
    return reduce(lambda x, y: x * y, ls, 1.0)
