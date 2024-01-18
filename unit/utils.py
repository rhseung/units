__all__ = [
    'total_calculating', 'number_to_numpy', 'str_pretty', 'BuiltinNumber', 'NumpyNumber', 'ValueType', 'CalculableDict'
]

import numpy as np
import operator
from typing import Callable, TypeVar

BinaryOperator = Callable[[any, any], any]
BuiltinNumber = int | float | complex
NumpyNumber = np.int32 | np.float64 | np.cdouble
ValueType = NumpyNumber | np.ndarray

def reverse_func(func: BinaryOperator) -> BinaryOperator:
    name = func.__name__.replace("_", "")

    def f(a, b):
        return func(a, b)
    f.__name__ = f"__r{name}__"
    f.__doc__ = func.__doc__

    return f

def total_calculating(cls):
    reverable_operators: list[BinaryOperator] = [
        operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv,
        operator.mod, operator.pow, operator.matmul, operator.lshift, operator.rshift,
        operator.and_, operator.or_, operator.xor, divmod
    ]

    for oper in reverable_operators:
        operator_name = oper.__name__.replace("_", "")
        func = getattr(cls, f"_{operator_name}", None)
        if func:
            forward, reverse = generate_operator(func, operator_name)
            setattr(cls, forward.__name__, forward)
            setattr(cls, reverse.__name__, reverse)

    return cls

def number_to_numpy(number: BuiltinNumber | NumpyNumber) -> NumpyNumber:
    if isinstance(number, int | np.int32):
        return np.int32(number)
    elif isinstance(number, float | np.float64):
        return np.float64(number)
    elif isinstance(number, complex | np.cdouble):
        return np.cdouble(number)
    else:
        raise TypeError(f"unsupported type {type(number)}")

def str_pretty(number: BuiltinNumber | NumpyNumber) -> str:
    number = number_to_numpy(number)

    if isinstance(number, np.int32):
        return f"{number}"
    elif isinstance(number, np.float64):
        return f"{number}".rstrip("0").rstrip(".")
    elif isinstance(number, np.cdouble):
        real = f"{number.real}".rstrip("0").rstrip(".")
        imag = f"{number.imag:+}".rstrip("0").rstrip(".")
        return f"{real}{imag}i"
    else:
        raise TypeError(f"unsupported type {type(number)}")

T = TypeVar('T')

class CalculableDict(dict[T, float]):
    def __init__(self, iterable=None, /, **kwargs):
        if not all(isinstance(value, float) for value in iterable.values()):
            raise TypeError(f"unsupported type {type(iterable)}, must be float")

        super().__init__(iterable)
        self.update(iterable, **kwargs)

    def __missing__(self, key):
        return 0

    def __mul__(self: 'CalculableDict', other: 'CalculableDict') -> 'CalculableDict':
        if not isinstance(self, CalculableDict):
            raise TypeError(f"unsupported type {type(self)}, must be CalculableDict")
        if not isinstance(other, CalculableDict):
            raise TypeError(f"unsupported type {type(other)}, must be CalculableDict")
        return CalculableDict({key: self[key] + other[key] for key in set(self) | set(other) if self[key] + other[key] != 0})

    def __truediv__(self: 'CalculableDict', other: 'CalculableDict') -> 'CalculableDict':
        if not isinstance(self, CalculableDict):
            raise TypeError(f"unsupported type {type(self)}, must be CalculableDict")
        if not isinstance(other, CalculableDict):
            raise TypeError(f"unsupported type {type(other)}, must be CalculableDict")
        return CalculableDict({key: self[key] - other[key] for key in set(self) | set(other) if self[key] - other[key] != 0})
