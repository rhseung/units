__all__ = [
    'builtin_to_numpy_number', 'number_to_str_pretty', 'CalculableDict',
    'BuiltinNumber', 'NumpyNumber', 'Int', 'Float', 'Complex', 'Matrix', 'ValueType',
]

import numpy as np
from typing import Callable, TypeVar

Int = np.int32
Float = np.float64
Complex = np.cdouble
Matrix = np.ndarray

BuiltinNumber = int | float | complex
NumpyNumber = Int | Float | Complex
ValueType = NumpyNumber | Matrix

BinaryOperator = Callable[[any, any], any]

def reverse_func(func: BinaryOperator) -> BinaryOperator:
    name = func.__name__.replace("_", "")

    def f(a, b):
        return func(a, b)
    f.__name__ = f"__r{name}__"
    f.__doc__ = func.__doc__

    return f

def builtin_to_numpy_number(number: BuiltinNumber | NumpyNumber | Matrix) -> ValueType:
    if isinstance(number, int | Int):
        return Int(number)
    elif isinstance(number, float | Float):
        return Float(number)
    elif isinstance(number, complex | Complex):
        return Complex(number)
    elif isinstance(number, Matrix):
        return number
    else:
        raise TypeError(f"unsupported type {type(number)}")

def number_to_str_pretty(number: BuiltinNumber | NumpyNumber) -> str:
    number = builtin_to_numpy_number(number)

    if isinstance(number, Int):
        return f"{number}"
    elif isinstance(number, Float):
        return f"{number}".rstrip("0").rstrip(".")
    elif isinstance(number, Complex):
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
