from .utils import total_calculating, str_pretty, BuiltinNumber, NumpyNumber

import numpy as np
from copy import deepcopy

class Dimension:
    def __init__(self,
                 length: BuiltinNumber | NumpyNumber = 0.,
                 mass: BuiltinNumber | NumpyNumber = 0.,
                 time: BuiltinNumber | NumpyNumber = 0.,
                 temperature: BuiltinNumber | NumpyNumber = 0.,
                 electric_current: BuiltinNumber | NumpyNumber = 0.,
                 amount_of_substance: BuiltinNumber | NumpyNumber = 0.,
                 luminous_intensity: BuiltinNumber | NumpyNumber = 0.
                 ):
        # todo: 지수에 complex?
        self._value: dict[str, np.float64] = {
            'L': np.float64(length),
            'M': np.float64(mass),
            'T': np.float64(time),
            'Θ': np.float64(temperature),
            'I': np.float64(electric_current),
            'N': np.float64(amount_of_substance),
            'J': np.float64(luminous_intensity)
        }

    def __deepcopy__(self, memodict={}):
        return Dimension(
            self._value['L'],
            self._value['M'],
            self._value['T'],
            self._value['Θ'],
            self._value['I'],
            self._value['N'],
            self._value['J']
        )

    def __hash__(self) -> int:
        return hash(tuple(self._value.items()))

    def __eq__(self, other: 'Dimension') -> bool:
        if not isinstance(other, Dimension):
            raise TypeError(f"unsupported type {type(other)}, must be Dimension")
        return self._value == other._value

    def __str__(self):
        return "*".join([f"{key}" if value == 1 else f"{key}^{str_pretty(value)}" for key, value in self._value.items() if value != 0])

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other: 'Dimension'):
        if not isinstance(other, Dimension):
            raise TypeError(f"unsupported type {type(other)}, must be Dimension")

        return Dimension(
            self._value['L'] + other._value['L'],
            self._value['M'] + other._value['M'],
            self._value['T'] + other._value['T'],
            self._value['Θ'] + other._value['Θ'],
            self._value['I'] + other._value['I'],
            self._value['N'] + other._value['N'],
            self._value['J'] + other._value['J']
        )

    def __truediv__(self, other: 'Dimension'):
        if not isinstance(other, Dimension):
            raise TypeError(f"unsupported type {type(other)}, must be Dimension")

        return Dimension(
            self._value['L'] - other._value['L'],
            self._value['M'] - other._value['M'],
            self._value['T'] - other._value['T'],
            self._value['Θ'] - other._value['Θ'],
            self._value['I'] - other._value['I'],
            self._value['N'] - other._value['N'],
            self._value['J'] - other._value['J']
        )

    def __pow__(self, other: BuiltinNumber | NumpyNumber):
        if not isinstance(other, BuiltinNumber | NumpyNumber):
            raise TypeError(f"unsupported type {type(other)}, must be CompatibleType | DType")

        other = np.float64(other)

        return Dimension(
            self._value['L'] * other,
            self._value['M'] * other,
            self._value['T'] * other,
            self._value['Θ'] * other,
            self._value['I'] * other,
            self._value['N'] * other,
            self._value['J'] * other
        )

__all__ = ['Dimension']
