from .utils import total_calculating, str_pretty

class Dimension:
    def __init__(self,
                 length: float = 0.,
                 mass: float = 0.,
                 time: float = 0.,
                 temperature: float = 0.,
                 electric_current: float = 0.,
                 amount_of_substance: float = 0.,
                 luminous_intensity: float = 0.
    ):
        # todo: 지수에 complex?
        self._value: dict[str, float] = {
            'L': length,
            'M': mass,
            'T': time,
            'Θ': temperature,
            'I': electric_current,
            'N': amount_of_substance,
            'J': luminous_intensity
        }

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

    def __mul__(self: 'Dimension', other: 'Dimension'):
        if not isinstance(self, Dimension):
            raise TypeError(f"unsupported type {type(self)}, must be Dimension")
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

    def __truediv__(self: 'Dimension', other: 'Dimension'):
        if not isinstance(self, Dimension):
            raise TypeError(f"unsupported type {type(self)}, must be Dimension")
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

    def __pow__(self: 'Dimension', other: float):
        if not isinstance(self, Dimension):
            raise TypeError(f"unsupported type {type(self)}, must be Dimension")
        if not isinstance(other, float):
            raise TypeError(f"unsupported type {type(other)}, must be float")

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
