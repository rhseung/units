from .utils import *
from .dimension import Dimension

import re
from sortedcontainers import SortedDict
from copy import deepcopy
from abc import ABC, abstractmethod

class AbstractUnit(ABC):
    """
    AbstractUnit is an abstract class for every unit class.
    implement basic methods for units.

    Attributes:
        scale: 단위를 si 기본 단위로 변환되었을 때 가지는 크기를 의미한다. (ValueType)
        dim: 단위가 가지는 기본 차원을 의미한다. (Dimension)
        depth: 단위가 복합 단위들에 의해 묶인 횟수를 의미한다. (int)
    """

    def __init__(self, dim: Dimension = Dimension(), scale: BuiltinNumber | ValueType = 1, depth: int = 0):
        if not isinstance(scale, BuiltinNumber | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {BuiltinNumber} or {ValueType}")
        if not isinstance(dim, Dimension):
            raise TypeError(f"unsupported type {type(dim)}, must be Dimension")

        self.scale: ValueType = builtin_to_numpy_number(scale)
        self.dim: Dimension = dim
        self.depth: int = depth

    def __eq__(self, other) -> bool:
        if other == 0:
            return self.scale == 0
        else:
            return isinstance(other, AbstractUnit) and self.scale == other.scale and self.dim == other.dim

    def __str__(self) -> str:
        return self.__format__("")

    def __abs__(self):
        return self.scale

    @abstractmethod
    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def __deepcopy__(self, memodict={}):
        return NotImplemented

    @abstractmethod
    def __format__(self, format_spec: str) -> str:
        return NotImplemented

    @abstractmethod
    def _repr_latex_(self) -> str:
        return NotImplemented

    @abstractmethod
    def __mul__(self, other):
        return NotImplemented

    @abstractmethod
    def __truediv__(self, other):
        return NotImplemented

    @abstractmethod
    def __pow__(self, power, modulo=None):
        return NotImplemented

class BaseUnit(AbstractUnit):
    """
    다른 이름이여도 물리량이 같은 단위 가능 -> aliase (ex. Torr & mmHg)
    같은 이름이여도 물리량이 다른 단위 가능 -> ex. T(Tera) & T(Tesla)
    """

    _caches: dict[tuple[str, Dimension, ValueType], 'BaseUnit'] = {}

    def __new__(cls, symbol: str, dim: Dimension, scale: BuiltinNumber | ValueType = 1):
        if not isinstance(symbol, str):
            raise TypeError(f"unsupported type {type(symbol)}, must be str")
        if re.match(r'^[\S_]+\d*$', symbol) is None:
            raise ValueError("symbol must be alphabetic or greek letter or underscore and can be followed by number")
        if not isinstance(dim, Dimension):
            raise TypeError(f"unsupported type {type(dim)}, must be Dimension")
        if not isinstance(scale, BuiltinNumber | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {BuiltinNumber} or {ValueType}")

        instance = super().__new__(cls)
        instance.symbol = symbol
        instance.dim = dim
        instance.scale = builtin_to_numpy_number(scale)

        key = instance.symbol, instance.dim, instance.scale

        if key not in cls._caches:
            cls._caches[key] = instance

        return cls._caches[key]

    def __init__(self, symbol: str, dim: Dimension):
        super().__init__(dim, 1)
        self.symbol = symbol

    def __deepcopy__(self, memodict={}):
        return self

    def __hash__(self) -> int:
        if self.scale == 0:
            return hash(0)
        else:
            return hash((self.dim, self.scale))

    def __mul__(self, other: 'BaseUnit') -> 'Units':
        if not isinstance(other, BaseUnit):
            return NotImplemented

        structure: SortedDict[BaseUnit, Float] = SortedDict()
        structure[self] = Float(1)
        if other in structure:
            structure[other] += Float(1)
        else:
            structure[other] = Float(1)

        return Units(structure)

    def __truediv__(self, other: 'BaseUnit') -> 'Units':
        if not isinstance(other, BaseUnit):
            return NotImplemented

        structure: SortedDict[BaseUnit, Float] = SortedDict()
        structure[self] = Float(1)
        if other in structure:
            structure[other] -= Float(1)
        else:
            structure[other] = Float(-1)

        return Units(structure)

    def __pow__(self, power: BuiltinNumber | NumpyNumber, modulo=None) -> 'Units':
        if not isinstance(power, BuiltinNumber | NumpyNumber):
            return NotImplemented

        return Units(SortedDict({self: Float(power)}))

    def __lt__(self, other: 'BaseUnit') -> bool:
        if not isinstance(other, BaseUnit):
            return NotImplemented

        a, b = self.symbol, other.symbol

        if a == b:
            return self.scale < other.scale
        elif len(a) != len(b):
            return len(a) > len(b)
        elif a.isupper() and b.islower():
            return True
        elif a.islower() and b.isupper():
            return False
        else:
            return a < b

    def __gt__(self, other: 'BaseUnit') -> bool:
        if not isinstance(other, BaseUnit):
            return NotImplemented

        a, b = self.symbol, other.symbol

        if a == b:
            return self.scale > other.scale
        elif len(a) != len(b):
            return len(a) < len(b)
        elif a.isupper() and b.islower():
            return False
        elif a.islower() and b.isupper():
            return True
        else:
            return a > b

    def __le__(self, other: 'BaseUnit') -> bool:
        return self < other or self == other

    def __ge__(self, other: 'BaseUnit') -> bool:
        return self > other or self == other

    def __format__(self, format_spec: str) -> str:
        if format_spec == 'scale':
            if self.scale == 1:
                return self.symbol
            else:
                return f"{self.symbol} (* {number_to_str_pretty(self.scale)})"
        else:
            return self.symbol

    def __repr__(self) -> str:
        return self.__format__("scale")

    def _repr_latex_(self) -> str:
        pass

class Unit(BaseUnit):
    def __new__(cls, symbol: str, base: AbstractUnit, scale: BuiltinNumber | ValueType = 1):
        if not issubclass(type(base), AbstractUnit):
            raise TypeError(f"unsupported type {type(base)}, must be inherit from AbstractUnit")
        if not isinstance(scale, BuiltinNumber | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {BuiltinNumber} or {ValueType}")

        return super().__new__(cls, symbol, base.dim, scale * base.scale)

    def __init__(self, symbol: str, base: AbstractUnit, scale: BuiltinNumber | ValueType = 1):
        super().__init__(symbol, base.dim)
        self.scale = base.scale * builtin_to_numpy_number(scale)
        self.depth = base.depth + 1
        self.base = base

class Units(AbstractUnit):
    def __new__(cls, structure: SortedDict[BaseUnit, Float], dim: Dimension = None,
                scale: BuiltinNumber | ValueType = None, depth: int = None):
        if not isinstance(structure, SortedDict):
            raise TypeError(f"unsupported type {type(structure)}, must be SortedDict")

        if len(structure) == 1 and (peek := structure.peekitem())[1] == 1:
            return peek[0]
        else:
            return super().__new__(cls)

    def __init__(self, structure: SortedDict[BaseUnit, Float], dim: Dimension = None,
                 scale: BuiltinNumber | ValueType = None, depth: int = None):
        super().__init__()  # 여기서 dim, scale, depth 기본 값 지정
        self.structure: SortedDict[BaseUnit, Float] \
            = SortedDict({unit: Float(exponent) for unit, exponent in structure.items() if exponent != 0})

        if dim:
            self.dim = dim
        if scale:
            self.scale = scale
        if depth:
            self.depth = depth

        for unit, exponent in self.structure.items():
            if not isinstance(unit, BaseUnit):
                raise TypeError(f"unsupported type {type(unit)}, must be BaseUnit")
            if not isinstance(exponent, Float):
                raise TypeError(f"unsupported type {type(exponent)}, must be float")

            if not dim:
                self.dim *= unit.dim ** exponent
            if not scale:
                self.scale *= unit.scale ** exponent
            unit.scale = 1      # fixme: ``sortedcontainers`` 모듈의 SortedDict는 key를 변경하면 KeyError가 남.
            if not depth:
                self.depth = max(self.depth, unit.depth + 1)

    def __deepcopy__(self, memodict={}):
        return Units(deepcopy(self.structure), self.dim, self.scale, self.depth)

    def __mul__(self, other: AbstractUnit) -> 'Units':
        if not isinstance(other, AbstractUnit):
            return NotImplemented

        # todo: other이 dimless라면 scale에만 곱하기

        structure: SortedDict[BaseUnit, Float] = deepcopy(self.structure)

        if isinstance(other, BaseUnit):
            if other in structure:
                structure[other] += Float(1)
            else:
                structure[other] = Float(1)
        elif isinstance(other, Units):
            for unit, exponent in other.structure.items():
                if unit in structure:
                    structure[unit] += exponent
                else:
                    structure[unit] = exponent
        else:
            raise TypeError(f"unsupported type {type(other)}, must be AbstractUnit")

        return Units(
            structure,
            dim=self.dim * other.dim,
            scale=self.scale * other.scale,
            depth=max(self.depth, other.depth + 1)
        )

    def __rmul__(self, other: AbstractUnit) -> 'Units':
        return self * other

    def __truediv__(self, other: AbstractUnit) -> 'Units':
        if not isinstance(other, AbstractUnit):
            return NotImplemented

        structure: SortedDict[BaseUnit, Float] = deepcopy(self.structure)

        if isinstance(other, BaseUnit):
            if other in structure:
                structure[other] -= Float(1)
            else:
                structure[other] = Float(-1)
        elif isinstance(other, Units):
            for unit, exponent in other.structure.items():
                if unit in structure:
                    structure[unit] -= exponent
                else:
                    structure[unit] = -exponent
        else:
            raise TypeError(f"unsupported type {type(other)}, must be AbstractUnit")

        return Units(
            structure,
            dim=self.dim / other.dim,
            scale=self.scale / other.scale,
            depth=max(self.depth, other.depth + 1)
        )

    def __rtruediv__(self, other: AbstractUnit) -> 'Units':
        return other * self ** -1

    def __pow__(self, power: BuiltinNumber | NumpyNumber, modulo=None) -> 'Units':
        if not isinstance(power, BuiltinNumber | NumpyNumber):
            return NotImplemented

        structure: SortedDict[BaseUnit, Float] = deepcopy(self.structure)

        for unit, exponent in structure.items():
            structure[unit] *= power

        return Units(
            structure,
            dim=self.dim ** power,
            scale=self.scale ** power,
            depth=self.depth
        )

    def __format__(self, format_spec: str) -> str:
        front_part = ''
        back_part = ''

        for unit, exponent in self.structure.items():
            if exponent > 0:
                if exponent == 1:
                    front_part += f"{unit}"
                else:
                    front_part += f"{unit}^{number_to_str_pretty(exponent)}"
                front_part += "⋅"
            else:   # exponent == 0인 경우는 구조 상 존재할 수 없음.
                if exponent == -1:
                    back_part += f"{unit}"
                else:
                    back_part += f"{unit}^{number_to_str_pretty(-exponent)}"
                back_part += "⋅"

        ret = (front_part.rstrip('⋅') + '/' + back_part.rstrip('⋅')).rstrip('/')

        if format_spec == 'scale':
            if self.scale == 1:
                return ret
            else:
                return f"{ret} (* {number_to_str_pretty(self.scale)})"
        else:
            return ret

    def __repr__(self) -> str:
        return self.__format__("scale")

    def _repr_latex_(self) -> str:
        pass

class Quantity:
    def __init__(self, value: ValueType, unit: AbstractUnit):
        if not isinstance(value, ValueType):
            raise TypeError(f"unsupported type {type(value)}, must be {ValueType}")
        if not isinstance(unit, AbstractUnit):
            raise TypeError(f"unsupported type {type(unit)}, must be AbstractUnit")

        self.value: ValueType = value
        self.unit: AbstractUnit = unit

    # todo: implement

__all__ = ['AbstractUnit', 'BaseUnit', 'Unit', 'Units', 'Quantity']
