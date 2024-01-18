from .utils import reverse_func, number_to_numpy, str_pretty, BuiltinNumber, NumpyNumber, ValueType, CalculableDict
from .dimension import Dimension

import re
import numpy as np
from sortedcontainers import SortedDict
from copy import deepcopy
from queue import PriorityQueue
from abc import ABC, abstractmethod
from functools import total_ordering, reduce

class AbstractUnit(ABC):
    def __init__(self, dim: Dimension = Dimension(), scale: BuiltinNumber | ValueType = 1, depth: int = 0):
        if not isinstance(scale, BuiltinNumber | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {BuiltinNumber} or {ValueType}")
        if not isinstance(dim, Dimension):
            raise TypeError(f"unsupported type {type(dim)}, must be Dimension")

        self.scale: ValueType = scale if isinstance(scale, np.ndarray) else number_to_numpy(scale)
        self.dim: Dimension = dim
        self.depth: int = depth

    def __hash__(self) -> int:
        return hash((self.scale, self.dim))

    @abstractmethod
    def __eq__(self, other) -> bool:
        return NotImplemented

    @abstractmethod
    def __format__(self, format_spec: str) -> str:
        return NotImplemented

    def __str__(self) -> str:
        return self.__format__("")

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def _repr_latex_(self) -> str:
        return NotImplemented

class BaseUnit(AbstractUnit):
    _caches: dict[int, 'BaseUnit'] = {}

    def __new__(cls, symbol: str, dim: Dimension, scale: BuiltinNumber | ValueType = 1):
        if not isinstance(symbol, str):
            raise TypeError(f"unsupported type {type(symbol)}, must be str")
        if re.match(r'^[a-zA-Z_]+\d*$', symbol) is None:
            raise ValueError("symbol must be alphabetic or underscore and can be followed by a number")
        if not isinstance(dim, Dimension):
            raise TypeError(f"unsupported type {type(dim)}, must be Dimension")
        if not isinstance(scale, BuiltinNumber | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {BuiltinNumber} or {ValueType}")

        instance = super().__new__(cls)
        instance.dim = dim
        instance.scale = scale if isinstance(scale, np.ndarray) else number_to_numpy(scale)

        hash_value = hash((instance.scale, instance.dim))

        if hash_value not in cls._caches:
            cls._caches[hash_value] = instance

        return cls._caches[hash_value]

    def __init__(self, symbol: str, dim: Dimension):
        super().__init__(dim, 1)
        self.symbol = symbol

    def __deepcopy__(self, memodict={}):
        return self

    def __hash__(self) -> int:
        return hash((self.scale, self.dim))

    def __mul__(self, other: 'BaseUnit') -> 'Units':
        if not isinstance(other, BaseUnit):
            return NotImplemented

        elements: SortedDict[BaseUnit, np.float64] = SortedDict()
        elements[self] = np.float64(1)
        if other in elements:
            elements[other] += np.float64(1)
        else:
            elements[other] = np.float64(1)

        return Units(elements)

    def __truediv__(self, other: 'BaseUnit') -> 'Units':
        if not isinstance(other, BaseUnit):
            return NotImplemented

        elements: SortedDict[BaseUnit, np.float64] = SortedDict()
        elements[self] = np.float64(1)
        if other in elements:
            elements[other] -= np.float64(1)
        else:
            elements[other] = np.float64(-1)

        return Units(elements)

    def __pow__(self, power: BuiltinNumber | NumpyNumber, modulo=None) -> 'Units':
        if not isinstance(power, BuiltinNumber | NumpyNumber):
            return NotImplemented

        return Units(SortedDict({self: np.float64(power)}))

    def __eq__(self, other: 'BaseUnit') -> bool:
        if not isinstance(other, BaseUnit):
            return NotImplemented
        # cc == cm**3 같이 symbol이 달라도 unit alias는 같다.
        return hash(self) == hash(other)

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
        return self.symbol

    def _repr_latex_(self) -> str:
        pass

class Unit(BaseUnit):
    def __new__(cls, symbol: str, base: AbstractUnit, scale: BuiltinNumber | ValueType = 1):
        if not issubclass(type(base), AbstractUnit):
            raise TypeError(f"unsupported type {type(base)}, must be inherit from AbstractUnit")

        return super().__new__(cls, symbol, base.dim, scale)

    def __init__(self, symbol: str, base: AbstractUnit, scale: BuiltinNumber | ValueType = 1):
        super().__init__(symbol, base.dim)
        self.scale = scale if isinstance(scale, np.ndarray) else number_to_numpy(scale)
        self.depth = base.depth + 1
        self.base = base

    def __format__(self, format_spec: str) -> str:
        return f"{self.symbol} ({str_pretty(self.scale)} {self.base})"

class Units(AbstractUnit):
    # todo: elements를 priority queue로 바꾸면, __format__에서 순서대로 출력할 수 있다.
    #   - priority queue를 어떻게 구현할 것인가?
    #       - key_func = cmp_to_key(compare_items); pq = queue.PriorityQueue(key=key_func) 이런 식으로 사용
    #   - priority queue이면서 지수 값을 저장하려면 어떻게 해야하는가?
    #       - (Unit, float) 튜플로 저장하면 됨
    #       - 또는, https://docs.python.org/ko/3.11/library/heapq.html 여기서 볼 수 있듯 PrioritizedItem 클래스를 만들고 지수는 compare=False로 설정하면 됨
    #   - priority queue를 사용하기 굉장히 애매함; Unit 클래스의 비교 연산자를 구현하고 priority queue를 사용하는 것이 더 나을 것 같음
    #   - 생성자에 elements 인자를 없애는게 어떰
    #       - 어차피 처음 생성되는 Units 인스턴스는 Unit 두 개의 곱임 -> __init__(self, unit1: Unit, unit2: Unit)
    #       - 그 다음부터는 Units 인스턴스의 priority queue에 접근해서 직접 삽입을 해주면 됨 (Units * Unit 상황일 때 self.priority_queue.put((unit, 1))) 같은 느낌)
    def __init__(self, elements: SortedDict[BaseUnit, np.float64], dim: Dimension = None, scale: BuiltinNumber | ValueType = None, depth: int = None):
        if not isinstance(elements, SortedDict):
            raise TypeError(f"unsupported type {type(elements)}, must be SortedDict")

        super().__init__()
        self.elements: SortedDict[BaseUnit, np.float64]\
            = SortedDict({unit: np.float64(exponent) for unit, exponent in elements.items() if exponent != 0})

        if dim:
            self.dim = dim
        if scale:
            self.scale = scale
        if depth:
            self.depth = depth

        for unit, exponent in elements.items():
            if not isinstance(unit, BaseUnit):
                raise TypeError(f"unsupported type {type(unit)}, must be BaseUnit")
            if not isinstance(exponent, np.float64):
                raise TypeError(f"unsupported type {type(exponent)}, must be float")

            if not dim:
                self.dim *= unit.dim ** exponent
            if not scale:
                self.scale *= unit.scale ** exponent
            if not depth:
                self.depth = max(self.depth, unit.depth + 1)

    def __deepcopy__(self, memodict={}):
        return Units(deepcopy(self.elements), self.dim, self.scale, self.depth)

    def __mul__(self, other: AbstractUnit) -> 'Units':
        if not isinstance(other, AbstractUnit):
            return NotImplemented

        elements: SortedDict[BaseUnit, np.float64] = deepcopy(self.elements)

        if isinstance(other, BaseUnit):
            if other in elements:
                elements[other] += np.float64(1)
            else:
                elements[other] = np.float64(1)
        elif isinstance(other, Units):
            for unit, exponent in other.elements.items():
                if unit in elements:
                    elements[unit] += exponent
                else:
                    elements[unit] = exponent
        else:
            raise TypeError(f"unsupported type {type(other)}, must be AbstractUnit")

        return Units(
            elements,
            dim=self.dim * other.dim,
            scale=self.scale * other.scale,
            depth=max(self.depth, other.depth + 1)
        )

    def __rmul__(self, other: AbstractUnit) -> 'Units':
        return self * other

    def __truediv__(self, other: AbstractUnit) -> 'Units':
        if not isinstance(other, AbstractUnit):
            return NotImplemented

        elements: SortedDict[BaseUnit, np.float64] = deepcopy(self.elements)

        if isinstance(other, BaseUnit):
            if other in elements:
                elements[other] -= np.float64(1)
            else:
                elements[other] = np.float64(-1)
        elif isinstance(other, Units):
            for unit, exponent in other.elements.items():
                if unit in elements:
                    elements[unit] -= exponent
                else:
                    elements[unit] = -exponent
        else:
            raise TypeError(f"unsupported type {type(other)}, must be AbstractUnit")

        return Units(
            elements,
            dim=self.dim * other.dim,
            scale=self.scale * other.scale,
            depth=max(self.depth, other.depth + 1)
        )

    def __rtruediv__(self, other: AbstractUnit) -> 'Units':
        return other * self ** -1

    def __pow__(self, power: BuiltinNumber | NumpyNumber, modulo=None) -> 'Units':
        if not isinstance(power, BuiltinNumber | NumpyNumber):
            return NotImplemented

        elements: SortedDict[BaseUnit, np.float64] = deepcopy(self.elements)

        for unit, exponent in elements.items():
            elements[unit] *= power

        return Units(
            elements,
            dim=self.dim ** power,
            scale=self.scale ** power,
            depth=self.depth
        )

    def __eq__(self, other: 'Units') -> bool:
        if not isinstance(other, Units):
            raise TypeError(f"unsupported type {type(other)}, must be Units")

        return self.elements == other.elements and self.scale == other.scale

    def __format__(self, format_spec) -> str:
        ret = ""

        for unit, exponent in self.elements.items():
            if exponent == 1:
                ret += f"{unit}"
            else:
                ret += f"{unit}**{exponent}"
            ret += " * "

        return ret.rstrip(" * ")

    def _repr_latex_(self) -> str:
        pass

__all__ = ['BaseUnit', 'Unit', 'Units']
