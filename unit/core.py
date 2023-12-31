from .utils import total_calculating, number_to_numpy, str_pretty, CompatibleType, DType, ValueType, CalculableDict
from .dimension import Dimension

import re
import numpy as np
from queue import PriorityQueue
from abc import ABC, abstractmethod
from functools import total_ordering, reduce

@total_calculating
class AbstractUnit(ABC):
    def __init__(self, dim: Dimension = Dimension(), scale: CompatibleType | ValueType = 1, depth: int = 0):
        if not isinstance(scale, CompatibleType | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {CompatibleType} or {ValueType}")
        if not isinstance(dim, Dimension):
            raise TypeError(f"unsupported type {type(dim)}, must be Dimension")

        self.scale: ValueType = scale if isinstance(scale, np.ndarray) else number_to_numpy(scale)
        self.dim: Dimension = dim
        self.depth: int = depth

    @abstractmethod
    def __eq__(self, other) -> bool:
        return NotImplemented

    @abstractmethod
    def __format__(self, format_spec) -> str:
        return NotImplemented

    def __str__(self) -> str:
        return self.__format__("")

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def _repr_latex_(self) -> str:
        return NotImplemented

class BaseUnit(AbstractUnit):
    _instances: dict[int, 'BaseUnit'] = {}

    def __new__(cls, symbol: str, dim: Dimension, scale: CompatibleType | ValueType = 1):
        if not isinstance(symbol, str):
            raise TypeError(f"unsupported type {type(symbol)}, must be str")
        if re.match(r'^[a-zA-Z_]+\d*$', symbol) is None:
            raise ValueError("symbol must be alphabetic or underscore and can be followed by a number")
        if not isinstance(dim, Dimension):
            raise TypeError(f"unsupported type {type(dim)}, must be Dimension")
        if not isinstance(scale, CompatibleType | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {CompatibleType} or {ValueType}")

        instance = super().__new__(cls)
        instance.dim = dim
        instance.scale = scale if isinstance(scale, np.ndarray) else number_to_numpy(scale)

        hash_value = hash(instance)

        if hash_value not in cls._instances:
            cls._instances[hash_value] = instance

        return cls._instances[hash_value]

    def __init__(self, symbol: str, dim: Dimension):
        super().__init__(dim, 1)
        self.symbol = symbol

    def __hash__(self):
        # {cc: 1} + {cm**3: 2} = {cc: 3} 같이 unit alias는 같은 hash를 가져야 한다.
        return hash((self.scale, self.dim))

    def __eq__(self, other: 'BaseUnit') -> bool:
        if not isinstance(other, BaseUnit):
            raise TypeError(f"unsupported type {type(other)}, must be Unit")
        # cc == cm**3 같이 unit alias는 같다.
        return hash(self) == hash(other)

    def __format__(self, format_spec: str) -> str:
        return self.symbol

    def _repr_latex_(self) -> str:
        pass

class Unit(BaseUnit):
    def __new__(cls, symbol: str, base: AbstractUnit, scale: CompatibleType | ValueType = 1):
        if not issubclass(type(base), AbstractUnit):
            raise TypeError(f"unsupported type {type(base)}, must be inherit from AbstractUnit")

        return super().__new__(cls, symbol, base.dim, scale)

    def __init__(self, symbol: str, base: AbstractUnit, scale: CompatibleType | ValueType = 1):
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
    def __init__(self, elements: CalculableDict[BaseUnit], scale: CompatibleType | ValueType = 1):
        if not isinstance(elements, CalculableDict):
            raise TypeError(f"unsupported type {type(elements)}, must be CalculableDict")
        if not isinstance(scale, CompatibleType | ValueType):
            raise TypeError(f"unsupported type {type(scale)}, must be {CompatibleType} or {ValueType}")

        super().__init__(scale=scale)
        self._elements = elements

        for k, v in self._elements.items():
            if not isinstance(k, BaseUnit):
                raise TypeError(f"unsupported type {type(k)}, must be Unit")
            self.dim *= k.dim ** v

    def __eq__(self, other: 'Units') -> bool:
        if not isinstance(other, Units):
            raise TypeError(f"unsupported type {type(other)}, must be Units")

        return self._elements == other._elements and self.scale == other.scale

    def __format__(self, format_spec) -> str:
        pass

    def _repr_latex_(self) -> str:
        pass

__all__ = ['BaseUnit', 'Unit', 'Units']
