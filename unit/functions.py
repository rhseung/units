__all__ = ['expand', 'si']

from sortedcontainers import SortedKeysView, SortedDict
from copy import deepcopy

from .core import *
from .utils import Float

def expand(unit: AbstractUnit) -> AbstractUnit:
    if isinstance(unit, Unit):
        return unit.base
    elif isinstance(unit, Units):
        # 다 depth가 똑같으면 맨 앞을 expand
        # depth가 다 똑같지 않으면 가장 depth가 큰 것을 expand

        u: Units = deepcopy(unit)
        keys: SortedKeysView[BaseUnit] = u.elements.keys()
        max_i: int = max(range(len(keys)), key=lambda i: keys[i].depth)
        max_k: BaseUnit = keys[max_i]
        max_v: Float = u.elements[max_k]

        if max_k.depth == 0:
            return u

        del u.elements[max_k]
        u *= expand(max_k) ** max_v

        return u  # test: expand(Units) 테스트
    elif isinstance(unit, AbstractUnit):
        return deepcopy(unit)
    else:
        raise TypeError("unit must be instance of inherited class of AbstractUnit")

def si(unit: AbstractUnit) -> AbstractUnit:
    if isinstance(unit, Unit):
        return si(unit.base)
    elif isinstance(unit, Units):
        tmp = Units(SortedDict({}))

        for k, v in unit.elements.items():
            tmp *= si(k) ** v

        return Units(tmp.elements)  # test: si(Units) 테스트
    elif isinstance(unit, AbstractUnit):
        return deepcopy(unit)
    else:
        raise TypeError("unit must be instance of inherited class of AbstractUnit")
