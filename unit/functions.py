from sortedcontainers import SortedKeysView, SortedDict
from copy import deepcopy

from .core import *
from .dimension import Dimension
from .unit import L, ohm, cm, m
from .utils import Float

def expand(unit: AbstractUnit, target: AbstractUnit = None) -> AbstractUnit:
    """
    복합 단위를 그 정의로 풀어주는 함수입니다. 예를 들어, `L`은 `cm**3`으로 풀어줍니다. 여러 개의 단위들의 곱의 형태인 경우, 가장 depth가 큰 단위를 풀어줍니다.

    >>> expand(L)
    cm^3
    >>> expand(ohm)
    V/A
    """

    if isinstance(unit, Unit):
        b = unit.base
        return b
    elif isinstance(unit, Units):
        # 다 depth가 똑같으면 맨 앞을 expand
        # depth가 다 똑같지 않으면 가장 depth가 큰 것을 expand

        u: Units = deepcopy(unit)
        keys: SortedKeysView[BaseUnit] = u.structure.keys()
        max_i: int = max(range(len(keys)), key=lambda i: keys[i].depth)
        max_k: BaseUnit = keys[max_i]
        max_v: Float = u.structure[max_k]

        if max_k.depth == 0:
            return u

        del u.structure[max_k]
        u *= expand(max_k) ** max_v

        return u
    elif isinstance(unit, AbstractUnit):
        return deepcopy(unit)
    else:
        raise TypeError("unit must be instance of inherited class of AbstractUnit")

def si(unit: AbstractUnit) -> AbstractUnit:
    """
    단위를 SI 단위로 변환해주는 함수입니다. 예를 들어, `L`은 `m**3`으로 변환해줍니다.

    >>> si(L)
    m^3 (* 1e-6)
    >>> si(ohm)
    kg⋅m^2/A^2⋅s^3
    """

    if isinstance(unit, Unit):
        b = si(unit.base)
        return b
    elif isinstance(unit, Units):
        tmp = Units(SortedDict({}))

        for k, v in unit.structure.items():
            tmp *= si(k) ** v

        return Units(tmp.structure)
    elif isinstance(unit, AbstractUnit):
        return deepcopy(unit)
    else:
        raise TypeError("unit must be instance of inherited class of AbstractUnit")

def dim(unit: AbstractUnit) -> Dimension:
    return unit.dim

def simplify(unit: AbstractUnit) -> AbstractUnit:
    """
    >>> simplify(cm/m**2)
    /m (* 1e-2)
    """

    ...
    # todo: implement
