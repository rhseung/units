__all__ = [
    'unit',

    'g', 'm', 's', 'A', 'K', 'mol', 'cd', 'rad',
    'N', 'J', 'Pa', 'W', 'atm', 'C', 'V', 'Ω',

    'Y', 'Yg', 'Ym', 'Ys', 'YA', 'YK', 'Ymol', 'Ycd', 'YN', 'YJ', 'YPa', 'YW', 'YV', 'YΩ', 'YT', 'YH', 'YF',
    'Z', 'Zg', 'Zm', 'Zs', 'ZA', 'ZK', 'Zmol', 'Zcd', 'ZN', 'ZJ', 'ZPa', 'ZW', 'ZV', 'ZΩ', 'ZT', 'ZH', 'ZF',
    'E', 'Eg', 'Em', 'Es', 'EA', 'EK', 'Emol', 'Ecd', 'EN', 'EJ', 'EPa', 'EW', 'EV', 'EΩ', 'ET', 'EH', 'EF',
    'P', 'Pg', 'Pm', 'Ps', 'PA', 'PK', 'Pmol', 'Pcd', 'PN', 'PJ', 'PPa', 'PW', 'PV', 'PΩ', 'PT', 'PH', 'PF',
    'T', 'Tg', 'Tm', 'Ts', 'TA', 'TK', 'Tmol', 'Tcd', 'TN', 'TJ', 'TPa', 'TW', 'TV', 'TΩ', 'TT', 'TH', 'TF',
    'G', 'Gg', 'Gm', 'Gs', 'GA', 'GK', 'Gmol', 'Gcd', 'GN', 'GJ', 'GPa', 'GW', 'GV', 'GΩ', 'GT', 'GH', 'GF',
    'M', 'Mg', 'Mm', 'Ms', 'MA', 'MK', 'Mmol', 'Mcd', 'MN', 'MJ', 'MPa', 'MW', 'MV', 'MΩ', 'MT', 'MH', 'MF',
    'k', 'kg', 'km', 'ks', 'kA', 'kK', 'kmol', 'kcd', 'kN', 'kJ', 'kPa', 'kW', 'kV', 'kΩ', 'kT', 'kH', 'kF',
    'h', 'hg', 'hm', 'hs', 'hA', 'hK', 'hmol', 'hcd', 'hN', 'hJ', 'hPa', 'hW', 'hV', 'hΩ', 'hT', 'hH', 'hF',
    'd', 'dg', 'dm', 'ds', 'dA', 'dK', 'dmol', 'dcd', 'dN', 'dJ', 'dPa', 'dW', 'dV', 'dΩ', 'dT', 'dH', 'dF',
    'c', 'cg', 'cm', 'cs', 'cA', 'cK', 'cmol', 'ccd', 'cN', 'cJ', 'cPa', 'cW', 'cV', 'cΩ', 'cT', 'cH', 'cF',
    'milli', 'mg', 'mm', 'ms', 'mA', 'mK', 'mmol', 'mcd', 'mN', 'mJ', 'mPa', 'mW', 'mV', 'mΩ', 'mT', 'mH', 'mF',
    'μ', 'μg', 'μm', 'μs', 'μA', 'μK', 'μmol', 'μcd', 'μN', 'μJ', 'μPa', 'μW', 'μV', 'μΩ', 'μT', 'μH', 'μF',
    'n', 'ng', 'nm', 'ns', 'nA', 'nK', 'nmol', 'ncd', 'nN', 'nJ', 'nPa', 'nW', 'nV', 'nΩ', 'nT', 'nH', 'nF',
    'p', 'pg', 'pm', 'ps', 'pA', 'pK', 'pmol', 'pcd', 'pN', 'pJ', 'pPa', 'pW', 'pV', 'pΩ', 'pT', 'pH', 'pF',
    'f', 'fg', 'fm', 'fs', 'fA', 'fK', 'fmol', 'fcd', 'fN', 'fJ', 'fPa', 'fW', 'fV', 'fΩ', 'fT', 'fH', 'fF',
    'a', 'ag', 'am', 'as', 'aA', 'aK', 'amol', 'acd', 'aN', 'aJ', 'aPa', 'aW', 'aV', 'aΩ', 'aT', 'aH', 'aF',
    'z', 'zg', 'zm', 'zs', 'zA', 'zK', 'zmol', 'zcd', 'zN', 'zJ', 'zPa', 'zW', 'zV', 'zΩ', 'zT', 'zH', 'zF',
    'y', 'yg', 'ym', 'ys', 'yA', 'yK', 'ymol', 'ycd', 'yN', 'yJ', 'yPa', 'yW', 'yV', 'yΩ', 'yT', 'yH', 'yF',
]
assert len(__all__) == len(set(__all__))

from .utils import *
from abc import ABC, abstractmethod
from enum import Enum
from functools import cmp_to_key
from typing import TypeAlias
from numpy import ndarray
import re

ValueType: TypeAlias = int | float | complex | Vector
Iterable: TypeAlias = list | ndarray

def unit(fmt: str | int | float) -> 'UnitBase':
    if isinstance(fmt, str):
        try:
            return Unit._instances[fmt, 1]
        except KeyError:
            return Unit(fmt)
    elif isinstance(fmt, int | float):
        return ComplexUnit(Counter(), fmt)
    else:
        raise TypeError(f"unit: {type(fmt)}")

def expand(x):
    # todo: .expand()
    ...

def si(x):
    # todo: .si()
    ...

def unit_sort_key(a_: 'Unit', b_: 'Unit'):
    a, b = a_.symbol, b_.symbol
    l_a, l_b = len(a), len(b)

    if a == b:
        return 0
    elif a.isupper() and b.islower():  # 대문자가 소문자보다 우선
        return -1
    elif a.islower() and b.isupper():  # 대문자가 소문자보다 우선
        return 1
    elif l_a != l_b:    # 길이가 짧은 것이 뒤로 밀림
        return -1 if l_a > l_b else 1
    else:   # 길이도 같고, 대소문자 여부도 동일한 상태는 사전 순
        return -1 if a < b else 1

def value_to_latex(value: ValueType) -> str:
    if isinstance(value, Vector):
        _value = value._repr_latex_(get=True)
    elif isinstance(value, complex):
        _real, _imag = '', ''

        if value.real != 0:
            _real = str(to_int_if_possible(value.real))
        if value.imag != 0:
            _imag = str(to_int_if_possible(abs(value.imag))) + r' \textit{i}'     # i는 italic으로 표시

        if value.real != 0 and value.imag != 0:     # a+bi -> (a + bi), a-bi -> (a - bi)
            _value = '(' + _real + (' + ' if value.imag >= 0 else ' - ') + _imag + ')'
        elif _real + _imag:     # a + 0i -> a, 0 + bi -> bi
            _value = _real + _imag
        else:   # 0 + 0i -> 0
            _value = '0'
    else:   # int, float
        _value = str(to_int_if_possible(value))

    def simplify_exp(match):
        # 4e-04 -> 4e-4
        # 4e+04 -> 4e4
        # 4e-0 -> 4e0
        # 4e+0 -> 4e0
        # 4e-04 -> 4 \times 10^{-4}

        sign, exp = match.groups()

        if sign == '+':
            exp = exp.lstrip('0') or '0'
        else:  # sign == '-'
            exp = sign + (exp.lstrip('0') or '0')

        return f" \\times 10^{{{exp}}}"

    find_exp = re.compile('e([+-]?)([0-9.]+)')
    _value = find_exp.sub(simplify_exp, _value)

    return _value

class Prefix(Enum):
    Y = 1e24
    Z = 1e21
    E = 1e18
    P = 1e15
    T = 1e12
    G = 1e9
    M = 1e6
    k = 1e3
    h = 1e2
    d = 1e-1
    c = 1e-2
    m = 1e-3
    µ = 1e-6
    n = 1e-9
    p = 1e-12
    f = 1e-15
    a = 1e-18
    z = 1e-21
    y = 1e-24

class UnitError(Exception):
    pass

class UnitBase(ABC):
    def __init__(self):
        self._scale: ValueType = 1

    @abstractmethod
    def __pow__(self, power: int | float) -> 'ComplexUnit':
        return NotImplemented

    @abstractmethod
    def __mul__(self, other) -> 'Unit | ComplexUnit | Quantity':
        if isinstance(other, Quantity):
            return Quantity(other.value, self * other.unit)
        elif isinstance(other, VecLike):
            return Quantity(Vector(*other), self)
        elif isinstance(other, ValueType):
            return Quantity(other, self)
        elif isinstance(other, Iterable):
            return type(other)([self * v for v in other])
        else:
            return NotImplemented

    def __rmul__(self, other) -> 'Unit | ComplexUnit | Quantity':
        return self * other

    def __truediv__(self, other) -> 'Unit | ComplexUnit | Quantity':
        return self * other ** -1

    def __rtruediv__(self, other) -> 'Unit | ComplexUnit | Quantity':
        return self ** -1 * other

    @abstractmethod
    def __repr__(self) -> str:
        return NotImplemented

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def _repr_latex_(self, get=False) -> str:
        return NotImplemented

    def to(self, unit: 'UnitBase') -> 'UnitBase':
        _si_a = self.si()
        _si_b = unit.si()

        if _si_a.elements == _si_b.elements:
            if _si_a.scale == _si_b.scale:
                return unit
            else:
                return ComplexUnit(unit.elements, _si_a.scale / _si_b.scale)
        else:
            return NotImplemented

    def is_dimensionless(self) -> bool:
        return False

    @abstractmethod
    def expand(self) -> 'Unit | ComplexUnit':
        return NotImplemented

    @abstractmethod
    def si(self) -> 'ComplexUnit | Unit':
        return NotImplemented

    @abstractmethod
    def one(self) -> 'UnitBase':
        return NotImplemented

    @property
    def scale(self) -> int | float | complex:
        return self._scale

    @property
    def elements(self) -> Counter:
        return NotImplemented

class Unit(UnitBase):
    _instances: dict[tuple[str, ValueType], 'Unit'] = {}

    def __new__(cls, symbol: str = '', scale: ValueType = 1):
        if (symbol, scale) in cls._instances:
            return cls._instances[symbol, scale]

        instance = super().__new__(cls)
        if scale == 1:     # scale이 1이 아닌 기본 Unit은 ComplexUnit으로 만들어야 하며, 허용되는 이유는 오직 Quantity 생성을 위해서.
            cls._instances[symbol, scale] = instance
        return instance

    def __init__(self, symbol: str = '', scale: ValueType = 1):
        super().__init__()
        if not symbol.isalpha():
            raise ValueError(f"Unit.__init__: '{symbol}' is not a valid unit.")

        self._symbol = symbol
        self._scale = scale

    def __deepcopy__(self) -> "Unit":
        return self

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other) -> bool:
        if isinstance(other, Unit):
            return self.symbol == other.symbol
        else:
            return NotImplemented

    def __pow__(self, power: int | float) -> 'ComplexUnit':
        new_elements = Counter((self, power))
        # power가 0이면 Counter가 알아서 지워줌

        return ComplexUnit(new_elements, self.scale ** power)

    def __mul__(self, other) -> 'ComplexUnit':
        ret = super(Unit, self).__mul__(other)
        if ret is not NotImplemented:
            return ret

        new_elements = Counter((self, 1))

        if isinstance(other, Unit):
            new_elements += Counter((other, 1))
        elif isinstance(other, ComplexUnit):
            new_elements += other.elements
        else:
            return NotImplemented

        return ComplexUnit(new_elements, self.scale * other.scale)

    def __repr__(self) -> str:
        return self.symbol

    def _repr_latex_(self, get=False) -> str:
        if get:
            return self.__repr__()
        else:
            return r'$\mathrm {' + self.__repr__() + '}$'

    def expand(self) -> 'ComplexUnit | Unit':
        return self

    def si(self) -> 'ComplexUnit | Unit':
        return self

    def one(self) -> 'Unit':
        return Unit(self.symbol, 1)

    def represent(self) -> 'Unit | ComplexUnit':
        return self

    @property
    def symbol(self):
        return self._symbol

    @property
    def elements(self) -> Counter:
        return Counter((self, 1))

class ComplexUnit(UnitBase):
    def __init__(self, elements: Counter[Unit] = None, scale: int | float | complex = 1):
        super().__init__()

        self._scale = scale
        self._elements: Counter[Unit] = elements or Counter()

    def __deepcopy__(self) -> "ComplexUnit":
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, ComplexUnit):
            return self.elements == other.elements
        else:
            return NotImplemented

    def __pow__(self, power: int | float) -> 'ComplexUnit':
        new_elements = self.elements.map(lambda p: p * power)

        return ComplexUnit(new_elements, self.scale ** power)

    def __mul__(self, other) -> 'Unit | ComplexUnit | Quantity':
        ret = super(ComplexUnit, self).__mul__(other)
        if ret is not NotImplemented:
            return ret

        new_elements = self.elements.copy()

        if isinstance(other, Unit):
            new_elements += {other: 1}
        elif isinstance(other, ComplexUnit):
            new_elements += other.elements
        else:
            raise TypeError(f"ComplexUnit.__mul__: {type(other)}")

        if len(new_elements) == 1 and self.scale == 1 and all(v == 1 for v in new_elements.values()):
            return next(iter(new_elements))
        else:
            return ComplexUnit(new_elements, self.scale * other.scale)

    def __repr__(self) -> str:
        if self.is_dimensionless():
            txt = f"dimensionless"
        else:
            slash = ['', '']
            dot_sep, pow_sep = '*', '**'

            for _u in sorted(list(self.elements.keys()), key=cmp_to_key(unit_sort_key)):
                _p = to_int_if_possible(self.elements[_u])

                if abs(_p) == 1:
                    slash[_p < 0] += f'{_u.symbol}{dot_sep}'
                elif _p < 0:
                    slash[1] += f'{_u.symbol}{pow_sep}{-_p}{dot_sep}'
                else:
                    slash[0] += f'{_u.symbol}{pow_sep}{_p}{dot_sep}'

            text = slash[0].rstrip(dot_sep) + ('' if not slash[1] else '/' + slash[1].rstrip(dot_sep))
            txt = f"{text}"

        if self.scale != 1:
            txt += f" with scale {self.scale}"

        return txt

    def _repr_latex_(self, get=False) -> str:
        _scale = ''
        if self.scale != 1:
            _scale = value_to_latex(self.scale)

        _unit = ''
        if not self.is_dimensionless():
            _unit = self.__repr__().replace('**', '^').replace('*', r' \cdot ')

        _txt = (_scale + r' \;\, ' + _unit).strip(r'\;\, ')

        if get:
            return _txt
        else:
            return r'$\mathrm {' + _txt + '}$'

    def expand(self) -> 'ComplexUnit':
        return NotImplemented

        # if self.is_dimensionless():
        #     return self
        #
        # ret = None
        # for _u, _p in self.elements.items():
        #     if ret is None:
        #         ret = _u.represent() ** _p
        #     else:
        #         ret *= _u.represent() ** _p
        #
        # return ComplexUnit(ret.elements, self.scale * ret.scale)

    def si(self) -> 'ComplexUnit | Unit':
        if self.is_dimensionless():
            return self

        # N*J .si
        # N .si * J .si

        ret = None
        for _u in sorted(list(self.elements.keys()), key=cmp_to_key(unit_sort_key)):
            _p = to_int_if_possible(self.elements[_u])
            # unit의 타입으로 올 수 있는 것은 Unit, AbbreviateUnit, PrefixUnit
            #  - Unit: si()가 Unit을 반환하므로 문제 없음.
            #  - AbbreviateUnit: si()가 ComplexUnit.si() 를 사용하므로 이 함수와 동일.
            #  - PrefixUnit: si()가 Unit.si() 또는 AbbreviateUnit.si() 를 사용하므로 문제 없거나 이 함수와 동일.

            # 계속 재귀 돌려서 Counter의 첫 번째 Unit이 Unit 타입(상속 말고)이 될 때까지 반복
            if ret is None:
                ret = _u.si() ** _p
            else:
                ret *= _u.si() ** _p

        return ComplexUnit(ret.elements, self.scale * ret.scale)

    def is_dimensionless(self) -> bool:
        return len(self.elements) == 0

    def one(self) -> 'ComplexUnit':
        return ComplexUnit(self.elements, 1)

    @property
    def elements(self):
        return self._elements

class DelayedUnit(Unit):
    def __new__(cls, symbol: str, represent: ComplexUnit, scale: ValueType = 1):
        if (symbol, scale) in cls._instances:
            return cls._instances[symbol, scale]

        instance = super().__new__(cls, symbol, scale)
        if scale == 1:
            cls._instances[symbol, scale] = instance
        return instance

    def __init__(self, symbol: str, represent: ComplexUnit, scale: ValueType = 1):
        super().__init__(symbol, scale)
        self._represent = represent

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other) -> bool:
        if isinstance(other, DelayedUnit):
            return self._represent == other._represent
        else:
            return NotImplemented

    def expand(self) -> ComplexUnit:
        return self._represent

    def si(self) -> ComplexUnit | Unit:
        return self._represent.si()

    def one(self) -> 'DelayedUnit':
        return DelayedUnit(self.symbol, self._represent, 1)

class PrefixUnit(Unit):
    def __new__(cls, prefix: Prefix, unit: Unit, scale: ValueType = 1):
        symbol = prefix.name + unit.symbol
        if symbol and (symbol, scale) in cls._instances:
            return cls._instances[symbol, scale]

        instance = super().__new__(cls, symbol, scale)
        if scale == 1:
            cls._instances[symbol, scale] = instance
        return instance

    def __init__(self, prefix: Prefix, unit: Unit, scale: ValueType = 1):
        if isinstance(unit, PrefixUnit):  # unit은 상속된 Unit 말고, 오직 Unit만 올 수 있음.
            raise TypeError(f"PrefixUnit.__init__: {type(unit)}")

        if unit.symbol == 'kg':
            raise ValueError("PrefixUnit.__init__: kg cannot be prefixed.")

        super().__init__(prefix.name + unit.symbol, scale)
        self._prefix = prefix
        self._unit = unit

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other) -> bool:
        if isinstance(other, PrefixUnit):
            return self.prefix == other.prefix and self.unit == other.unit
        else:
            return NotImplemented

    def __lt__(self, other):
        if self.unit != other:
            raise UnitError("PrefixUnit.__lt__: Cannot compare different units.")
        return self.unit == other and self._prefix.value < 1

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return self != other and self >= other

    def expand(self) -> ComplexUnit:
        return ComplexUnit(Counter((self.unit, 1)), self._prefix.value)

    def si(self) -> ComplexUnit:
        _si = self.unit.si()

        # self._prefix가 1인 경우는 Unit을 반환할 수도 있으나, Prefix 중에 1e0인 것은 없으므로 항상 ComplexUnit을 반환한다.
        if isinstance(_si, Unit):
            return ComplexUnit(Counter((_si, 1)), self._prefix.value * _si.scale)
        else:  # ComplexUnit
            return ComplexUnit(_si.elements, self._prefix.value * _si.scale)

    def one(self) -> 'PrefixUnit':
        return PrefixUnit(self.prefix, self.unit, 1)

    @property
    def prefix(self) -> Prefix:
        return self._prefix

    @property
    def unit(self) -> Unit:
        return self._unit

class Quantity:
    def __init__(self, value: ValueType, unit: UnitBase):
        self._value: ValueType = value * unit.scale
        self._unit: UnitBase = unit.one()

        if isinstance(self._value, int | float):
            self._value = to_int_if_possible(self._value)

    def __iter__(self):
        if isinstance(self.value, Vector):
            yield from self.value
        else:
            yield self.value

    def __bool__(self):
        return bool(self.value)

    def __len__(self):
        if isinstance(self.value, Vector):
            return len(self.value)
        else:
            return 1

    def __abs__(self):
        return Quantity(abs(self.value), self.unit)

    def __format__(self, format_spec):
        return f"{self.value:{format_spec}} [{self.unit}]"

    def __repr__(self) -> str:
        return self.__format__('')

    def _repr_latex_(self) -> str:
        _value = value_to_latex(self.value)
        _unit = self.unit._repr_latex_(get=True)
        _txt = (_value + r' \;\, ' + _unit).strip(r'\;\, ')

        return r'$\mathrm {' + _txt + '}$'

    def __str__(self) -> str:
        return self.__repr__()

    def __pos__(self) -> "Quantity":
        return self

    def __neg__(self) -> "Quantity":
        return Quantity(-self.value, self.unit)

    def __eq__(self, other) -> bool:
        # todo: 2*km == 2000*1000*mm
        if isinstance(other, Quantity):
            if self.value == other.value:  # 값과 단위 모두가 같아야 같은 Quantity, 그러나 값이 0이면 단위가 달라도 동일
                return self.unit == other.unit if self.value != 0 else True
        elif isinstance(other, ValueType):  # 0은 단위가 없든 있든 동일.
            return self.value == 0 and other == 0
        elif isinstance(other, UnitBase):
            self_si = self.unit.si()
            other_si = other.si()

            return self_si.elements == other_si.elements and self.value * self_si.scale == other_si.scale
        else:
            return NotImplemented

    def __lt__(self, other) -> bool:
        # todo: 3*km > 2000*1000*mm
        if isinstance(other, Quantity):
            _si_a = self.si()
            _si_b = other.si()

            if _si_a == _si_b:
                return _si_a.scale * self.value < _si_b.scale * other.value
            else:
                raise UnitError(f"Cannot compare {self.unit} and {other.unit}.")
        else:
            return self.value < other

    def __ge__(self, other) -> bool:
        return not self < other

    def __gt__(self, other) -> bool:
        return self != other and self >= other

    def __le__(self, other) -> bool:
        return not self > other

    def __pow__(self, other) -> "Quantity":
        if isinstance(other, int | float):
            return Quantity(self.value ** other, self.unit ** other)
        else:
            return NotImplemented

    def __add__(self, other) -> "Quantity":
        # todo: dimensionless 일 땐 ValueType와의 연산 가능
        if isinstance(other, Quantity):
            if self.unit == other.unit:
                return Quantity(self.value + other.value, self.unit)

            _si_a = self.unit.si()
            _si_b = other.unit.si()
            if _si_a.elements == _si_b.elements:
                _scale1 = self.unit.prefix.value if isinstance(self.unit, PrefixUnit) else _si_a.scale
                _scale2 = other.unit.prefix.value if isinstance(other.unit, PrefixUnit) else _si_b.scale

                if _scale1 < _scale2:
                    return Quantity(self.value + other.value * _scale2 / _scale1, self.unit)
                else:
                    return Quantity(self.value * _scale1 / _scale2 + other.value, other.unit)
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __radd__(self, other) -> "Quantity":
        return self + other

    def __sub__(self, other) -> "Quantity":
        return self + (-other)

    def __rsub__(self, other) -> "Quantity":
        return -self + other

    def __mul__(self, other) -> "Quantity":
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.unit * other.unit)
        elif isinstance(other, VecLike):
            return Quantity(self.value * Vector(*other), self.unit)
        elif isinstance(other, ValueType):
            return Quantity(self.value * other, self.unit)
        elif isinstance(other, Iterable):
            return type(other)([self * v for v in other])
        else:
            return NotImplemented

    def __rmul__(self, other) -> "Quantity":
        return self * other

    def __truediv__(self, other) -> "Quantity":
        return self * other ** -1

    def __rtruediv__(self, other) -> "Quantity":
        return self ** -1 * other

    def is_vector(self, unit: Unit = None) -> bool:
        return isinstance(self.value, Vector) and (unit is None or self.unit == unit)

    def is_scalar(self, unit: Unit = None) -> bool:
        return not self.is_vector() and (unit is None or self.unit == unit)

    def represent(self):
        return Quantity(self.value, self.unit.expand())

    def si(self):
        return Quantity(self.value, self.unit.si())

    def to(self, unit: UnitBase) -> "Quantity":
        _si_a = self.unit.si()
        _si_b = unit.si()

        if _si_a.elements == _si_b.elements:
            return Quantity(self.value, self.unit.to(unit))
        else:
            raise UnitError(f"Cannot convert {self.unit} to {unit}.")

    @property
    def value(self) -> ValueType:
        return self._value

    @property
    def unit(self) -> UnitBase:
        return self._unit

    @property
    def e(self) -> tuple['Quantity', ...]:
        return tuple(v * self.unit for v in self.value)

kg = Unit('kg')
m = Unit('m')
s = Unit('s')
A = Unit('A')
K = Unit('K')
mol = Unit('mol')
cd = Unit('cd')
rad = Unit('rad')

g = DelayedUnit('g', ComplexUnit(Counter((kg, 1)), 1e-3))
N = DelayedUnit('N', kg * m / s ** 2)
J = DelayedUnit('J', N * m)
Pa = DelayedUnit('Pa', N / m ** 2)
W = DelayedUnit('W', J / s)
atm = DelayedUnit('atm', ComplexUnit(Counter((Pa, 1)), 101325.))
C = DelayedUnit('C', A * s)
V = DelayedUnit('V', J / C)
Ω = DelayedUnit('Ω', V / A)
Wb = DelayedUnit('Wb', V * s)
T = DelayedUnit('T', Wb / m ** 2)
H = DelayedUnit('H', Wb / A)
F = DelayedUnit('F', C / V)

prefix_variants = [g, m, s, A, K, mol, cd, N, J, Pa, W, V, Ω, T, H, F]

for _p in Prefix:
    if _p == Prefix.m:
        globals()['milli'] = ComplexUnit(Counter(), _p.value)
    else:
        globals()[_p.name] = ComplexUnit(Counter(), _p.value)

    for _u in prefix_variants:
        if _p == Prefix.k and _u == g:
            continue

        globals()[_p.name + _u.symbol] = PrefixUnit(_p, _u)
