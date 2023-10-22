from .vector import *
from .utils import Counter
from abc import ABC, abstractmethod
from enum import Enum

ScaleType = int | float | complex
ValueType = ScaleType | Vector

def to_int_if_possible(value: float) -> int | float:
    return int(value) if value.is_integer() else value

def unit(fmt: str | int | float) -> 'UnitBase':
    if isinstance(fmt, str):
        try:
            return Unit._instances[fmt]
        except KeyError:
            return Unit(fmt)
    elif isinstance(fmt, int | float):
        return ComplexUnit(Counter({}), fmt)
    else:
        raise TypeError(f"unit: {type(fmt)}")

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

# todo: 모든 클래스의 __eq__를 테스트해보자. 뭔가 양쪽에서 NotImplemented가 나오는 허점이 있을 것 같다.
# todo: _repr_latex_ 구현하기

class UnitBase(ABC):
    def __init__(self):
        self._scale: ScaleType = 1

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
    def represent(self) -> 'Unit | ComplexUnit':
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

class Unit(UnitBase):
    _instances: dict[tuple[str, ScaleType], 'Unit'] = {}

    def __new__(cls, symbol: str = '', scale: ScaleType = 1):
        if (symbol, scale) in cls._instances:
            return cls._instances[symbol, scale]

        instance = super().__new__(cls)
        if scale == 1:     # scale이 1이 아닌 기본 Unit은 ComplexUnit으로 만들어야 하며, 허용되는 이유는 오직 Quantity 생성을 위해서.
            cls._instances[symbol, scale] = instance
        return instance

    def __init__(self, symbol: str = '', scale: ScaleType = 1):
        super().__init__()
        if not symbol.isalpha():
            raise ValueError(f"Unit.__init__: '{symbol}' is not a valid unit.")

        self._symbol = symbol
        self._scale = scale

    def __deepcopy__(self) -> "Unit":
        return self

    def __hash__(self):
        return hash(self._symbol)

    def __eq__(self, other) -> bool:
        if isinstance(other, Unit):
            return self._symbol == other._symbol
        else:
            return NotImplemented

    def __pow__(self, power: int | float) -> 'ComplexUnit':
        new_counter = Counter({self: float(power)})
        # power가 0이면 Counter가 알아서 지워줌

        return ComplexUnit(new_counter, self.scale ** power)

    def __mul__(self, other) -> 'ComplexUnit':
        ret = super(Unit, self).__mul__(other)
        if ret is not NotImplemented:
            return ret

        new_counter = Counter({self: 1})

        if isinstance(other, Unit):
            new_counter += {other: 1}
        elif isinstance(other, ComplexUnit):
            new_counter += other.counter
        else:
            return NotImplemented

        return ComplexUnit(new_counter, self.scale * other.scale)

    def __repr__(self) -> str:
        return self._symbol

    def represent(self) -> 'ComplexUnit | Unit':
        return self

    def si(self) -> 'ComplexUnit | Unit':
        return self

    def one(self) -> 'Unit':
        return Unit(self._symbol, 1)

    @property
    def symbol(self):
        return self._symbol

    @property
    def counter(self) -> Counter:
        return Counter({self: 1})

class ComplexUnit(UnitBase):
    def __init__(self, elements: Counter[Unit] = None, scale: int | float | complex = 1):
        super().__init__()

        self._scale = scale
        self._counter: Counter[Unit] = elements or Counter({})

    def __deepcopy__(self) -> "ComplexUnit":
        return self

    def __eq__(self, other):
        if isinstance(other, ComplexUnit):
            return self._counter == other._counter
        else:
            return NotImplemented

    def __pow__(self, power: int | float) -> 'ComplexUnit':
        new_counter = self._counter.map(lambda p: p * power)

        return ComplexUnit(new_counter, self._scale ** power)

    def __mul__(self, other) -> 'Unit | ComplexUnit | Quantity':
        ret = super(ComplexUnit, self).__mul__(other)
        if ret is not NotImplemented:
            return ret

        new_counter = self._counter.copy()

        if isinstance(other, Unit):
            new_counter += {other: 1}
        elif isinstance(other, ComplexUnit):
            new_counter += other._counter
        else:
            raise TypeError(f"ComplexUnit.__mul__: {type(other)}")

        if len(new_counter) == 1 and self._scale == 1:
            return next(iter(new_counter))
        else:
            return ComplexUnit(new_counter, self._scale * other.scale)

    def __repr__(self) -> str:
        if self.is_dimensionless():
            txt = f"dimensionless"
        else:
            slash = ['', '']
            dot_sep, pow_sep = '*', '**'

            for _u, _p in self._counter.items():
                _p = to_int_if_possible(_p)

                if abs(_p) == 1:
                    slash[_p < 0] += f'{_u.symbol}{dot_sep}'
                elif _p < 0:
                    slash[1] += f'{_u.symbol}{pow_sep}{-_p}{dot_sep}'
                else:
                    slash[0] += f'{_u.symbol}{pow_sep}{_p}{dot_sep}'

            text = slash[0].rstrip(dot_sep) + ('' if not slash[1] else '/' + slash[1].rstrip(dot_sep))
            txt = f"{text}"

        if self._scale != 1:
            txt += f" with scale {self._scale}"

        return txt

    def represent(self) -> 'ComplexUnit':
        if self.is_dimensionless():
            return self

        ret = None
        for unit, power in self._counter.items():
            if ret is None:
                ret = unit.represent() ** power
            else:
                ret *= unit.represent() ** power

        return ComplexUnit(ret.counter, self._scale * ret.scale)

    def si(self) -> 'ComplexUnit | Unit':
        if self.is_dimensionless():
            return self

        # N*J .si
        # N .si * J .si

        ret = None
        for unit, power in self._counter.items():
            # unit의 타입으로 올 수 있는 것은 Unit, AbbreviateUnit, PrefixUnit
            #  - Unit: si()가 Unit을 반환하므로 문제 없음.
            #  - AbbreviateUnit: si()가 ComplexUnit.si() 를 사용하므로 이 함수와 동일.
            #  - PrefixUnit: si()가 Unit.si() 또는 AbbreviateUnit.si() 를 사용하므로 문제 없거나 이 함수와 동일.

            # 계속 재귀 돌려서 Counter의 첫 번째 Unit이 Unit 타입(상속 말고)이 될 때까지 반복
            if ret is None:
                ret = unit.si() ** power
            else:
                ret *= unit.si() ** power

        return ComplexUnit(ret.counter, self._scale * ret.scale)

    def is_dimensionless(self) -> bool:
        return len(self._counter) == 0

    def one(self) -> 'ComplexUnit':
        return ComplexUnit(self._counter, 1)

    @property
    def counter(self):
        return self._counter

class DelayedUnit(Unit):
    def __new__(cls, symbol: str, represent: ComplexUnit, scale: ScaleType = 1):
        if (symbol, scale) in cls._instances:
            return cls._instances[symbol, scale]

        instance = super().__new__(cls, symbol, scale)
        if scale == 1:
            cls._instances[symbol, scale] = instance
        return instance

    def __init__(self, symbol: str, represent: ComplexUnit, scale: ScaleType = 1):
        super().__init__(symbol, scale)
        self._represent = represent

    def __hash__(self):
        return hash(self._symbol)

    def __eq__(self, other):
        if isinstance(other, DelayedUnit):
            return self._represent == other._represent
        else:
            return NotImplemented

    def represent(self) -> ComplexUnit:
        return self._represent

    def si(self) -> ComplexUnit | Unit:
        return self._represent.si()

    def one(self) -> 'DelayedUnit':
        return DelayedUnit(self._symbol, self._represent, 1)

class PrefixUnit(Unit):
    def __new__(cls, prefix: Prefix, unit: Unit, scale: ScaleType = 1):
        symbol = prefix.name + unit.symbol
        if symbol and (symbol, scale) in cls._instances:
            return cls._instances[symbol, scale]

        instance = super().__new__(cls, symbol, scale)
        if scale == 1:
            cls._instances[symbol, scale] = instance
        return instance

    def __init__(self, prefix: Prefix, unit: Unit, scale: ScaleType = 1):
        if isinstance(unit, PrefixUnit):  # unit은 상속된 Unit 말고, 오직 Unit만 올 수 있음.
            raise TypeError(f"PrefixUnit.__init__: {type(unit)}")

        if unit.symbol == 'kg':
            raise ValueError("PrefixUnit.__init__: kg cannot be prefixed.")

        super().__init__(prefix.name + unit.symbol, scale)
        self._prefix = prefix
        self._unit = unit

    def __hash__(self):
        return hash(self._symbol)

    def __eq__(self, other):
        if isinstance(other, PrefixUnit):
            return self._prefix == other._prefix and self._unit == other._unit
        else:
            return NotImplemented

    def __lt__(self, other):
        if self._unit != other:
            raise UnitError("PrefixUnit.__lt__: Cannot compare different units.")
        return self._unit == other and self._prefix.value < 1

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return self != other and self >= other

    def represent(self) -> ComplexUnit:
        return ComplexUnit(Counter({self._unit: 1}), self._prefix.value)

    def si(self) -> ComplexUnit:
        _si = self._unit.si()

        # self._prefix가 1인 경우는 Unit을 반환할 수도 있으나, Prefix 중에 1e0인 것은 없으므로 항상 ComplexUnit을 반환한다.
        if isinstance(_si, Unit):
            return ComplexUnit(Counter({_si: 1}), self._prefix.value * _si.scale)
        else:  # ComplexUnit
            return ComplexUnit(_si.counter, self._prefix.value * _si.scale)

    def one(self) -> 'PrefixUnit':
        return PrefixUnit(self._prefix, self._unit, 1)

    @property
    def prefix(self):
        return self._prefix

class Quantity:
    def __init__(self, value: ValueType, unit: UnitBase):
        self._value = value * unit.scale
        self._unit = unit.one()

    def __iter__(self):
        if isinstance(self._value, Vector):
            yield from self._value
        else:
            yield self._value

    def __bool__(self):
        return bool(self._value)

    def __len__(self):
        if isinstance(self._value, Vector):
            return len(self._value)
        else:
            return 1

    def __abs__(self):
        return Quantity(abs(self._value), self._unit)

    def __format__(self, format_spec):
        return f"{self._value:{format_spec}} [{self._unit}]"

    def __repr__(self) -> str:
        return self.__format__('')

    def __str__(self) -> str:
        return self.__repr__()

    def __pos__(self) -> "Quantity":
        return self

    def __neg__(self) -> "Quantity":
        return Quantity(-self._value, self._unit)

    def __eq__(self, other) -> bool:
        if isinstance(other, Quantity):
            if self._value == other._value:  # 값과 단위 모두가 같아야 같은 Quantity, 그러나 값이 0이면 단위가 달라도 동일
                return self._unit == other._unit if self._value != 0 else True
        elif isinstance(other, ValueType):  # 0은 단위가 없든 있든 동일.
            return self._value == 0 and other == 0
        elif isinstance(other, UnitBase):
            self_si = self._unit.si()
            other_si = other.si()

            return self_si.counter == other_si.counter and self._value * self_si.scale == other_si.scale
        else:
            return NotImplemented

    def __lt__(self, other) -> bool:
        if isinstance(other, Quantity):
            if self._unit.si() == other._unit.si():
                return self._value < other._value
            else:
                raise UnitError(f"Cannot compare {self._unit} and {other._unit}.")
        else:
            return self._value < other

    def __ge__(self, other) -> bool:
        return not self < other

    def __gt__(self, other) -> bool:
        return self != other and self >= other

    def __le__(self, other) -> bool:
        return not self > other

    def __pow__(self, other) -> "Quantity":
        if isinstance(other, int | float):
            return Quantity(self._value ** other, self._unit ** other)
        else:
            return NotImplemented

    def __add__(self, other) -> "Quantity":
        if isinstance(other, Quantity):
            # todo: PrefixUnit + Unit 은 Prefix가 작은 쪽으로 합쳐지도록
            if self._unit.si().counter == other._unit.si().counter:
                return Quantity(self._value + other._value, self._unit)
            else:
                raise UnitError(f"Cannot add {self._unit} and {other._unit}.")
        else:
            return Quantity(self._value + other, self._unit)

    def __radd__(self, other) -> "Quantity":
        return self + other

    def __sub__(self, other) -> "Quantity":
        return self + (-other)

    def __rsub__(self, other) -> "Quantity":
        return -self + other

    def __mul__(self, other) -> "Quantity":
        # todo: 테스트
        if isinstance(other, Quantity):
            return Quantity(self._value * other._value, self._unit * other._unit)
        elif isinstance(other, VecLike):
            return Quantity(self._value * Vector(*other), self._unit)
        elif isinstance(other, ValueType):
            return Quantity(self._value * other, self._unit)
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
        return isinstance(self._value, Vector) and (unit is None or self.unit == unit)

    def is_scalar(self, unit: Unit = None) -> bool:
        return not self.is_vector() and (unit is None or self.unit == unit)

    def represent(self):
        return Quantity(self._value, self._unit.represent())

    def si(self):
        return Quantity(self._value, self._unit.si())

    @property
    def value(self) -> ValueType:
        return self._value

    @property
    def unit(self) -> UnitBase:
        return self._unit

    @property
    def e(self) -> tuple['Quantity', ...]:
        return tuple(v * self.unit for v in self._value)

kg = Unit('kg')
m = Unit('m')
s = Unit('s')
A = Unit('A')
K = Unit('K')
mol = Unit('mol')
cd = Unit('cd')
rad = Unit('rad')

g = DelayedUnit('g', ComplexUnit(Counter({kg: 1}), 1e-3))
N = DelayedUnit('N', kg * m / s ** 2)
J = DelayedUnit('J', N * m)
Pa = DelayedUnit('Pa', N / m ** 2)
W = DelayedUnit('W', J / s)
atm = DelayedUnit('atm', ComplexUnit(Counter({Pa: 1}), 101325.))
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
        globals()['milli'] = ComplexUnit(Counter({}), _p.value)
    else:
        globals()[_p.name] = ComplexUnit(Counter({}), _p.value)

    for _u in prefix_variants:
        if _p == Prefix.k and _u == g:
            continue

        globals()[_p.name + _u.symbol] = PrefixUnit(_p, _u)

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

# 겹치는 단위 있는지 체크
assert len(__all__) == len(set(__all__))
