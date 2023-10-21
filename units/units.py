from .vector import *
from .utils import Counter
from abc import ABC, abstractmethod
from enum import Enum

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

# todo: _repr_latex_ 구현하기

class UnitBase(ABC):
	def __init__(self):
		self._scale: int | float | complex = 1.
	
	@abstractmethod
	def __pow__(self, power: int | float) -> 'ComplexUnit':
		return NotImplemented
	
	@abstractmethod
	def __mul__(self, other) -> 'Unit | ComplexUnit':
		return NotImplemented
	
	def __rmul__(self, other) -> 'Unit | ComplexUnit':
		return self * other
	
	def __truediv__(self, other) -> 'Unit | ComplexUnit':
		return self * other ** -1
	
	def __rtruediv__(self, other) -> 'Unit | ComplexUnit':
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

	@property
	def scale(self) -> int | float | complex:
		return self._scale

class Unit(UnitBase):
	_instances: dict[str, 'Unit'] = {}
	
	def __new__(cls, symbol: str):
		if symbol in cls._instances:
			return cls._instances[symbol]
	
		instance = super().__new__(cls)
		cls._instances[symbol] = instance
		return instance
	
	def __init__(self, symbol: str = ''):
		super().__init__()
		if not symbol.isalpha():
			raise ValueError(f"Unit.__init__: '{symbol}' is not a valid unit.")
		
		self._symbol = symbol
	
	def __deepcopy__(self) -> "Unit":
		return self
	
	def __hash__(self):
		return hash(self._symbol)
	
	def __eq__(self, other) -> bool:
		return isinstance(other, Unit) and self._symbol == other._symbol
	
	def __pow__(self, power: int | float) -> 'ComplexUnit':
		new_counter = Counter({self: float(power)})
		# power가 0이면 Counter가 알아서 지워줌
		
		return ComplexUnit(new_counter, self.scale ** power)
	
	def __mul__(self, other) -> 'ComplexUnit':
		new_counter = Counter({self: 1})
		
		if isinstance(other, Unit):
			new_counter += {other: 1}
		elif isinstance(other, ComplexUnit):
			new_counter += other.counter
		else:
			raise TypeError(f"Unit.__mul__: {type(other)}")
		
		return ComplexUnit(new_counter, self.scale * other.scale)
	
	def __repr__(self) -> str:
		return self._symbol
	
	def represent(self) -> 'ComplexUnit | Unit':
		return self
	
	def si(self) -> 'ComplexUnit | Unit':
		return self
	
	@property
	def symbol(self):
		return self._symbol

class ComplexUnit(UnitBase):
	def __init__(self, elements: Counter[Unit] = None, scale: int | float | complex = 1.):
		super().__init__()
		
		self._scale = scale
		self._counter: Counter[Unit] = elements or Counter({})

	def __deepcopy__(self) -> "ComplexUnit":
		return self

	def __eq__(self, other):
		return isinstance(other, ComplexUnit) and self._counter == other._counter

	def __pow__(self, power: int | float) -> 'ComplexUnit':
		new_counter = self._counter.map(lambda p: p * power)

		return ComplexUnit(new_counter, self._scale ** power)

	def __mul__(self, other) -> 'Unit | ComplexUnit':
		new_counter = self._counter.copy()
		
		if isinstance(other, Unit):
			new_counter += {other: 1}
		elif isinstance(other, ComplexUnit):
			new_counter += other._counter
		else:
			raise TypeError(f"ComplexUnit.__mul__: {type(other)}")
		
		if len(new_counter) == 1 and self._scale == 1.:
			return next(iter(new_counter))
		else:
			return ComplexUnit(new_counter, self._scale * other.scale)

	def __repr__(self):
		if self.is_dimensionless():
			return f"dimensionless with scale {self._scale}"
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
			return f"{text} with scale {self._scale}"

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
	
	@property
	def counter(self):
		return self._counter

class DelayedUnit(Unit):
	def __new__(cls, symbol: str, represent: ComplexUnit):
		if symbol in cls._instances:
			return cls._instances[symbol]
		
		instance = super().__new__(cls, symbol)
		cls._instances[symbol] = instance
		return instance
	
	def __init__(self, symbol: str, represent: ComplexUnit):
		super().__init__(symbol)
		self._represent = represent

	def __hash__(self):
		return hash(self._symbol)
	
	def __eq__(self, other):
		return isinstance(other, DelayedUnit) and self._represent == other._represent

	def represent(self) -> ComplexUnit:
		return self._represent

	def si(self) -> ComplexUnit | Unit:
		return self._represent.si()

class PrefixUnit(Unit):
	def __new__(cls, prefix: Prefix, unit: Unit):
		symbol = prefix.name + unit.symbol
		if symbol and symbol in cls._instances:
			return cls._instances[symbol]
		
		instance = super().__new__(cls, symbol)
		cls._instances[symbol] = instance
		return instance
	
	def __init__(self, prefix: Prefix, unit: Unit):
		if isinstance(unit, PrefixUnit):
			raise TypeError(f"PrefixUnit.__init__: {type(unit)}")
		
		if unit.symbol == 'kg':
			raise ValueError("PrefixUnit.__init__: kg cannot be prefixed.")
		
		super().__init__(prefix.name + unit.symbol)
		self._prefix = prefix
		self._unit = unit
	
	def __hash__(self):
		return hash(self._symbol)
	
	# todo: km == Mm/1000 같은 Quantities 비교하기 -> 이건 Quantity에서 구현해야 할 듯
	def __eq__(self, other):
		return isinstance(other, PrefixUnit) and self._prefix == other._prefix and self._unit == other._unit
	
	def represent(self) -> ComplexUnit:
		return ComplexUnit(Counter({self._unit: 1}), self._prefix.value)

	def si(self) -> ComplexUnit:
		_si = self._unit.si()
		
		# self._prefix가 1인 경우는 Unit을 반환할 수도 있으나, Prefix 중에 1e0인 것은 없으므로 항상 ComplexUnit을 반환한다.
		if isinstance(_si, Unit):
			return ComplexUnit(Counter({_si: 1}), self._prefix.value * _si.scale)
		else:   # ComplexUnit
			return ComplexUnit(_si.counter, self._prefix.value * _si.scale)

	@property
	def prefix(self):
		return self._prefix

# todo: Quantity 클래스 구현하기

class Quantity:
	def __init__(self, value: float | Vector, unit: UnitBase):
		self._value = value
		self._unit = unit

	def is_vector(self, unit: Unit = None) -> bool:
		return isinstance(self._value, Vector) and (unit is None or self.unit == unit)

# class Quantity:
# 	def __init__(self, value: float | Vector, unit: Unit):
# 		self.__value = value
# 		self.__unit = unit
#
# 	def is_vector(self, unit: Unit = None) -> bool:
# 		return isinstance(self.__value, Vector) and (unit is None or self.unit == unit)
#
# 	def is_scalar(self, unit: Unit = None) -> bool:
# 		return not self.is_vector() and (unit is None or self.unit == unit)
#
# 	@property
# 	def value(self) -> float | Vector:
# 		return self.__value
#
# 	@property
# 	def unit(self) -> Unit:
# 		return self.__unit
#
# 	@property
# 	def e(self):
# 		return tuple(v * self.unit for v in self.__value)
#
# 	def __iter__(self):
# 		yield self.e
#
# 	def __bool__(self):
# 		return bool(self.__value)
#
# 	def __len__(self):
# 		return len(self.__value)
#
# 	def __abs__(self):
# 		return Quantity(abs(self.__value), self.__unit)
#
# 	def _repr_latex_(self):
# 		# todo: self.value 가 벡터일 때 latex 표현하기
# 		# fixme: this is a temporary fix
# 		return f"$\mathrm{{ {self.value if self.is_scalar() else self.value._repr_latex_(True)}~~{self.unit._repr_latex_(True)} }}$"
#
# 	def __format__(self, format_spec):
# 		return f"{self.__value:{format_spec}} [{self.__unit}]"
#
# 	def __repr__(self) -> str:
# 		return self.__format__('')
#
# 	def __str__(self) -> str:
# 		return self.__repr__()
#
# 	def __pos__(self) -> "Quantity":
# 		return self
#
# 	def __neg__(self) -> "Quantity":
# 		return Quantity(-self.__value, self.__unit)
#
# 	def __eq__(self, other) -> bool:
# 		if isinstance(other, Quantity):
# 			if self.__value == other.__value:
# 				return True if self.__value == 0 else self.__unit == other.__unit
# 		elif isinstance(other, int | float):
# 			return self.__value == 0 and other == 0
# 		else:
# 			return NotImplemented
#
# 	def __lt__(self, other) -> bool:
# 		if isinstance(other, Quantity):
# 			if self.__unit == other.__unit:
# 				return self.__value < other.__value
# 			else:
# 				raise UnitError(f"Cannot compare {self.__unit} and {other.__unit}.")
# 		else:
# 			return self.__value < other
#
# 	def __gt__(self, other) -> bool:
# 		return other < self
#
# 	def __le__(self, other) -> bool:
# 		return self == other or self < other
#
# 	def __ge__(self, other) -> bool:
# 		return self == other or self > other
#
# 	def __add__(self, other) -> "Quantity":
# 		if isinstance(other, Quantity):
# 			if self.__unit == other.__unit:
# 				return Quantity(self.__value + other.__value, self.__unit)
# 			else:
# 				raise UnitError(f"Cannot add {self.__unit} and {other.__unit}.")
# 		else:
# 			return Quantity(self.__value + other, self.__unit)
#
# 	def __radd__(self, other) -> "Quantity":
# 		return self.__add__(other)
#
# 	def __sub__(self, other) -> "Quantity":
# 		return self.__add__(-other)
#
# 	def __rsub__(self, other) -> "Quantity":
# 		return -(self.__add__(-other))
#
# 	def __mul__(self, other) -> "Quantity":
# 		if isinstance(other, Quantity):
# 			return Quantity(self.__value * other.__value, self.__unit * other.__unit)
# 		elif isinstance(other, VecLike):
# 			return Quantity(self.__value * Vector(*other), self.__unit)
# 		elif isinstance(other, int | float):
# 			return Quantity(self.__value * other, self.__unit)
# 		elif isinstance(other, Iterable):
# 			for i in range(len(other)):
# 				other[i] *= self
# 			return other
# 		else:
# 			return NotImplemented
#
# 	def __rmul__(self, other) -> "Quantity":
# 		return self.__mul__(other)
#
# 	def __truediv__(self, other) -> "Quantity":
# 		return self.__mul__(1 / other)
#
# 	def __rtruediv__(self, other) -> "Quantity":
# 		if isinstance(other, VecLike):
# 			return Quantity(Vector(*other) / self.__value, -self.__unit)
# 		elif isinstance(other, int | float):
# 			return Quantity(other / self.__value, -self.__unit)
# 		else:
# 			return Quantity(other / self.__value, -self.__unit)
#
# 	def __pow__(self, other) -> "Quantity":
# 		if isinstance(other, int | float):
# 			return Quantity(self.__value ** other, self.__unit ** other)
# 		else:
# 			return NotImplemented

# todo: prefix units

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
