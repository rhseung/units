from .vector import *
from .utils import Counter
from abc import ABC, abstractmethod
from enum import Enum

def to_int_if_possible(value: float) -> int | float:
	return int(value) if value.is_integer() else value

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

# class Unit:
# 	presets: dict[str, "Unit"] = {}
#
# 	def __init__(self, *, kg=0., m=0., s=0., A=0., K=0., mol=0., cd=0., rad=0.):
# 		self.__data: dict[str, float] = {
# 			'kg': kg,
# 			'm': m,
# 			's': s,
# 			'A': A,
# 			'K': K,
# 			'mol': mol,
# 			'cd': cd,
# 			'rad': rad
# 		}
#
# 		# todo: __str은 생성 시 알아서 변경됨
# 		#  /s**-2 같은 건 s**2 가 되도록 생각하면서 구현하기
#
# 	@property
# 	def kg(self) -> float:
# 		return self.__data['kg']
#
# 	@property
# 	def m(self) -> float:
# 		return self.__data['m']
#
# 	@property
# 	def s(self) -> float:
# 		return self.__data['s']
#
# 	@property
# 	def A(self) -> float:
# 		return self.__data['A']
#
# 	@property
# 	def K(self) -> float:
# 		return self.__data['K']
#
# 	@property
# 	def mol(self) -> float:
# 		return self.__data['mol']
#
# 	@property
# 	def cd(self) -> float:
# 		return self.__data['cd']
#
# 	@property
# 	def rad(self) -> float:
# 		return self.__data['rad']
#
# 	def __bool__(self) -> bool:
# 		return any(self.__data.values())
#
# 	def _repr_latex_(self, get=False) -> str:
# 		_content = str(self.unit).replace('**', '^').replace('*', ' \\cdot ')
#
# 		if get:
# 			return _content
# 		else:
# 			return f"$\\mathrm{{ {_content} }}$"
#
# 	def __repr__(self) -> str:
# 		_mul = '*'  # '⋅'
# 		_pow = '**'  # '^'
#
# 		ret = ["", ""]
#
# 		for name, power in self.__data.items():
# 			# todo: 복합단위 표현하기
# 			if power == 0:
# 				continue
# 			elif abs(power) == 1:
# 				ret[power < 0] += f"{name}{_mul}"
# 			else:
# 				ret[power < 0] += f'{name}{_pow}{int(abs(power)) if power.is_integer() else abs(power)}{_mul}'
#
# 		front, back = ret
# 		front = front.strip(_mul)
# 		back = back.strip(_mul)
#
# 		return f"{front}{'/' if back else ''}{back}".strip()
#
# 	def __str__(self) -> str:
# 		return self.__repr__()
#
# 	def __iter__(self) -> iter:
# 		yield from self.__data.items()
#
# 	def __pos__(self) -> "Unit":
# 		return self
#
# 	def __neg__(self) -> "Unit":
# 		return Unit(**{name: -power for name, power in self})
#
# 	def __eq__(self, other) -> bool:
# 		if isinstance(other, Unit):
# 			return self.__data == other.__data
# 		else:
# 			return NotImplemented
#
# 	def __mul__(self, other) -> "Unit":
# 		if isinstance(other, Unit):
# 			return Unit(**{name: power + other.__data[name] for name, power in self})
# 		else:
# 			return NotImplemented
#
# 	def __rmul__(self, other) -> "Quantity | Unit":
# 		if isinstance(other, int | float):
# 			return Quantity(other, self)
# 		elif isinstance(other, VecLike):
# 			return Quantity(Vector(*other), self)
# 		else:
# 			return self.__mul__(other)
#
# 	def __truediv__(self, other) -> "Unit":
# 		return self.__mul__(-other)
#
# 	def __rtruediv__(self, other) -> "Quantity | Unit":
# 		if isinstance(other, int | float):
# 			return Quantity(other, -self)
# 		elif isinstance(other, VecLike):
# 			return Quantity(Vector(*other), -self)
# 		else:
# 			return NotImplemented
#
# 	def __divmod__(self, other):
# 		# todo: divmod 구현하기
# 		...
#
# 	def __pow__(self, other) -> "Unit":
# 		if isinstance(other, int | float):
# 			return Unit(**{name: power * float(other) for name, power in self})
# 		else:
# 			return NotImplemented

# todo: scale, dimensionless

# todo: UnitBase 만들고 Unit과 AbbreviateUnit은 UnitBase를 상속받게 하기 -> AbbreviateUnit은 인스턴스 추적 ㄴㄴ

class UnitBase(ABC):
	def __init__(self):
		self._scale: float = 1.
	
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
	def scale(self) -> float:
		return self._scale

class Unit(UnitBase):
	_instances = {}
	
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
		return f"Unit('{self._symbol}')"
	
	def represent(self) -> 'ComplexUnit | Unit':
		return self
	
	def si(self) -> 'ComplexUnit | Unit':
		return self
	
	@property
	def symbol(self):
		return self._symbol

class ComplexUnit(UnitBase):
	def __init__(self, elements: Counter[Unit] = None, scale: float = 1.):
		super().__init__()
		
		self._scale = float(scale)
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
			raise TypeError(f"CompositeUnit.__mul__: {type(other)}")
		
		if len(new_counter) == 1 and self._scale == 1.:
			return next(iter(new_counter))
		else:
			return ComplexUnit(new_counter, self._scale * other.scale)

	def __repr__(self):
		if self.is_dimensionless():
			text = 'dimensionless'
		else:
			arr = self._counter.map_to_list(lambda unit, power: f'{unit.symbol}**{to_int_if_possible(power)}')
			text = ' '.join(arr)
			
		return f"CompositeUnit('{'' if self._scale == 1. else f'{to_int_if_possible(self._scale):.0e}' + ' '}{text}')"

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

class AbbreviateUnit(Unit):
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
		return isinstance(other, AbbreviateUnit) and self._represent == other._represent

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

N = AbbreviateUnit('N', kg * m/s**2)
J = AbbreviateUnit('J', N * m)
Pa = AbbreviateUnit('Pa', N / m**2)
g = AbbreviateUnit('g', ComplexUnit(Counter({kg: 1}), 1e-3))

# for prefix in Prefix:
# 	for unit in [g, m, s, A, K, mol, cd, rad]:
# 		if unit == g and prefix == Prefix.k:
# 			continue
#
# 		symbol_name = prefix.name + unit.symbol
# 		print(symbol_name)
# 		globals()[symbol_name] = PrefixUnit(prefix, unit)

for prefix in Prefix:
	for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]:
		if prefix == Prefix.k and unit == g:
			continue

		symbol_name = (prefix.name if prefix != Prefix.µ else 'micro_') + unit.symbol
		globals()[symbol_name] = PrefixUnit(prefix, unit)

# Yg, Ym, Ys, YA, YK, Ymol, Ycd, Yrad, YN, YJ, YPa = [PrefixUnit(Prefix.Y, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# Zg, Zm, Zs, ZA, ZK, Zmol, Zcd, Zrad, ZN, ZJ, ZPa = [PrefixUnit(Prefix.Z, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# Eg, Em, Es, EA, EK, Emol, Ecd, Erad, EN, EJ, EPa = [PrefixUnit(Prefix.E, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# Pg, Pm, Ps, PA, PK, Pmol, Pcd, Prad, PN, PJ, PPa = [PrefixUnit(Prefix.P, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# Tg, Tm, Ts, TA, TK, Tmol, Tcd, Trad, TN, TJ, TPa = [PrefixUnit(Prefix.T, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# Gg, Gm, Gs, GA, GK, Gmol, Gcd, Grad, GN, GJ, GPa = [PrefixUnit(Prefix.G, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# Mg, Mm, Ms, MA, MK, Mmol, Mcd, Mrad, MN, MJ, MPa = [PrefixUnit(Prefix.M, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# km,     ks, kA, kK, kmol, kcd, krad, kN, kJ, kPa = [PrefixUnit(Prefix.k, unit) for unit in [m, s, A, K, mol, cd, rad, N, J, Pa]]
# hg, hm, hs, hA, hK, hmol, hcd, hrad, hN, hJ, hPa = [PrefixUnit(Prefix.h, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# dg, dm, ds, dA, dK, dmol, dcd, drad, dN, dJ, dPa = [PrefixUnit(Prefix.d, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# cg, cm, cs, cA, cK, cmol, ccd, crad, cN, cJ, cPa = [PrefixUnit(Prefix.c, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# mg, mm, ms, mA, mK, mmol, mcd, mrad, mN, mJ, mPa = [PrefixUnit(Prefix.m, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# micro_g, micro_m, micro_s, micro_A, micro_K, micro_mol, micro_cd, micro_rad, micro_N, micro_J, micro_Pa = [PrefixUnit(Prefix.µ, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# ng, nm, ns, nA, nK, nmol, ncd, nrad, nN, nJ, nPa = [PrefixUnit(Prefix.n, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# pg, pm, ps, pA, pK, pmol, pcd, prad, pN, pJ, pPa = [PrefixUnit(Prefix.p, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# fg, fm, fs, fA, fK, fmol, fcd, frad, fN, fJ, fPa = [PrefixUnit(Prefix.f, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# ag, am, atto_s, aA, aK, amol, acd, arad, aN, aJ, aPa = [PrefixUnit(Prefix.a, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# zg, zm, zs, zA, zK, zmol, zcd, zrad, zN, zJ, zPa = [PrefixUnit(Prefix.z, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]
# yg, ym, ys, yA, yK, ymol, ycd, yrad, yN, yJ, yPa = [PrefixUnit(Prefix.y, unit) for unit in [g, m, s, A, K, mol, cd, rad, N, J, Pa]]

__all__ = [
	'g', 'm', 's', 'A', 'K', 'mol', 'cd', 'rad',
	'N', 'J', 'Pa',
	'Yg', 'Ym', 'Ys', 'YA', 'YK', 'Ymol', 'Ycd', 'Yrad', 'YN', 'YJ', 'YPa',
	'Zg', 'Zm', 'Zs', 'ZA', 'ZK', 'Zmol', 'Zcd', 'Zrad', 'ZN', 'ZJ', 'ZPa',
	'Eg', 'Em', 'Es', 'EA', 'EK', 'Emol', 'Ecd', 'Erad', 'EN', 'EJ', 'EPa',
	'Pg', 'Pm', 'Ps', 'PA', 'PK', 'Pmol', 'Pcd', 'Prad', 'PN', 'PJ', 'PPa',
	'Tg', 'Tm', 'Ts', 'TA', 'TK', 'Tmol', 'Tcd', 'Trad', 'TN', 'TJ', 'TPa',
	'Gg', 'Gm', 'Gs', 'GA', 'GK', 'Gmol', 'Gcd', 'Grad', 'GN', 'GJ', 'GPa',
	'Mg', 'Mm', 'Ms', 'MA', 'MK', 'Mmol', 'Mcd', 'Mrad', 'MN', 'MJ', 'MPa',
	'kg', 'km', 'ks', 'kA', 'kK', 'kmol', 'kcd', 'krad', 'kN', 'kJ', 'kPa',
	'hg', 'hm', 'hs', 'hA', 'hK', 'hmol', 'hcd', 'hrad', 'hN', 'hJ', 'hPa',
	'dg', 'dm', 'ds', 'dA', 'dK', 'dmol', 'dcd', 'drad', 'dN', 'dJ', 'dPa',
	'cg', 'cm', 'cs', 'cA', 'cK', 'cmol', 'ccd', 'crad', 'cN', 'cJ', 'cPa',
	'mg', 'mm', 'ms', 'mA', 'mK', 'mmol', 'mcd', 'mrad', 'mN', 'mJ', 'mPa',
	'micro_g', 'micro_m', 'micro_s', 'micro_A', 'micro_K', 'micro_mol', 'micro_cd', 'micro_rad', 'micro_N', 'micro_J', 'micro_Pa',
	'ng', 'nm', 'ns', 'nA', 'nK', 'nmol', 'ncd', 'nrad', 'nN', 'nJ', 'nPa',
	'pg', 'pm', 'ps', 'pA', 'pK', 'pmol', 'pcd', 'prad', 'pN', 'pJ', 'pPa',
	'fg', 'fm', 'fs', 'fA', 'fK', 'fmol', 'fcd', 'frad', 'fN', 'fJ', 'fPa',
	'ag', 'am', 'as', 'aA', 'aK', 'amol', 'acd', 'arad', 'aN', 'aJ', 'aPa',
	'zg', 'zm', 'zs', 'zA', 'zK', 'zmol', 'zcd', 'zrad', 'zN', 'zJ', 'zPa',
	'yg', 'ym', 'ys', 'yA', 'yK', 'ymol', 'ycd', 'yrad', 'yN', 'yJ', 'yPa',
]
