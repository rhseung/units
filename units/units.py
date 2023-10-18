from .vector import *
from math import pi

class UnitError(Exception):
	pass

class Unit:
	__preset: dict[str, "Unit"] = {}

	def __init__(self, *, kg=0., m=0., s=0., A=0., K=0., mol=0., cd=0., rad=0.):
		self.__data: dict[str, float] = {
			'kg': kg,
			'm': m,
			's': s,
			'A': A,
			'K': K,
			'mol': mol,
			'cd': cd,
			'rad': rad
		}

	@property
	def kg(self) -> float:
		return self.__data['kg']

	@property
	def m(self) -> float:
		return self.__data['m']
	
	@property
	def s(self) -> float:
		return self.__data['s']
	
	@property
	def A(self) -> float:
		return self.__data['A']
	
	@property
	def K(self) -> float:
		return self.__data['K']
	
	@property
	def mol(self) -> float:
		return self.__data['mol']
	
	@property
	def cd(self) -> float:
		return self.__data['cd']
	
	@property
	def rad(self) -> float:
		return self.__data['rad']
	
	def __bool__(self) -> bool:
		return any(self.__data.values())

	def _repr_latex_(self, get=False) -> str:
		_content = str(self.unit).replace('**', '^').replace('*', ' \\cdot ')
		
		if get:
			return _content
		else:
			return f"$\\mathrm{{ {_content} }}$"

	def __repr__(self) -> str:
		_mul = '*'  # '⋅'
		_pow = '**'  # '^'
		
		ret = ["", ""]
		
		for name, power in self.__data.items():
			# todo: 복합단위 표현하기
			if power == 0:
				continue
			elif abs(power) == 1:
				ret[power < 0] += f"{name}{_mul}"
			else:
				ret[power < 0] += f'{name}{_pow}{int(abs(power)) if power.is_integer() else abs(power)}{_mul}'
		
		front, back = ret
		front = front.strip(_mul)
		back = back.strip(_mul)
		
		return f"{front}{'/' if back else ''}{back}".strip()

	def __str__(self) -> str:
		return self.__repr__()

	def __iter__(self) -> iter:
		yield from self.__data.items()

	def __pos__(self) -> "Unit":
		return self
	
	def __neg__(self) -> "Unit":
		return Unit(**{name: -power for name, power in self})

	def __eq__(self, other) -> bool:
		if isinstance(other, Unit):
			return self.__data == other.__data
		else:
			return NotImplemented

	def __mul__(self, other) -> "Unit":
		if isinstance(other, Unit):
			return Unit(**{name: power + other.__data[name] for name, power in self})
		else:
			return NotImplemented

	def __rmul__(self, other) -> "Quantity | Unit":
		if isinstance(other, int | float):
			return Quantity(other, self)
		elif isinstance(other, VecLike):
			return Quantity(Vector(*other), self)
		else:
			return self.__mul__(other)

	def __truediv__(self, other) -> "Unit":
		return self.__mul__(-other)

	def __rtruediv__(self, other) -> "Quantity | Unit":
		if isinstance(other, int | float):
			return Quantity(other, -self)
		elif isinstance(other, VecLike):
			return Quantity(Vector(*other), -self)
		else:
			return NotImplemented

	def __pow__(self, other) -> "Unit":
		if isinstance(other, int | float):
			return Unit(**{name: power * other for name, power in self})
		else:
			return NotImplemented

class Quantity:	
	def __init__(self, value: float | Vector, unit: Unit):
		self.__value = value
		self.__unit = unit

	def is_vector(self, unit: Unit = None) -> bool:
		return isinstance(self.__value, Vector) and (unit is None or self.unit == unit)
	
	def is_scalar(self, unit: Unit = None) -> bool:
		return not self.is_vector() and (unit is None or self.unit == unit)

	@property
	def value(self) -> float | Vector:
		return self.__value
	
	@property
	def unit(self) -> Unit:
		return self.__unit
	
	@property
	def e(self):
		return tuple(v * self.unit for v in self.__value)
	
	def __iter__(self):
		yield self.e
	
	def __bool__(self):
		return bool(self.__value)
	
	def __len__(self):
		return len(self.__value)
	
	def __abs__(self):
		return Quantity(abs(self.__value), self.__unit)

	def _repr_latex_(self):
		# todo: self.value 가 벡터일 때 latex 표현하기
		# fixme: this is a temporary fix
		return f"$\mathrm{{ {self.value if self.is_scalar() else self.value._repr_latex_(True)}~~{self.unit._repr_latex_(True)} }}$"

	def __format__(self, format_spec):
		return f"{self.__value:{format_spec}} [{self.__unit}]"
	
	def __repr__(self) -> str:
		return self.__format__('')
	
	def __str__(self) -> str:
		return self.__repr__()
	
	def __pos__(self) -> "Quantity":
		return self
	
	def __neg__(self) -> "Quantity":
		return Quantity(-self.__value, self.__unit)
	
	def __eq__(self, other) -> bool:
		if isinstance(other, Quantity):
			if self.__value == other.__value:
				return True if self.__value == 0 else self.__unit == other.__unit
		elif isinstance(other, int | float):
			return self.__value == 0 and other == 0
		else:
			return NotImplemented
		
	def __lt__(self, other) -> bool:
		if isinstance(other, Quantity):
			if self.__unit == other.__unit:
				return self.__value < other.__value
			else:
				raise UnitError(f"Cannot compare {self.__unit} and {other.__unit}.")
		else:
			return self.__value < other
		
	def __gt__(self, other) -> bool:
		return other < self
		
	def __le__(self, other) -> bool:
		return self == other or self < other
	
	def __ge__(self, other) -> bool:
		return self == other or self > other
	
	def __add__(self, other) -> "Quantity":
		if isinstance(other, Quantity):
			if self.__unit == other.__unit:
				return Quantity(self.__value + other.__value, self.__unit)
			else:
				raise UnitError(f"Cannot add {self.__unit} and {other.__unit}.")
		else:
			return Quantity(self.__value + other, self.__unit)

	def __radd__(self, other) -> "Quantity":
		return self.__add__(other)

	def __sub__(self, other) -> "Quantity":
		return self.__add__(-other)

	def __rsub__(self, other) -> "Quantity":
		return -(self.__add__(-other))
	
	def __mul__(self, other) -> "Quantity":
		if isinstance(other, Quantity):
			return Quantity(self.__value * other.__value, self.__unit * other.__unit)
		elif isinstance(other, VecLike):
			return Quantity(self.__value * Vector(*other), self.__unit)
		elif isinstance(other, int | float):
			return Quantity(self.__value * other, self.__unit)
		elif isinstance(other, Iterable):
			for i in range(len(other)):
				other[i] *= self
			return other
		else:
			return NotImplemented

	def __rmul__(self, other) -> "Quantity":
		return self.__mul__(other)

	def __truediv__(self, other) -> "Quantity":
		return self.__mul__(1 / other)
	
	def __rtruediv__(self, other) -> "Quantity":
		if isinstance(other, VecLike):
			return Quantity(Vector(*other) / self.__value, -self.__unit)
		elif isinstance(other, int | float):
			return Quantity(other / self.__value, -self.__unit)
		else:
			return Quantity(other / self.__value, -self.__unit)

	def __pow__(self, other) -> "Quantity":
		if isinstance(other, int | float):
			return Quantity(self.__value ** other, self.__unit ** other)
		else:
			return NotImplemented

prefixes = {
	'Y': 1e24,
	'Z': 1e21,
	'E': 1e18,
	'P': 1e15,
	'T': 1e12,
	'G': 1e9,
	'M': 1e6,
	'k': 1e3,
	'h': 1e2,
	'd': 1e-1,
	'c': 1e-2,
	'm': 1e-3,
	'µ': 1e-6,
	'n': 1e-9,
	'p': 1e-12,
	'f': 1e-15,
	'a': 1e-18,
	'z': 1e-21,
	'y': 1e-24
}

kg = Unit(kg=1)
m = Unit(m=1)
s = Unit(s=1)
A = Unit(A=1)
K = Unit(K=1)
mol = Unit(mol=1)
cd = Unit(cd=1)
rad = Unit(rad=1)

Unit.presets['N'] = kg * m/s**2
Unit.presets['J'] = (kg * m/s**2) * m
Unit.presets['Pa'] = (kg * m/s**2) / m**2
# todo: 복합 단위 더 추가하기

__all__ = [
	"Unit", "UnitError", "Quantity", "prefixes",
	"kg", "m", "s", "A", "K", "mol", "cd", "rad"
]
