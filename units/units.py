from .vector import *
from math import pi

class UnitError(Exception):
	pass

class Unit:
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

	def __repr__(self) -> str:
		_mul = '*'  # '⋅'
		_pow = '**'  # '^'
		
		ret = ["", ""]
		
		for name, power in self.__data.items():
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
	presets: dict[str, Unit] = {}
	
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

# class Quantity:
# 	def __init__(self, value: float | Vector, dimension: Unit = Unit.dimensionless):
# 		self.value = value
# 		self.dimension = dimension
#
# 	def __str__(self):
# 		dot = '⋅'
# 		dimension_string = ["", ""]
# 		for name, power in self.dimension:
# 			if power == 0:
# 				continue
# 			elif abs(power) == 1:
# 				dimension_string[power < 0] += name + dot
# 			else:
# 				__str = str(int(abs(power)) if power.is_integer() else abs(power))
# 				dimension_string[power < 0] += name + '^' + __str + dot
#
# 		front, back = dimension_string
# 		front = front.strip(dot)
# 		back = back.strip(dot)
#
# 		return f"{self.value} {front}{'/' if back else ''}{back}".strip()
#
# 	def __repr__(self):
# 		return self.__str__()
#
# 	def __pos__(self):
# 		return self
#
# 	def __neg__(self):
# 		return Quantity(-self.value, self.dimension)
#
# 	def __eq__(self, other):
# 		if isinstance(other, Quantity):
# 			if self.value == other.value:
# 				return True if self.value == 0 else self.dimension == other.dimension
# 		elif isinstance(other, int | float):
# 			return self.value == 0 and other == 0
# 		else:
# 			return NotImplemented
#
# 	def __lt__(self, other):
# 		if isinstance(other, Quantity):
# 			if self.dimension == other.dimension:
# 				return self.value < other.value
# 			else:
# 				raise UnitError(f"Cannot compare {self.dimension} and {other.dimension}.")
# 		else:
# 			return self.value < other
#
# 	def __gt__(self, other):
# 		return other < self
#
# 	def __add__(self, other):
# 		if isinstance(other, Quantity):
# 			if self.dimension == other.dimension:
# 				return Quantity(self.value + other.value, self.dimension)
# 			else:
# 				raise UnitError(f"Cannot add {self.dimension} and {other.dimension}.")
# 		else:
# 			return Quantity(self.value + other, self.dimension)
#
# 	def __radd__(self, other):
# 		return self.__add__(other)
#
# 	def __sub__(self, other):
# 		return self.__add__(-other)
#
# 	def __rsub__(self, other):
# 		return -(self.__add__(-other))
#
# 	def __mul__(self, other):
# 		if isinstance(other, Quantity):
# 			ret = Quantity(self.value * other.value, self.dimension + other.dimension)
# 			return ret
#
# 		# todo: 무차원 상수를 어떻게 다뤄야 하지
# 		# if ret.dimension == Unit.dimensionless:
# 		# 	return ret.value
# 		# else:
# 		# 	return ret
# 		elif isinstance(other, tuple):
# 			return Quantity(self.value * Vector(other), self.dimension)
# 		elif isinstance(other, Iterable):
# 			for i in range(len(other)):
# 				other[i] *= self
# 			return other
# 		else:
# 			return Quantity(self.value * other, self.dimension)
#
# 	def __rmul__(self, other):
# 		return self.__mul__(other)
#
# 	def __truediv__(self, other):
# 		return self.__mul__(1 / other)
#
# 	def __rtruediv__(self, other):
# 		if isinstance(other, tuple):
# 			return Quantity(Vector(other) / self.value, -self.dimension)
# 		else:
# 			return Quantity(other / self.value, -self.dimension)
#
# 	def __pow__(self, other: int):
# 		if isinstance(other, int | float):
# 			return Quantity(self.value ** other, self.dimension * other)
# 		else:
# 			return NotImplemented
#
# 	def __abs__(self):
# 		return Quantity(abs(self.value), self.dimension)
#
# 	def __iter__(self):
# 		if self.is_vector():
# 			yield self.x
# 			yield self.y
# 		else:
# 			yield self
#
# 	def __int__(self):
# 		return int(self.value)
#
# 	def __float__(self):
# 		return float(self.value)
#
# 	def is_vector(self, unit: "Quantity" = None):
# 		return isinstance(self.value, Vector) and (unit is None or self.unit == unit.unit)
#
# 	def is_scalar(self, unit: "Quantity" = None):
# 		return not self.is_vector() and (unit is None or self.unit == unit.unit)
#
# 	def magnitude(self):
# 		return abs(self)
#
# 	@property
# 	def unit(self):
# 		return Quantity(1, self.dimension)
#
# 	@property
# 	def x(self):
# 		if self.is_vector():
# 			return Quantity(self.value.x, self.dimension)
# 		else:
# 			raise TypeError("This quantity is not a vector.")
#
# 	@property
# 	def y(self):
# 		if self.is_vector():
# 			return Quantity(self.value.y, self.dimension)
# 		else:
# 			raise TypeError("This quantity is not a vector.")
#
# 	@staticmethod
# 	def xy(x: "Quantity", y: "Quantity"):
# 		if x.unit != y.unit:
# 			raise UnitError(f"{x.unit} and {y.unit} are not same.")
# 		return (x.value, y.value) * x.unit

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

Quantity.presets['N'] = kg * m/s**2
Quantity.presets['J'] = (kg * m/s**2) * m
Quantity.presets['Pa'] = (kg * m/s**2) / m**2

__all__ = [
	"Unit", "UnitError", "Quantity", "prefixes",
	"kg", "m", "s", "A", "K", "mol", "cd", "rad"
]
