from dataclasses import dataclass
from .vec import Vec, Iterable
from math import pi

@dataclass
class Unit:
	kg: float = 0
	m: float = 0
	s: float = 0
	A: float = 0
	K: float = 0
	mol: float = 0
	cd: float = 0
	
	def __iter__(self):
		yield 'kg', self.kg
		yield 'm', self.m
		yield 's', self.s
		yield 'A', self.A
		yield 'K', self.K
		yield 'mol', self.mol
		yield 'cd', self.cd
	
	def __pos__(self):
		return self
	
	def __neg__(self):
		return Unit(
			-self.kg,
			-self.m,
			-self.s,
			-self.A,
			-self.K,
			-self.mol,
			-self.cd
		)
	
	def __add__(self, other):
		if isinstance(other, Unit):
			return Unit(
				self.kg + other.kg,
				self.m + other.m,
				self.s + other.s,
				self.A + other.A,
				self.K + other.K,
				self.mol + other.mol,
				self.cd + other.cd
			)
		else:
			return NotImplemented
	
	def __sub__(self, other):
		return self.__add__(-other)
	
	def __mul__(self, other):
		if isinstance(other, int | float):
			return Unit(
				self.kg * other,
				self.m * other,
				self.s * other,
				self.A * other,
				self.K * other,
				self.mol * other,
				self.cd * other
			)
		else:
			return NotImplemented
	
	def __rmul__(self, other):
		return self.__mul__(other)
	
	def __eq__(self, other):
		if isinstance(other, Unit):
			return all(_s == _o for _s, _o in zip(self, other))
		else:
			return NotImplemented
	
	def __repr__(self):
		ret = "Unit("
		for name, power in self:
			if power == 0:
				continue
			
			ret += f"{name}={power}, "
		ret = ret[:-2] + ")"
		
		return ret
	
	__str__ = __repr__

Unit.dimensionless = Unit()

class UnitError(Exception):
	pass

class Quantity:
	def __init__(self, value: float | Vec, dimension: Unit = Unit.dimensionless):
		self.value = value
		self.dimension = dimension
	
	def __str__(self):
		dot = '⋅'
		dimension_string = ["", ""]
		for name, power in self.dimension:
			if power == 0:
				continue
			elif abs(power) == 1:
				dimension_string[power < 0] += name + dot
			else:
				__str = str(int(abs(power)) if power.is_integer() else abs(power))
				dimension_string[power < 0] += name + '^' + __str + dot
		
		front, back = dimension_string
		front = front.strip(dot)
		back = back.strip(dot)
		
		return f"{self.value} {front}{"/" if back else ""}{back}".strip()
	
	def __repr__(self):
		return self.__str__()
	
	def __pos__(self):
		return self
	
	def __neg__(self):
		return Quantity(-self.value, self.dimension)
	
	def __eq__(self, other):
		if isinstance(other, Quantity):
			if self.value == other.value:
				return True if self.value == 0 else self.dimension == other.dimension
		elif isinstance(other, int | float):
			return self.value == 0 and other == 0
		else:
			return NotImplemented
	
	def __lt__(self, other):
		if isinstance(other, Quantity):
			if self.dimension == other.dimension:
				return self.value < other.value
			else:
				raise UnitError(f"Cannot compare {self.dimension} and {other.dimension}.")
		else:
			return self.value < other
	
	def __gt__(self, other):
		return other < self
	
	def __add__(self, other):
		if isinstance(other, Quantity):
			if self.dimension == other.dimension:
				return Quantity(self.value + other.value, self.dimension)
			else:
				raise UnitError(f"Cannot add {self.dimension} and {other.dimension}.")
		else:
			return Quantity(self.value + other, self.dimension)
	
	def __radd__(self, other):
		return self.__add__(other)
	
	def __sub__(self, other):
		return self.__add__(-other)
	
	def __rsub__(self, other):
		return -(self.__add__(-other))
	
	def __mul__(self, other):
		if isinstance(other, Quantity):
			ret = Quantity(self.value * other.value, self.dimension + other.dimension)
			return ret
		
		# todo: 무차원 상수를 어떻게 다뤄야 하지
		# if ret.dimension == Unit.dimensionless:
		# 	return ret.value
		# else:
		# 	return ret
		elif isinstance(other, tuple):
			return Quantity(self.value * Vec(other), self.dimension)
		elif isinstance(other, Iterable):
			for i in range(len(other)):
				other[i] *= self
			return other
		else:
			return Quantity(self.value * other, self.dimension)
	
	def __rmul__(self, other):
		return self.__mul__(other)
	
	def __truediv__(self, other):
		return self.__mul__(1 / other)
	
	def __rtruediv__(self, other):
		if isinstance(other, tuple):
			return Quantity(Vec(other) / self.value, -self.dimension)
		else:
			return Quantity(other / self.value, -self.dimension)
	
	def __pow__(self, other: int):
		if isinstance(other, int | float):
			return Quantity(self.value ** other, self.dimension * other)
		else:
			return NotImplemented
	
	def __abs__(self):
		return Quantity(abs(self.value), self.dimension)
	
	def __iter__(self):
		if self.is_vector():
			yield self.x
			yield self.y
		else:
			yield self
	
	def __int__(self):
		return int(self.value)
	
	def __float__(self):
		return float(self.value)
	
	def is_vector(self, unit: "Quantity" = None):
		return isinstance(self.value, Vec) and (unit is None or self.unit == unit.unit)
	
	def is_scalar(self, unit: "Quantity" = None):
		return not self.is_vector() and (unit is None or self.unit == unit.unit)
	
	def magnitude(self):
		return abs(self)
	
	@property
	def unit(self):
		return Quantity(1, self.dimension)
	
	@property
	def x(self):
		if self.is_vector():
			return Quantity(self.value.x, self.dimension)
		else:
			raise TypeError("This quantity is not a vector.")
	
	@property
	def y(self):
		if self.is_vector():
			return Quantity(self.value.y, self.dimension)
		else:
			raise TypeError("This quantity is not a vector.")
	
	@staticmethod
	def xy(x: "Quantity", y: "Quantity"):
		if x.unit != y.unit:
			raise UnitError(f"{x.unit} and {y.unit} are not same.")
		return (x.value, y.value) * x.unit

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

kg = Quantity(1, Unit(kg=1))
g = 1e-3 * kg
m = Quantity(1, Unit(m=1))
km = 1e3 * m
cm = 1e-2 * m
s = Quantity(1, Unit(s=1))
ms = 1e-3 * s
A = Quantity(1, Unit(A=1))
K = Quantity(1, Unit(K=1))
mol = Quantity(1, Unit(mol=1))
cd = Quantity(1, Unit(cd=1))
rad = Quantity(180 / pi, Unit.dimensionless)

__all__ = [
	"Unit", "UnitError", "Quantity", "prefixes",
	"kg", "m", "s", "A", "K", "mol", "cd",
	"g", "km", "cm", "ms", "rad",
]
