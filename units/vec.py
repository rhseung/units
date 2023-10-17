from numpy import ndarray
from math import cos, sin, atan2
from typing import TypeAlias

class DimensionError(Exception):
	pass

class Vector:
	def __init__(self, *args: int | float):
		self.__args = args
	
	@staticmethod
	def polar(r, theta) -> "Vector":
		return Vector(r * cos(theta), r * sin(theta))
	
	def magnitude(self) -> float:
		return sum(arg ** 2 for arg in self.__args) ** 0.5
	
	def angle(self) -> float:
		if len(self) == 2:
			return atan2(self.j, self.i)
		else:
			raise DimensionError(f"input: {len(self)}, expected: 2")
		
	def _repr_latex_(self) -> str:
		# todo: 어떻게 하는 거임
		...
	
	@property
	def dim(self) -> int:
		return len(self.__args)
	
	@property
	def e(self) -> tuple:
		return self.__args
	
	@property
	def i(self) -> float:
		return self.__args[0]
	
	@property
	def j(self) -> float:
		return self.__args[1]
	
	@property
	def k(self) -> float:
		return self.__args[2]
	
	@property
	def r(self) -> float:
		return self.magnitude()
	
	@property
	def theta(self) -> float:
		return self.angle()
	
	def __abs__(self) -> float:
		return self.magnitude()
	
	def __len__(self) -> int:
		return self.dim
	
	def __pos__(self) -> "Vector":
		return self
	
	def __neg__(self) -> "Vector":
		return Vector(*[-arg for arg in self.__args])
	
	def __add__(self, other) -> "Vector":
		if isinstance(other, Vector):
			if len(self) != len(other):
				raise ValueError(f"Vector.__add__: {len(self)} != {len(other)}")
			
			return Vector(*[a + b for a, b in zip(self.__args, other.__args)])
		elif isinstance(other, Iterable):
			if len(self) != len(other):
				raise ValueError(f"Vector.__add__: {len(self)} != {len(other)}")
			
			return Vector(*[a + b for a, b in zip(self.__args, other)])
		else:
			return NotImplemented
	
	def __radd__(self, other) -> "Vector":
		return self.__add__(other)
	
	def __sub__(self, other) -> "Vector":
		return self.__add__(-other)
	
	def __rsub__(self, other) -> "Vector":
		return -(self.__add__(-other))
	
	def __matmul__(self, other) -> float:
		if isinstance(other, Vector):
			if len(self) != len(other):
				raise ValueError(f"Vector.__matmul__: {len(self)} != {len(other)}")
			
			return sum(a * b for a, b in zip(self.__args, other.__args))
		else:
			return NotImplemented
	
	def __rmatmul__(self, other) -> float:
		return self.__matmul__(other)
	
	def __mul__(self, other) -> "Vector":
		if isinstance(other, Vector):
			if self.dim != other.dim:
				raise ValueError(f"Vector.__mul__: {self.dim} != {other.dim}")
			
			if self.dim == 3:
				return Vector(
					other.k * self.j - other.j * self.k,
					other.i * self.k - other.k * self.i,
					other.j * self.i - other.i * self.j
				)
			else:
				raise DimensionError(f"Vector.__mul__: {self.dim}")
		else:
			return Vector(*[a * other for a in self.__args])
		
	def __rmul__(self, other) -> "Vector":
		return -self.__mul__(other)
	
	def __truediv__(self, other) -> "Vector":
		if isinstance(other, Iterable | VecLike):
			raise TypeError(f"Vector.__truediv__: {type(other)}")
			
		return self.__mul__(1 / other)
	
	def __repr__(self) -> str:
		return f"{self}"
	
	def __str__(self) -> str:
		return self.__repr__()
	
	def __format__(self, format_spec) -> str:
		if format_spec.endswith('p'):  # polar coordinates
			if self.dim != 2:
				raise DimensionError(f"Vector.__format__: {self.dim}")
			
			options = '{:' + format_spec[:-1] + 'f}'
			text = [options.format(self.magnitude()), options.format(self.angle())]
			
			return f"({text[0]}, {text[1]} rad)"
		elif format_spec.endswith('u'):  # unit vector
			options = '{:' + format_spec[:-1] + 'f}'
			text = [options.format(_e) for _e in self.__args]
			
			subscripts = '₀₁₂₃₄₅₆₇₈₉'
			ret = ' + '.join(f"{txt} e{''.join(subscripts[int(c)] for c in str(i))}" for i, txt in enumerate(text))
			
			return '(' + ret + ')'
		else:
			ret = f"{', '.join(map(lambda x: ('{:' + format_spec + '}').format(x), self.__args))}"
			
			return '(' + ret + ')'
	
	def __iter__(self) -> iter:
		return iter(self.__args)

	def __reversed__(self) -> "Vector":
		return Vector(*reversed(self.__args))

	def __eq__(self, other) -> bool:
		if not isinstance(other, Vector):
			return False
		else:
			return self.__args == other.__args

	def __lt__(self, other) -> bool:
		if not isinstance(other, Vector):
			raise TypeError
		
		if self.dim != other.dim:
			raise DimensionError
		
		return self.magnitude() < other.magnitude()

	def __le__(self, other):
		return self == other or self < other

	def __gt__(self, other):
		return not (self <= other)

	def __ge__(self, other):
		return not (self < other)

	def __int__(self) -> int:
		return int(self.magnitude())
	
	def __float__(self) -> float:
		return float(self.magnitude())
	
	def __complex__(self) -> complex:
		if self.dim == 2:
			return complex(self.i, self.j)
		else:
			raise DimensionError(f"input: {len(self)}, expected: 2")

Iterable: TypeAlias = tuple | list | ndarray
VecLike: TypeAlias = Vector | tuple

__all__ = ['Vector', 'VecLike', 'Iterable']
