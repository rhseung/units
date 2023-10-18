from numpy import ndarray
from math import cos, sin, atan2
from typing import TypeAlias

class DimensionError(Exception):
	pass

class Vector:
	def __init__(self, *args: int | float):
		if len(args) == 1 and isinstance(args[0], Vector):
			self.__args = args[0].__args
		if any(not isinstance(arg, int | float) for arg in args):
			raise TypeError(f"Vector.__init__: {args}")
		
		self.__args = args
	
	@staticmethod
	def polar(r, theta, phi=None) -> "Vector":
		if phi is None:
			return Vector(r*cos(theta), r*sin(theta))
		else:
			return Vector(r*sin(theta)*cos(phi), r**sin(theta)*sin(phi), r*cos(theta))
	
	def _repr_latex_(self) -> str:
		return f"$\\begin{{bmatrix}}{self.i}\\\\{self.j}\\end{{bmatrix}}$"
	
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
		return sum(arg ** 2 for arg in self.__args) ** 0.5
	
	@property
	def theta(self) -> float:
		if self.dim == 2:
			return atan2(self.j, self.i)
		elif self.dim == 3:
			return atan2((self.i**2 + self.j**2) ** 0.5, self.k)
		else:
			raise DimensionError(f"input: {len(self)}, expected: 2")
	
	@property
	def phi(self) -> float:
		if self.dim == 3:
			return atan2(self.j, self.i)
		else:
			raise DimensionError(f"input: {len(self)}, expected: 3")
	
	def __abs__(self) -> float:
		return self.r
	
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
			if self.dim not in (2, 3):
				raise DimensionError(f"Vector.__format__: {self.dim}")
			
			options = '{:' + format_spec[:-1] + '}'
			text = [options.format(self.r), options.format(self.theta)]
			
			if self.dim == 3:
				text.append(options.format(self.phi))
				return f"({text[0]}, {text[1]}, {text[2]})"
			else:
				return f"({text[0]}, {text[1]})"
		elif format_spec.endswith('u'):  # unit vector
			options = '{:' + format_spec[:-1] + '}'
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
		
		return abs(self) < abs(other)

	def __le__(self, other):
		return self == other or self < other

	def __gt__(self, other):
		return not (self <= other)

	def __ge__(self, other):
		return not (self < other)

	def __int__(self) -> int:
		return int(abs(self))
	
	def __float__(self) -> float:
		return float(abs(self))
	
	def __bool__(self):
		return bool(self.r)
	
	def __complex__(self) -> complex:
		if self.dim == 2:
			return complex(self.i, self.j)
		else:
			raise DimensionError(f"input: {len(self)}, expected: 2")

Iterable: TypeAlias = tuple | list | ndarray
VecLike: TypeAlias = Vector | tuple

__all__ = ['Vector', 'VecLike', 'Iterable']