__all__ = ['to_int_if_possible', 'Counter', 'DimensionError', 'Vector', 'VecLike']

from math import cos, sin, atan2
from collections import UserDict
from typing import TypeVar, TypeAlias, Callable, Tuple, Dict

ValueType: TypeAlias = int | float
VecLike: TypeAlias = tuple

def to_int_if_possible(value: ValueType) -> ValueType:
	if isinstance(value, int):
		return value
	else:
		return int(value) if value.is_integer() else value

T = TypeVar('T')
class Counter(UserDict[T, float]):
	def __init__(self, *args: Tuple[T, float], raw_dict: Dict[T, float] = None):
		if raw_dict is not None:
			super().__init__(raw_dict)
		else:
			super().__init__((key, float(value)) for key, value in args if value != 0.)

	def map(self, f: Callable[[float], float]) -> "Counter":
		return Counter(*((key, f(value)) for key, value in self.items()))

	def map_to_list(self, f: Callable[[T, float], any]) -> list[any]:
		return [f(key, value) for key, value in self.items()]

	def __pos__(self) -> "Counter":
		return self.copy()

	def __neg__(self) -> "Counter":
		return self.map(lambda x: -x)

	def __getitem__(self, item) -> float:
		if item in self:
			return super().__getitem__(item)
		else:
			return 0.

	def __add__(self, other) -> "Counter":
		if isinstance(other, Counter):
			return Counter(*((key, self.get(key, 0.) + other.get(key, 0.)) for key in set(self) | set(other)))
		elif isinstance(other, dict):
			return self + Counter(raw_dict=other)
		else:
			raise TypeError(type(other))

	def __sub__(self, other) -> "Counter":
		if isinstance(other, Counter):
			return self.__add__(-other)
		elif isinstance(other, dict):
			return self - Counter(raw_dict=other)
		else:
			raise TypeError(type(other))

class DimensionError(Exception):
	def __init__(self, expected: int | str, input: int | str, subject: str = "self"):
		super().__init__(
			f"DimensionError: `{subject}` expected {str(expected) + 'd' if isinstance(expected, int) else expected}, but input is {str(input) + 'd' if isinstance(input, int) else input}")

class Vector:
	def __init__(self, *args: ValueType, raw_vector: "Vector" = None):
		self._args: Tuple = None

		if raw_vector is not None:
			self._args = raw_vector._args
		elif any(not isinstance(arg, ValueType) for arg in args):
			raise TypeError(f"Vector.__init__: {args}")
		else:
			self._args = tuple(map(to_int_if_possible, args))

	@staticmethod
	def polar(r, theta, phi=None) -> "Vector":
		if phi is None:
			return Vector(r * cos(theta), r * sin(theta))
		else:
			return Vector(r * sin(theta) * cos(phi), r ** sin(theta) * sin(phi), r * cos(theta))

	def _repr_latex_(self, get=False) -> str:
		if get:
			return r'(' + r',\,'.join(map(lambda x: str(to_int_if_possible(x)), self._args)) + r')'
		else:
			return r"$\begin{pmatrix} " + r'\\'.join(
				map(lambda x: str(to_int_if_possible(x)), self._args)) + r" \end{pmatrix}$"

	@property
	def dim(self) -> int:
		return len(self._args)

	@property
	def e(self) -> tuple:
		return self._args

	@property
	def i(self) -> float:
		if self.dim >= 1:
			return self._args[0]
		else:
			raise DimensionError(expected='>= 1d', input=self.dim)

	@property
	def j(self) -> float:
		if self.dim >= 2:
			return self._args[1]
		else:
			raise DimensionError(expected='>= 2d', input=self.dim)

	@property
	def k(self) -> float:
		if self.dim >= 3:
			return self._args[2]
		else:
			raise DimensionError(expected='>= 3d', input=self.dim)

	@property
	def r(self) -> float:
		return sum(arg ** 2 for arg in self._args) ** 0.5

	@property
	def theta(self) -> float:
		if self.dim == 2:
			return atan2(self.j, self.i)
		elif self.dim == 3:
			return atan2((self.i ** 2 + self.j ** 2) ** 0.5, self.k)
		else:
			raise DimensionError(expected=2, input=self.dim)

	@property
	def phi(self) -> float:
		if self.dim == 3:
			return atan2(self.j, self.i)
		else:
			raise DimensionError(expected=3, input=self.dim)

	def __abs__(self) -> float:
		return self.r

	def __len__(self) -> int:
		return self.dim

	def __pos__(self) -> "Vector":
		return self

	def __neg__(self) -> "Vector":
		return Vector(*[-arg for arg in self._args])

	def __add__(self, other) -> "Vector":
		if isinstance(other, Vector):
			if self.dim != other.dim:
				raise DimensionError(subject='other', expected=self.dim, input=other.dim)

			return Vector(*[a + b for a, b in zip(self._args, other._args)])
		elif isinstance(other, VecLike):
			_other = Vector(*other)
			return self + _other
		else:
			raise TypeError(type(other))

	def __radd__(self, other) -> "Vector":
		return self + other

	def __sub__(self, other) -> "Vector":
		return self + (-other)

	def __rsub__(self, other) -> "Vector":
		return -(self - other)

	def __mul__(self, other) -> "Vector | float":
		if isinstance(other, Vector):
			if self.dim != other.dim:
				raise DimensionError(subject='other', expected=self.dim, input=other.dim)

			return sum(a * b for a, b in zip(self._args, other._args))
		elif isinstance(other, VecLike):
			_other = Vector(*other)
			return self * _other
		elif isinstance(other, ValueType):		# 스칼라 곱
			return Vector(*((v * other) for v in self))
		else:
			raise TypeError(type(other))

	def __rmul__(self, other) -> "Vector | float":
		return self * other

	def __matmul__(self, other) -> "Vector":
		if isinstance(other, Vector):
			if self.dim != other.dim:
				raise DimensionError(subject='other', expected=self.dim, input=other.dim)

			if self.dim == 3:
				return Vector(
					other.k * self.j - other.j * self.k,
					other.i * self.k - other.k * self.i,
					other.j * self.i - other.i * self.j
				)
			else:
				raise DimensionError(expected=3, input=self.dim)
		elif isinstance(other, VecLike):
			_other = Vector(*other)
			return self @ _other
		else:
			raise TypeError(type(other))

	def __rmatmul__(self, other) -> "Vector":
		return self @ other

	def __truediv__(self, other) -> "Vector":
		return self * other ** -1

	def __repr__(self) -> str:
		return f"{self}"

	def __str__(self) -> str:
		return self.__repr__()

	def __format__(self, format_spec) -> str:
		if format_spec.endswith('p'):  # polar coordinates
			if not (2 <= self.dim <= 3):
				raise DimensionError(expected='2d or 3d', input=self.dim)

			options = '{:' + format_spec[:-1] + '}'
			text = [options.format(self.r), options.format(self.theta)]

			if self.dim == 3:
				text.append(options.format(self.phi))
				return f"({text[0]}, {text[1]}, {text[2]})"
			else:
				return f"({text[0]}, {text[1]})"
		elif format_spec.endswith('u'):  # unit vector
			options = '{:' + format_spec[:-1] + '}'
			text = [options.format(_e) for _e in self._args]

			if self.dim <= 3:
				ret = ' + '.join(f"{txt} {'ijk'[i]}" for i, txt in enumerate(text))
			else:
				ret = ' + '.join(f"{txt} e_{i}" for i, txt in enumerate(text))

			return '(' + ret + ')'
		else:
			ret = f"{' '.join(map(lambda x: ('{:' + format_spec + '}').format(x), self._args))}"

			return '(' + ret + ')'

	def __iter__(self) -> iter:
		return iter(self._args)

	def __reversed__(self) -> "Vector":
		return Vector(*reversed(self._args))

	def __eq__(self, other) -> bool:
		if isinstance(other, Vector):
			if self.dim != other.dim:
				raise DimensionError(subject='other', expected=self.dim, input=other.dim)

			return self._args == other._args
		elif isinstance(other, tuple):
			_other = Vector(*other)
			return self == _other
		else:
			raise TypeError(type(other))

	def __lt__(self, other) -> bool:
		if isinstance(other, Vector):
			if self.dim != other.dim:
				raise DimensionError(subject='other', expected=self.dim, input=other.dim)

			return abs(self) < abs(other)
		elif isinstance(other, tuple):
			_other = Vector(*other)
			return self < _other
		else:
			raise TypeError(type(other))

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
		return bool(abs(self))

	def __complex__(self) -> complex:
		if self.dim == 2:
			return complex(self.i, self.j)
		else:
			raise DimensionError(expected=2, input=self.dim)
