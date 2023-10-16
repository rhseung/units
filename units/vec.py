from numpy import ndarray
from math import cos, sin, atan2
from typing import TypeAlias

class Vec[T]:
	def __init__(self, *args, r=None, theta=None):
		if len(args) == 1 and isinstance(args[0], VecLike):
			self.x, self.y = args[0]
		elif r is not None and theta is not None and isinstance(r, int | float) and isinstance(theta, float):
			self.r = r
			self.theta = theta
			self.x = r * cos(theta)
			self.y = r * sin(theta)
		else:
			self.x, self.y = args
			self.r = (self.x ** 2 + self.y ** 2) ** 0.5
			self.theta = atan2(self.y, self.x)
	
	@staticmethod
	def P(r, theta):
		return Vec(r=r, theta=theta)
	
	def __pos__(self):
		return self
	
	def __neg__(self):
		return Vec(-self.x, -self.y)
	
	def __add__(self, other):
		if isinstance(other, Vec):
			return Vec(self.x + other.x, self.y + other.y)
		elif isinstance(other, ndarray):
			if not (other.ndim == 1 and len(other) == 2 and other.dtype == T):
				raise TypeError
			return Vec(self.x + other[0], self.y + other[1])
		elif isinstance(other, int | float):
			return Vec(self.x + other, self.y + other)
		else:
			return NotImplemented
	
	def __radd__(self, other):
		return self.__add__(Vec(other))
	
	def __sub__(self, other):
		return self.__add__(-Vec(other))
	
	def __rsub__(self, other):
		return -(self.__add__(-Vec(other)))
	
	def __mul__(self, other):
		if isinstance(other, int | float):
			return Vec(self.x * other, self.y * other)
		else:
			return NotImplemented
	
	def __rmul__(self, other):
		return self.__mul__(other)
	
	def __truediv__(self, other):  # self / other
		return self.__mul__(1 / other)
	
	def __rtruediv__(self, other):  # other / self
		# return Vec(other / self.x, other / self.y)
		return 1 / self.__truediv__(other)
	
	def __iter__(self):
		yield self.x
		yield self.y
	
	def __repr__(self):
		return f"({self.x}, {self.y})"
	
	def __abs__(self):
		return (self.x ** 2 + self.y ** 2) ** 0.5
	
	__str__ = __repr__

Iterable: TypeAlias = tuple | list | ndarray
VecLike: TypeAlias = Vec | Iterable

__all__ = ['Vec', 'VecLike', 'Iterable']
