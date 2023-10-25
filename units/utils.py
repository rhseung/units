__all__ = ['Counter']

from collections import UserDict
from typing import TypeVar, Callable
from functools import cmp_to_key

T = TypeVar('T')
class Counter(UserDict[T, float]):
	def __init__(self, data: dict[T, float]):
		super().__init__({key: float(value) for key, value in data.items() if value != 0.})
	
	def map(self, f: Callable[[float], float]) -> "Counter":
		return Counter({key: f(value) for key, value in self.items()})
	
	def map_to_list(self, f: Callable[[T, float], any]) -> list[any]:
		return [f(key, value) for key, value in self.items()]

	def items(self):
		# A-Z-a-z
		def compare(a, b):
			a, b = a[0], b[0]		# (symbol, value) -> symbol

			if a.symbol.isupper() and b.symbol.isupper():
				return a.symbol < b.symbol
			elif a.symbol.islower() and b.symbol.islower():
				return a.symbol < b.symbol
			else:
				return a.symbol.isupper()

		return sorted(super().items(), key=cmp_to_key(compare), reverse=True)

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
			return Counter({key: self.get(key, 0.) + other.get(key, 0.) for key in set(self) | set(other)})
		elif isinstance(other, dict):
			return self + Counter(other)
		else:
			return NotImplemented

	def __sub__(self, other) -> "Counter":
		if isinstance(other, Counter):
			return self.__add__(-other)
		elif isinstance(other, dict):
			return self - Counter(other)
		else:
			return NotImplemented
