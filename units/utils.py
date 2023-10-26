__all__ = ['Counter']

from collections import UserDict
from typing import TypeVar, Callable, Generic, List, Tuple, Any, Dict

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
			return NotImplemented

	def __sub__(self, other) -> "Counter":
		if isinstance(other, Counter):
			return self.__add__(-other)
		elif isinstance(other, dict):
			return self - Counter(raw_dict=other)
		else:
			return NotImplemented
