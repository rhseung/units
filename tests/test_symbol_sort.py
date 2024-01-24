import pytest
from contextlib import contextmanager

from ..unit import *
from ..unit.core import Unit
from ..unit.functions import *

a = Unit('a', m)
b = Unit('b', m**2)
A = Unit('A', m**3)
B = Unit('B', m**4)
aa = Unit('aa', s)
ab = Unit('ab', s**2)
bb = Unit('bb', s**3)
AA = Unit('AA', s**4)
BB = Unit('BB', s**5)

def test_sort():
    assert str(a * b * A * B * aa * ab * bb * AA * BB) == 'AA⋅BB⋅aa⋅ab⋅bb⋅A⋅B⋅a⋅b'
