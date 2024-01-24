import pytest
from contextlib import contextmanager

from sortedcontainers import SortedDict

from ..unit import *
from ..unit.core import Units
from ..unit.functions import *

def test_expand():
    assert str(expand(ohm)) == 'V/A'

# test: 크기가 있는 expand

def test_si():
    assert str(si(ohm)) == 'kg⋅m^2/A^2⋅s^3'

def test_zip():
    assert type(Units(SortedDict({atm: 1}))) == Unit

def test_notzip():
    us = Units(SortedDict({atm: 1, m: 1}))
    assert type(us) == Units
