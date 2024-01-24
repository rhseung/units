import pytest
from contextlib import contextmanager

from ..unit import *
from ..unit.core import Unit
from ..unit.functions import *

def test_dimless():
    km = Unit('km', m, 1000)
    assert km == kilo*m
