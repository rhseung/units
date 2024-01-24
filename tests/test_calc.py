import pytest
from contextlib import contextmanager

from ..unit import *
from ..unit.core import Unit
from ..unit.functions import *

BU = m
U = T
US = K**2

@contextmanager
def not_raises():
    try:
        yield
    except Exception as e:
        pytest.fail(f"raised Exception {e}")

def test_mul():
    with not_raises():
        for e in (BU, U, US):
            for e2 in (BU, U, US):
                _ = e * e2

def test_div():
    with not_raises():
        for e in (BU, U, US):
            for e2 in (BU, U, US):
                _ = e / e2
