import pytest
from contextlib import contextmanager

from unit import *
from unit.core import Unit
from unit.functions import *

kilometer = Unit('kilometer', m, 1000)
km = Unit('km', m, 1000)

def test_dict():
    o = {kilometer: 3}
    assert o[kilometer] == o[km]

def test_aliase():
    assert km == kilometer

def test_aliase2():
    A_ = Unit("A", m)
    B_ = Unit("B", m)
    C_ = Unit("C", m)
    D_ = Unit("D", m)
    assert A_ * B_ * C_ * D_ == A_ ** 4

def test_eq():
    cm = Unit('cm', m, 0.01)
    cc = Unit('cc', cm**3)
    assert cc == cm**3

zero1 = Unit('z1', m, 0)

def test_zero():
    assert zero1 == 0

def test_zero2():
    assert Unit('z2', K*A, 0) == Unit('z3', s, 0)

def test_zero3():
    o = {0: 4}
    o[zero1] += 3
    assert o[0] == 7

def test_zero4():
    o = {zero1: 4}
    o[0] += 3
    assert o[zero1] == 7
