import unittest

from unit import *
from unit.core import Unit
from unit.functions import *

class HashTest(unittest.TestCase):
    def setUp(self):
        self.kilometer = Unit('kilometer', m, 1000)

    def test_dict(self):
        o = {km: 3}
        self.assertEqual(o[self.kilometer], o[km])

    def test_aliase(self):
        self.assertEqual(km, self.kilometer)

    def test_aliase2(self):
        A_ = Unit("A", m)
        B_ = Unit("B", m)
        C_ = Unit("C", m)
        D_ = Unit("D", m)
        self.assertEqual(A_ * B_ * C_ * D_, A_ ** 4)

    def test_eq(self):
        cm = Unit('cm', m, 0.01)
        cc = Unit('cc', cm**3)
        self.assertEqual(cc, cm**3)

class SortTest(unittest.TestCase):
    def setUp(self):
        self.a = Unit('a', m)
        self.b = Unit('b', m**2)
        self.A = Unit('A', m**3)
        self.B = Unit('B', m**4)
        self.aa = Unit('aa', s)
        self.ab = Unit('ab', s**2)
        self.bb = Unit('bb', s**3)
        self.AA = Unit('AA', s**4)
        self.BB = Unit('BB', s**5)

    def test_sort(self):
        self.assertEqual(str(self.a * self.b * self.A * self.B * self.aa * self.ab * self.bb * self.AA * self.BB),
                         'AA⋅BB⋅aa⋅ab⋅bb⋅A⋅B⋅a⋅b')

class ExpandTest(unittest.TestCase):
    def test_expand(self):
        self.assertEqual(str(expand(Ω)), 'V/A')

    def test_si(self):
        self.assertEqual(str(si(Ω)), 'kg⋅m^2/A^2⋅s^3')

class DimlessTest(unittest.TestCase):
    def test_dimless(self):
        k = Unit("k", dimless, 10e4)
        print(km == k*m)    # fixme: 이거 왜 false 나옴
        self.assertEqual(si(km), si(k*m))

class CalcTest(unittest.TestCase):
    def setUp(self):
        self.BU = m
        self.U = T
        self.US = K**2

    def test_BU_U(self):
        self.assertIs(self.BU * self.U)
        ...

    # todo: 나머지 경우에 대해서도 모두 implement

tesla = Unit("Tesla", N / (C * m / s))
print(tesla.dim, T.dim)     # fixme: dim이 제대로 반영이 안됨
print(si(N / (C * m / s)), si(T))

# test: 무차원 테스트
