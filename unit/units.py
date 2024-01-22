from .core import BaseUnit, Unit
from .dimension import Dimension

from math import pi

# todo: 검색 함수 ex. with(m) => m 들어가는 모든 단위

# mks
m = BaseUnit("m", Dimension(length=1.))
kg = BaseUnit("kg", Dimension(mass=1.))
s = BaseUnit("s", Dimension(time=1.))
K = BaseUnit("K", Dimension(temperature=1.))
A = BaseUnit("A", Dimension(electric_current=1.))
mol = BaseUnit("mol", Dimension(amount_of_substance=1.))
cd = BaseUnit("cd", Dimension(luminous_intensity=1.))
dimless = BaseUnit("dimless", Dimension())

# prefixs
femto = Unit("f", dimless, 1e-15)
pico = Unit("p", dimless, 1e-12)
nano = Unit("n", dimless, 1e-9)
micro = Unit("μ", dimless, 1e-6)
milli = Unit("m", dimless, 1e-3)
kilo = Unit("k", dimless, 1e3)
mega = Unit("M", dimless, 1e3)
giga = Unit("G", dimless, 1e9)
tera = Unit("T", dimless, 1e12)
peta = Unit("P", dimless, 1e15)

# common units
cm = Unit("cm", m, 1e-2)
g = Unit("g", kg, 1e-3)
Gal = Unit("Gal", m/s**2, 1e-2)
minute = Unit("min", s, 60)
hour = Unit("h", s, 3600)
N = Unit("N", kg * m / s ** 2)
Pa = Unit("Pa", N / m ** 2)
atm = Unit("atm", Pa, 101325)   # test: 1 atm 을 expand 하면 101325 Pa가 나와야 함
Torr = Unit("Torr", Pa, 101325 / 760)
bar = Unit("bar", Pa, 1e5)
oz = Unit("oz", kg, 0.028349523125)
cc = Unit("cc", cm ** 3)
L = Unit("L", cm ** 3, 1000)
J = Unit("J", N * m)
cal = Unit("cal", J, 4.184)
angstrom = Unit("Å", m, 1e-10)

# todo: 덧셈 관계 구현
# degC = Unit("°C", K, 1, 273.15)
# degF = Unit("°F", K, 5 / 9, 459.67)
W = Unit("W", J / s)
Wh = Unit("Wh", W * hour)
C = Unit("C", A * s)
V = Unit("V", J / C)
F = Unit("F", C / V)
ohm = Unit("Ω", V / A)
Wb = Unit("Wb", V * s)
T = Unit("T", Wb / m ** 2)
G = Unit("G", T, 1e-4)
H = Unit("H", Wb / A)
kn = Unit("kn", kilo*m/hour, 1.852)
rad = Unit("rad", dimless, 180 / pi)

# cgs
dyn = Unit("dyn", N, 1e-5)
erg = Unit("erg", J, 1e-7)

# US units
inch = Unit("in", m, 0.254)
ft = Unit("ft", inch, 12)
yd = Unit("yd", ft, 3)
mile = Unit("mi", yd, 1760)
lb = Unit("lb", oz, 16)
