from .core import BaseUnit, Unit
from .dimension import Dimension

m = BaseUnit("m", Dimension(length=1.))
kg = BaseUnit("kg", Dimension(mass=1.))
s = BaseUnit("s", Dimension(time=1.))
K = BaseUnit("K", Dimension(temperature=1.))
A = BaseUnit("A", Dimension(electric_current=1.))
mol = BaseUnit("mol", Dimension(amount_of_substance=1.))
cd = BaseUnit("cd", Dimension(luminous_intensity=1.))
dimless = BaseUnit("dimless", Dimension())

N = Unit("N", kg * m / s ** 2)
Pa = Unit("Pa", N / m ** 2)
J = Unit("J", N * m)
W = Unit("W", J / s)
C = Unit("C", A * s)
V = Unit("V", J / C)
F = Unit("F", C / V)
Ω = Unit("Ω", V / A)
Wb = Unit("Wb", V * s)
T = Unit("T", Wb / m ** 2)
H = Unit("H", Wb / A)

km = Unit("km", m, 1000)
atm = Unit("atm", Pa, 101325)

__all__ = ['m', 'kg', 's', 'K', 'A', 'mol', 'cd', 'dimless',
           'N', 'Pa', 'J', 'W', 'C', 'V', 'F', 'Ω', 'Wb', 'T', 'H', 'km', 'atm'
           ]
