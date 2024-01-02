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

# N = kg * m / s**2
# Pa = N / m

# km = Unit("km", m, 1000)
# atm = Unit("atm", Pa, 101325)

__all__ = ['m', 'kg', 's', 'K', 'A', 'mol', 'cd', 'dimless']
