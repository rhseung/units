from .core import BaseUnit, Unit
from .units import *

from math import pi

c = Unit("c", m/s, 299792458)
h = Unit("h", J*s, 6.62607015e-34)
hbar = Unit("ħ", h, 1/(2*pi))
e = Unit("e", C, 1.602176634e-19)
eV = Unit("eV", J, 1.602176634e-19)
