# from _units import *
#
# v = Vector(1, 2, 3)
# print(f"{v:p}")

import unit as u
from unit.core import BaseUnit

Tera = u.Unit("T", u.dimless, 10e12)
Tesla = u.Unit("T", u.kg, 132)
print(Tera, Tesla)
print(type(Tera.scale), type(Tesla.scale))
