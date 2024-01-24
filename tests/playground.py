from ..unit import *
from ..unit.core import Unit

A_ = Unit('A', m, 10)       # A = 10 m
B_ = Unit('B', A_*m, 10)    # B = 10 A m = 100 m^2
C_ = Unit('C', B_*m, 10)    # C = 10 B m = 100 A m^2 = 1000 m^3

print(C_, expand(C_), expand(expand(C_)), expand(expand(expand(C_))))
