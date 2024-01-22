from unit import *
from unit.functions import *
import unit.constants as C

A_ = Unit('A', m, 10)
B_ = Unit('B', A_*m, 10)
C_ = Unit('C', B_*m, 10)
print(abs(A_), abs(B_), abs(C_))

C_e = expand(C_)
C_e_e = expand(C_e)
print(C_e_e, abs(C_e_e))
print(C_e, abs(C_e))
print(C_, abs(C_))
