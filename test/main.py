from units import *

x = (1, 6) / g / s * YPa * nV * Ts
z = 1/(kg*m)*A*s**5

print(x * z)
print((x*z) + (1, 3))
