from units import *

v1 = Vector(1.5, 2.3, 3.4)
v2 = Vector(3.5, 4.7, -0.9)
v3 = Vector(3.5, 4.7)
v4 = Vector.polar(3, 3.14159/6)
v5 = Vector(3)
print(v1 * v2)
print(v1 @ v2)
print(v4.angle())
print(f"{v4:u}")
print(v5, type(v5))
