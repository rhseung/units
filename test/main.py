from units import *

x = J * N * atm
print('original', x)
print('expand1 ', expand(x))
print('expand2 ', expand(expand(x)))
print('expand3 ', expand(expand(expand(x))))
print('expand4 ', expand(expand(expand(expand(x)))))
print('expand5 ', expand(expand(expand(expand(expand(x))))))
print('expand6 ', expand(expand(expand(expand(expand(expand(x)))))))
print('si      ', si(x))
