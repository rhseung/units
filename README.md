# units

## how to install
```powershell
pip install rhseung.units
```

## features

```py
from _units import *

print(3 * kg * m / s ** 2)
```

`kg`, `m`, `s`, `A`, `K`, `mol`, `cd`, `rad`, `N`, `J`, `Pa`, `W`, `atm`, `C`, `V`, `Ω`, `Wb`, `T`, `H`, `F` 등의 단위와  
`Y`, `Z`, `E`, ..., `μ`, `n`, `p`, `f`, `a`, `z`, `y` 등의 모든 접두사가 붙은 단위도 전부 이미 정의되어 있습니다.

<img src="https://github.com/rhseung/units/assets/56152093/5a124ebd-803f-4eb1-96b9-3788820695b8)https://github.com/rhseung/units/assets/56152093/5a124ebd-803f-4eb1-96b9-3788820695b8" alt="LaTeX compatible" width="250"/>  

jupyter notebook 환경에서 사용할 시 LaTeX를 지원합니다.  
<br>

```py
print(3*kg)
print(3*kg*m)
print(3.5*N/m**2)
print(4.5*Ω)
print((3 + 4j)*Ω)
print((1, 2)*m/s)
print((1, 2, 3, 4, 5.4)*m/s**2)
```
기본적으로 위와 같이 단위에 `ValueType`인 객체를 곱해서 물리량을 정의합니다.  
`ValueType`는 `int`, `float`, `complex`, `tuple` 등이 있습니다.  
`tuple`의 경우, 벡터로 인식합니다. 벡터의 원소에는 `int`, `float`만이 가능합니다.  
<br>
```py
import numpy as np

print(np.array([1, 2, 3]) * m)  # array([1 [m] 2 [m] 3[m]])
print(np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]))
# array([[1 [m] 2 [m] 3 [m]]
#        [4 [m] 5 [m] 6 [m]]
#        [7 [m] 8 [m] 9[m]]])
print([1, 2, 3]*m)  # [1 [m], 2 [m], 3[m]]
```
`numpy`의 `array`와 `list`는 브로드캐스팅 가능한 객체로 인식합니다.  
이 경우, `array` 혹은 `list`의 모든 요소에 단위가 곱해집니다. 2차원 이상의 `array`에서도 성립합니다.  
<br>  
```py
quantity = 3*N
print(quantity)     # 3 [N]
print(quantity.value)   # 3
print(quantity.unit)    # N
```
물리량은 `Quantity` 클래스로 정의됩니다.  
`Quantity` 클래스는 `value`와 `unit` 속성을 가집니다. `value`는 물리량의 값을 반환하며, `unit`은 물리량의 단위를 반환합니다.  
<br>
```py
expand(atm)
expand(3*J*N)
```
`expand` 함수는 주어진 단위 또는 물리량의 단위를 전개합니다.  
`atm`을 `expand` 함수에 넣으면 `101325 Pa`, `Pa`를 넣으면 `N/m**2` 가 됩니다.  
<br>
```py
si(atm)
si(J)
si(4*T)
```
`si` 함수는 주어진 단위 또는 물리량의 단위를 si 단위계로 표현합니다.  
`atm`을 `si` 함수에 넣으면 `101325 kg/m*s**2`가 됩니다.  
`expand` 함수를 무한히 사용하면 `si` 함수의 결과와 동일해집니다.  
