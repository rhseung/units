# units

## how to install
```powershell
pip install rhseung.units
```

## how to use
```py
from units import *

print(3*kg*m/s**2)
```

`kg`, `m`, `s`, `A`, `K`, `mol`, `cd`, `rad`, `N`, `J`, `Pa`, `W`, `atm`, `C`, `V`, `Ω`, `Wb`, `T`, `H`, `F` 등의 단위와  
`Y`, `Z`, `E`, ..., `μ`, `n`, `p`, `f`, `a`, `z`, `y` 등의 모든 접두사가 붙은 단위도 전부 이미 정의되어 있습니다.

<img src="https://github.com/rhseung/units/assets/56152093/5a124ebd-803f-4eb1-96b9-3788820695b8)https://github.com/rhseung/units/assets/56152093/5a124ebd-803f-4eb1-96b9-3788820695b8" alt="LaTeX compatible" width="250"/>  

jupyter notebook 환경에서 사용할 시 LaTeX를 지원합니다.

## examples
```py
expand(atm)
expand(J*N)
```
`expand` 함수는 주어진 단위 또는 물리량의 단위를 전개합니다.  
`atm`을 `expand` 함수에 넣으면 `101325 Pa`, `Pa`를 넣으면 `N/m**2` 가 됩니다.  

```py
si(atm)
si(J)
```
`si` 함수는 주어진 단위 또는 물리량의 단위를 si 단위계로 표현합니다.  
`atm`을 `si` 함수에 넣으면 `101325 kg/m*s**2`가 됩니다.  
`expand` 함수를 무한히 사용하면 `si` 함수의 결과와 동일해집니다.  
