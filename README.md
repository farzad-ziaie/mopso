# mopso
a simple code for multi objective particle swarm optimization

refrences:
https://ieeexplore.ieee.org/document/1004388

MOPSO: a proposal for multiple objective particle swarm optimization
C.A. Coello Coello ; M.S. Lechuga

https://faradars.org/courses/mvrmo9012-multiobjective-optimization-video-tutorials-pack

S. Mostapha Kalami Heris
the codes are a replication from the matlab ones, so they are not that clean

example:

```python
from mopso import mopso
import numpy as np

class cost:            
    def evaluate(self, x):
        z1 = np.sum(x ** 2) + np.sum(x ** 3)
        z2 = np.sum(x ** 4)
        z3 = np.sum(x ** 6)
        return np.array([z1, z2, z3])

mo = mopso(cost__ = cost(), debug=True)
mo.fit(maxiter=2)
```






