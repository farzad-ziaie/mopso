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

mo = mopso(cost__ = cost())
mo.fit(maxiter=2)
```

the outputs: 
call the mo.repo to see the best positions, it's a list of best particles
each of the elements in the mo.repo is a dictionary.
```python
mo.repo[0] = {'position': random_pos[i],
  'velocity': np.zeros(self.weight_size),
  'cost': [],
  'best_position': [],
  'best_cost': [],
  'isdominated': [],
  'gridindex': [],
  'gridsubindex': []}
```



