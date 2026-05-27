import numpy as np

foo = [1,2,3,4,5,6]
print(foo)
pos = np.array(foo)
print(pos)
print(pos.shape)

bar = pos.reshape(2,3)
print(bar)
print(bar.shape)

bar[0,1] = 100
print(pos)
