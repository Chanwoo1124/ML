import numpy as np


w_true = np.array([0.2,0.4,0.6,0.8]).reshape(-1,1)

b_true = 2.0

x_data = (np.random.random((5, 4)) - 0.5 ) * 8

y_data = x_data @ w_true + b_true

print(y_data)




