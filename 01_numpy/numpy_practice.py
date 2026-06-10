import numpy as np

N, D = 5, 4

w_true = np.array([0.2,0.4,0.5,0.7]).reshape(-1,1)

b_true = 2.0


y_data = np.random.random((N, D) - 0,5) * 8

y_pred = x_data @ w_true + b_true # N, 1

print(y_data)





