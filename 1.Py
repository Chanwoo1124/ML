# ============================================================
# 1. 데이터셋
#    정답: H(x) = 0.5x + 2  →  w=0.5, b=2
# ============================================================
import random

x_data = [1, 2, 3, 4]
y_data = [2.5, 3.0, 3.5, 4.0]
num_of_samples = len(x_data)

epochs = 1

w = random.random()
b = random.random()

for epoch in range(1,epochs+1):
    print(f"Epoch {epoch}번째 epochs")

    w_grad = 0.0
    b_grad = 0.0

    #샘플을 순회
    for x, y in zip(x_data, y_data):
        print(f"  x: {x}\t, y: {y}")

        #예측 값 계산
        #h(x) = wx + b
        predict = w * x + b

        #Error 계산
        error = predict - y
        print(f"  Error: {error}")

        #w의 기울기(Loss) 값 누적
        w_grad +=2 * x * error

        #b의 기울기(Loss) 값 누적
        b_grad += 2 * error

w_grad /= num_of_samples
b_grad /= num_of_samples