# 데이터셋
import random


x_data = [count for count in range(1,51)]
y_data = [0.3 * x + 1 for x in x_data]
num_of_sample = len(x_data)

#파라매터 초기화
w = random.random()
b = random.random()

#하이퍼 파라매터 설정
learning_rate = 0.001
epochs = 1000

#학습 루트
for epoch in range(1,epochs+1):
    #그래디언트 초기화
    grad_w = 0.0
    grad_b = 0.0
    loss = 0.0
    # 샘플 순회
    for x,y in zip(x_data,y_data):
        #예측값 도출
        predict = w * x + b
        # 오차
        error = predict - y

        #그래디언트 누적
        #MSE를 W,b로 미분
        grad_w += 2 * x * error
        grad_b += error * 2

        loss += error ** 2

    #그라디언트 평균
    grad_w /= num_of_sample
    grad_b /= num_of_sample
    #loss 누적
    loss /= num_of_sample

    # 매개변수 업데이트
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    if epoch % 100 == 0:
        print(f" epoch : {epoch:.4f} | w : {w:.4f} | b : {b:.4f}")

print("학습 완료")
print(f"w : {w:.4f} | b : {b:.4f}")


