#데이터 샘플
import random


x_data = [x for x in range(1,51)]
#정답 데이터
y_data = [0.3 * x + 1 for x in x_data]
#샘플 개수
num_of_sample = len(x_data)

#파라매터 설정
w = random.random()
b = random.random()

#하이퍼 파라매터 설정
learning_rate = 0.001
epochs = 1000

#학습 루프
for epoch in range(1,epochs+1):
    #그래디언트 초기화
    grad_w = 0.0
    grad_b = 0.0
    loss = 0.0
    # 샘플 순회
    for x,y in zip(x_data,y_data):
        # 예측 값 도출
        predict = w * x + b
        # 오차 계산
        error = predict - y
        
        #그래디언트 값 누적
        #MSE w,b 미분 계산
        grad_w += 2 * x * error
        grad_b += error * 2

        loss += error ** 2
    
    #그래디언트 평균값 계산
    grad_w /= num_of_sample
    grad_b /= num_of_sample

    loss /= num_of_sample

    #파라매터 업데이트
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
