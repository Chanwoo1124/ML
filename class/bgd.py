import random

# ============================================================
# 1. 데이터셋
#    정답: H(x) = 0.3x + 1  →  w=0.3, b=1
# ============================================================
# x_data = [ 1, 2, 3, 4, 5 ... 50]
# y_data = [ 1.3, 1.6, 1.9, 2.2, 2.5 ... 16.0]
x_data = [count for count in range(1, 51)]
y_data = [0.3 * x + 1 for x in x_data]
num_of_sample = len(x_data)


# 2. 파라미터 초기화 (w, b)
# 랜덤한 값으로 시작. 학습을 거치면서 정답(w=0.3, b=1)에 가까워진다.

w = random.random()
b = random.random()

# 3. 하이퍼파라미터 설정 (learning_rate, epochs)
learning_rate = 0.001   # 한 번에 얼마나 이동할지 (보폭)
epochs = 500          # 전체 데이터를 몇 번 반복 학습할지

# 4. 학습 루프 (epoch 반복)
for epoch in range(1, epochs + 1):

    # 4-1. 그래디언트 초기화

    # 매 epoch마다 새로 누적해야 하므로 0으로 초기화한다.
    grad_w = 0.0
    grad_b = 0.0
    loss = 0.0

    # 4-2. 전체 샘플 순회 (BGD: 모든 샘플을 다 본 뒤 업데이트)
    for x, y in zip(x_data, y_data):

        # (a) 예측값 계산: H(x) = wx + b
        predict = w * x + b

        # (b) 오차 계산: error = 예측값 - 실제값
        error = predict - y

        # (c) 그래디언트 누적: grad_w += 2 * x * error
        #                      grad_b += 2 * error
        # MSE를 w, b로 편미분하면 이 식이 나온다 (체인룰 적용)
        grad_w += 2 * x * error
        grad_b += 2 * error

        # loss 누적 (나중에 평균 낼 것)
        loss += error ** 2

    # 4-3. 평균 그래디언트 계산 (누적값 / n)
    # 샘플 수로 나눠서 평균을 구한다. 데이터가 많아도 스케일이 일정해진다.
    grad_w /= num_of_sample
    grad_b /= num_of_sample
    loss /= num_of_sample

    # 4-4. 파라미터 업데이트: w = w - lr * grad_w
    # 기울기의 반대 방향으로 이동 → loss가 줄어드는 방향
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # 4-5. 손실(loss) 출력 (학습 경과 확인)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | w: {w:.4f}, b: {b:.4f}")


# 5. 최종 결과 출력
print()
print("========== 학습 완료 ==========")
print(f"학습된 w: {w:.4f}  (정답: 0.3)")
print(f"학습된 b: {b:.4f}  (정답: 1.0)")
