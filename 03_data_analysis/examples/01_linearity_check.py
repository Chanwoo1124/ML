"""
데이터 분석 - 선형성 판단 예제
산점도와 MSE를 활용하여 데이터가 선형 관계인지 판단하는 방법을 보여준다.

판단 기준:
  1. 산점도: 점들이 직선 형태로 분포하는가?
  2. 선형 회귀 적용 후 잔차(residual): 잔차가 무작위로 흩어지면 선형,
     패턴이 보이면 비선형 관계를 의심한다.
"""

import platform

import matplotlib.pyplot as plt

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 선형 회귀 학습 함수 (02장에서 가져온 것과 동일)
# ============================================================
def train_linear_regression(x_data, y_data, lr=0.001, epochs=5000):
    """경사하강법으로 w, b를 학습하여 (w, b, loss)를 반환한다."""
    w = 0.0
    b = 0.0
    n = len(x_data)

    for epoch in range(epochs):
        y_pred = [w * x + b for x in x_data]

        grad_w = 0.0
        grad_b = 0.0
        for i in range(n):
            error = y_pred[i] - y_data[i]
            grad_w += error * x_data[i]
            grad_b += error
        grad_w = (2 / n) * grad_w
        grad_b = (2 / n) * grad_b

        w = w - lr * grad_w
        b = b - lr * grad_b

    y_pred = [w * x + b for x in x_data]
    loss = sum((p - y) ** 2 for p, y in zip(y_pred, y_data)) / n
    return w, b, loss


# ============================================================
# 2. 데이터셋 2개: 하나는 선형, 하나는 비선형
# ============================================================

# (A) 선형: 주행 거리(만 km)에 따른 중고차 가격(만 원)
#     거리가 늘수록 가격이 일정하게 하락 → 선형 관계
mileage = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
car_price = [2850, 2620, 2400, 2210, 1980, 1770, 1560, 1350, 1180, 950]

# (B) 비선형: 약물 투여량(mg)에 따른 효과(%)
#     처음엔 빠르게 증가하다가 점점 포화 → 로그 형태
dosage = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50]
effect = [15, 30, 42, 55, 65, 72, 78, 82, 86, 89]


# ============================================================
# 3. 분석 및 시각화
# ============================================================
datasets = [
    ("주행 거리 vs 중고차 가격", mileage, car_price, 0.00001),
    ("약물 투여량 vs 효과", dosage, effect, 0.0001),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for idx, (title, x_data, y_data, lr) in enumerate(datasets):
    w, b, loss = train_linear_regression(x_data, y_data, lr=lr)
    y_pred = [w * x + b for x in x_data]
    residuals = [y_data[i] - y_pred[i] for i in range(len(x_data))]

    # 왼쪽: 산점도 + 회귀선
    ax_left = axes[idx][0]
    ax_left.scatter(x_data, y_data, color="steelblue", zorder=3, label="데이터")
    plot_x = [min(x_data) + (max(x_data) - min(x_data)) * i / 100 for i in range(101)]
    plot_y = [w * x + b for x in plot_x]
    ax_left.plot(plot_x, plot_y, color="tomato", linewidth=2, label=f"H(x) = {w:.2f}x + {b:.2f}")
    ax_left.set_title(f"{title}\n(MSE: {loss:.2f})", fontsize=12)
    ax_left.legend()
    ax_left.grid(True, alpha=0.3)

    # 오른쪽: 잔차 그래프
    ax_right = axes[idx][1]
    ax_right.scatter(x_data, residuals, color="darkorange", zorder=3)
    ax_right.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax_right.set_title(f"잔차 (residual) 분포", fontsize=12)
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("잔차 (실제 - 예측)")
    ax_right.grid(True, alpha=0.3)

print("=" * 60)
print("선형성 판단 가이드")
print("=" * 60)
print()

for idx, (title, x_data, y_data, lr) in enumerate(datasets):
    w, b, loss = train_linear_regression(x_data, y_data, lr=lr)
    y_pred = [w * x + b for x in x_data]
    residuals = [y_data[i] - y_pred[i] for i in range(len(x_data))]

    print(f"[데이터셋 {idx + 1}] {title}")
    print(f"  회귀식: H(x) = {w:.2f}x + {b:.2f}")
    print(f"  MSE: {loss:.2f}")

    # 잔차 패턴 확인: 부호가 연속으로 바뀌지 않으면 비선형 의심
    sign_changes = sum(
        1 for i in range(1, len(residuals))
        if (residuals[i] > 0) != (residuals[i - 1] > 0)
    )
    print(f"  잔차 부호 변경 횟수: {sign_changes}회 (많을수록 무작위 → 선형 가능성 높음)")
    print()

print("판단 방법:")
print("  1. 산점도에서 점들이 직선 형태로 분포하는가?")
print("  2. 잔차가 무작위로 흩어져 있는가?")
print("     → 잔차에 곡선 패턴이 보이면 비선형 관계를 의심한다.")

plt.suptitle("선형성 판단: 산점도와 잔차 분석", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
