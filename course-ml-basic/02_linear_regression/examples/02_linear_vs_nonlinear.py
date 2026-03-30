"""
Linear vs Nonlinear - 선형 회귀의 한계
선형 데이터와 비선형 데이터에 각각 선형 회귀를 적용하여,
선형 모델이 언제 잘 작동하고 언제 실패하는지 시각적으로 확인한다.
"""

import math
import platform

import matplotlib.pyplot as plt

# 한글 폰트 설정 (OS별 자동 선택)
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 데이터셋 생성
# ============================================================

# (A) 선형 데이터: y = 2x + 1 (+ 약간의 노이즈)
#     선형 회귀가 잘 맞는 경우
linear_x = [i * 0.5 for i in range(1, 21)]  # 0.5 ~ 10.0

# 간단한 난수 생성 (시드 고정을 위해 직접 구현)
# Linear Congruential Generator
_seed = 42


def _simple_random():
    """재현 가능한 간단한 난수 생성 (-0.5 ~ 0.5 범위)"""
    global _seed
    _seed = (1103515245 * _seed + 12345) % (2**31)
    return (_seed / (2**31)) - 0.5


linear_y = [2 * x + 1 + _simple_random() * 2 for x in linear_x]

# (B) 비선형 데이터: y = 0.3x² - 2x + 5 (+ 약간의 노이즈)
#     선형 회귀로는 패턴을 포착할 수 없는 경우
nonlinear_x = [i * 0.5 for i in range(1, 21)]  # 0.5 ~ 10.0
nonlinear_y = [0.3 * x**2 - 2 * x + 5 + _simple_random() * 1.5 for x in nonlinear_x]

# ============================================================
# 2. 선형 회귀 학습 (순수 Python)
#    01번 예제와 동일한 경사하강법을 함수로 묶어 두 데이터에 적용
# ============================================================


def train_linear_regression(x_data, y_data, lr=0.001, epochs=5000):
    """경사하강법으로 w, b를 학습하여 반환한다."""
    w = 0.0
    b = 0.0
    n = len(x_data)

    for epoch in range(epochs):
        # 예측
        y_pred = [w * x + b for x in x_data]

        # 그래디언트 계산
        grad_w = 0.0
        grad_b = 0.0
        for i in range(n):
            error = y_pred[i] - y_data[i]
            grad_w += error * x_data[i]
            grad_b += error
        grad_w = (2 / n) * grad_w
        grad_b = (2 / n) * grad_b

        # 업데이트
        w = w - lr * grad_w
        b = b - lr * grad_b

    # 최종 손실 계산
    y_pred = [w * x + b for x in x_data]
    loss = sum((p - y) ** 2 for p, y in zip(y_pred, y_data)) / n

    return w, b, loss


# 두 데이터셋에 동일한 선형 회귀 적용
w_lin, b_lin, loss_lin = train_linear_regression(linear_x, linear_y)
w_non, b_non, loss_non = train_linear_regression(nonlinear_x, nonlinear_y)

# ============================================================
# 3. 결과 출력
# ============================================================
print("=" * 50)
print("선형 데이터에 선형 회귀 적용")
print(f"  학습된 모델: H(x) = {w_lin:.4f}x + {b_lin:.4f}")
print(f"  MSE 손실: {loss_lin:.4f}")
print()
print("비선형 데이터에 선형 회귀 적용")
print(f"  학습된 모델: H(x) = {w_non:.4f}x + {b_non:.4f}")
print(f"  MSE 손실: {loss_non:.4f}")
print("=" * 50)
print()
print(f"손실 비교: 비선형({loss_non:.4f}) vs 선형({loss_lin:.4f})")
print("→ 비선형 데이터에서 손실이 훨씬 크다 = 선형 모델이 데이터를 잘 설명하지 못한다.")

# ============================================================
# 4. 시각화
#    왼쪽: 선형 데이터 + 회귀선 (잘 맞음)
#    오른쪽: 비선형 데이터 + 회귀선 (잘 안 맞음)
# ============================================================

# 회귀선을 그리기 위한 x 범위
plot_x = [i * 0.1 for i in range(0, 110)]  # 0.0 ~ 10.9

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- 왼쪽: 선형 데이터 ---
ax1 = axes[0]
ax1.scatter(linear_x, linear_y, color="steelblue", label="데이터", zorder=3)
pred_line = [w_lin * x + b_lin for x in plot_x]
ax1.plot(plot_x, pred_line, color="tomato", linewidth=2, label=f"H(x) = {w_lin:.2f}x + {b_lin:.2f}")
ax1.set_title(f"선형 데이터 (MSE: {loss_lin:.4f})", fontsize=13)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 오른쪽: 비선형 데이터 ---
ax2 = axes[1]
ax2.scatter(nonlinear_x, nonlinear_y, color="steelblue", label="데이터", zorder=3)
pred_line_non = [w_non * x + b_non for x in plot_x]
ax2.plot(plot_x, pred_line_non, color="tomato", linewidth=2, label=f"H(x) = {w_non:.2f}x + {b_non:.2f}")

# 실제 비선형 곡선도 함께 표시
true_curve = [0.3 * x**2 - 2 * x + 5 for x in plot_x]
ax2.plot(plot_x, true_curve, color="gray", linewidth=1.5, linestyle="--", label="실제 곡선 (y=0.3x²-2x+5)")

ax2.set_title(f"비선형 데이터 (MSE: {loss_non:.4f})", fontsize=13)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("선형 회귀는 선형 관계에서만 잘 작동한다", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
