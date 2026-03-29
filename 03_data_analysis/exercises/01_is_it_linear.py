"""
실습: 이 데이터, 선형 회귀로 풀 수 있을까?
==========================================

4개의 실생활 데이터셋이 주어진다.
각 데이터셋에 대해 다음을 수행하라:

  1. 산점도를 그려 데이터의 분포를 눈으로 확인한다.
  2. 선형 회귀를 적용하고 MSE 손실을 계산한다.
  3. 잔차(residual = 실제값 - 예측값)를 계산하고 그래프로 확인한다.
  4. 위 결과를 종합하여 "선형 회귀가 적합한가?"를 판단한다.

지시사항:
  - TODO 로 표시된 부분을 채워 넣으세요.
  - 각 함수의 docstring을 참고하세요.
  - 마지막에 판단 결과를 answers 딕셔너리에 기록하세요.
"""

import platform

import matplotlib.pyplot as plt

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 데이터셋 4개
# ============================================================

datasets = {
    "A": {
        "title": "공부 시간 vs 시험 점수",
        "description": "학생 12명의 일일 공부 시간(h)과 시험 점수(100점 만점)",
        "x": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "y": [23, 30, 35, 42, 48, 55, 59, 65, 72, 78, 82, 88],
    },
    "B": {
        "title": "차량 속도 vs 제동 거리",
        "description": "차량 속도(km/h)에 따른 제동 거리(m). 물리 법칙상 제동 거리는 속도의 제곱에 비례한다.",
        "x": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        "y": [6, 12, 22, 34, 50, 68, 90, 115, 142, 172, 206],
    },
    "C": {
        "title": "기온 vs 아이스크림 판매량",
        "description": "일일 평균 기온(°C)과 아이스크림 판매량(개). 기온이 오를수록 판매량도 꾸준히 증가한다.",
        "x": [15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
        "y": [120, 145, 170, 200, 235, 260, 295, 320, 350, 380, 410],
    },
    "D": {
        "title": "나무 나이 vs 높이",
        "description": "나무 나이(년)에 따른 높이(m). 어릴 때 빠르게 자라다가 점점 성장이 둔화된다.",
        "x": [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70],
        "y": [0.5, 1.8, 3.5, 6.2, 9.5, 13.0, 16.5, 19.0, 21.5, 23.0, 24.0],
    },
}


# ============================================================
# 함수 정의
# ============================================================

def train_linear_regression(x_data, y_data, lr, epochs):
    """
    경사하강법으로 선형 회귀를 학습한다.

    Args:
        x_data: 입력 데이터 리스트
        y_data: 실제값 리스트
        lr: 학습률
        epochs: 반복 횟수
    Returns:
        (w, b, loss) 튜플 - 학습된 파라미터와 최종 MSE 손실
    """
    w = 0.0
    b = 0.0
    n = len(x_data)

    for epoch in range(epochs):
        # TODO: 예측값 리스트를 구하세요. y_pred = [w * x + b for x in x_data]
        y_pred = None

        # TODO: 그래디언트를 계산하세요.
        #   grad_w = (2/n) * Σ (y_pred[i] - y_data[i]) * x_data[i]
        #   grad_b = (2/n) * Σ (y_pred[i] - y_data[i])
        grad_w = 0.0
        grad_b = 0.0

        # TODO: 파라미터를 업데이트하세요.
        #   w = w - lr * grad_w
        #   b = b - lr * grad_b
        pass

    # TODO: 최종 손실(MSE)을 계산하세요.
    y_pred = [w * x + b for x in x_data]
    loss = None

    return w, b, loss


def compute_residuals(x_data, y_data, w, b):
    """
    잔차(residual)를 계산한다.
    잔차 = 실제값 - 예측값

    잔차가 무작위로 흩어져 있으면 → 선형 모델이 적합
    잔차에 곡선 패턴이 보이면     → 비선형 관계를 의심

    Args:
        x_data: 입력 데이터 리스트
        y_data: 실제값 리스트
        w: 학습된 가중치
        b: 학습된 편향
    Returns:
        잔차 리스트
    """
    # TODO: 각 데이터 포인트에 대해 (실제값 - 예측값)을 계산하여 리스트로 반환하세요.
    pass


# ============================================================
# 분석 실행
# ============================================================
if __name__ == "__main__":
    # 학습률 (데이터 스케일에 따라 다르게 설정)
    learning_rates = {"A": 0.01, "B": 0.0000001, "C": 0.00001, "D": 0.00001}

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))

    results = {}

    for idx, (key, data) in enumerate(datasets.items()):
        x = data["x"]
        y = data["y"]
        lr = learning_rates[key]

        print(f"\n[데이터셋 {key}] {data['title']}")
        print(f"  설명: {data['description']}")

        # 학습
        w, b, loss = train_linear_regression(x, y, lr=lr, epochs=5000)
        print(f"  회귀식: H(x) = {w:.4f}x + {b:.4f}")
        print(f"  MSE: {loss:.4f}")

        # 잔차 계산
        residuals = compute_residuals(x, y, w, b)

        results[key] = {"w": w, "b": b, "loss": loss, "residuals": residuals}

        # --- 왼쪽: 산점도 + 회귀선 ---
        ax_left = axes[idx][0]
        ax_left.scatter(x, y, color="steelblue", zorder=3, label="데이터")
        plot_x_min, plot_x_max = min(x), max(x)
        plot_x = [plot_x_min + (plot_x_max - plot_x_min) * i / 100 for i in range(101)]
        plot_y = [w * xi + b for xi in plot_x]
        ax_left.plot(plot_x, plot_y, color="tomato", linewidth=2,
                     label=f"H(x) = {w:.2f}x + {b:.2f}")
        ax_left.set_title(f"[{key}] {data['title']} (MSE: {loss:.2f})", fontsize=11)
        ax_left.legend(fontsize=9)
        ax_left.grid(True, alpha=0.3)

        # --- 오른쪽: 잔차 그래프 ---
        ax_right = axes[idx][1]
        ax_right.scatter(x, residuals, color="darkorange", zorder=3)
        ax_right.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax_right.set_title(f"[{key}] 잔차 분포", fontsize=11)
        ax_right.set_ylabel("잔차")
        ax_right.grid(True, alpha=0.3)

    plt.suptitle("각 데이터셋의 산점도와 잔차 분석", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ============================================================
    # 최종 판단
    # ============================================================
    # TODO: 각 데이터셋에 대해 선형 회귀가 적합한지 판단하세요.
    #       "linear" 또는 "nonlinear" 로 기록하세요.
    #
    # 판단 기준:
    #   - 산점도에서 점들이 직선 형태로 분포하는가?
    #   - 잔차가 무작위로 흩어져 있는가? (패턴 없이 0 주변에 분포)
    #   - 잔차에 곡선 패턴(U자, 역U자 등)이 보이면 비선형을 의심
    answers = {
        "A": None,  # TODO: "linear" 또는 "nonlinear"
        "B": None,  # TODO: "linear" 또는 "nonlinear"
        "C": None,  # TODO: "linear" 또는 "nonlinear"
        "D": None,  # TODO: "linear" 또는 "nonlinear"
    }

    print("\n" + "=" * 50)
    print("최종 판단")
    print("=" * 50)
    for key in datasets:
        label = answers[key] if answers[key] else "(미작성)"
        print(f"  [{key}] {datasets[key]['title']}: {label}")

    plt.show()
