"""
src/evaluation/metrics.py
--------------------------
模型评估指标。

为什么不用准确率？
  足球结果分布极度不均（平局只有约25%），
  准确率会被"永远预测主队赢"的模型欺骗。
  更好的指标：

  1. Log-loss（对数损失）
     惩罚高置信度的错误预测，越小越好。
     好的体育预测模型通常 log-loss < 1.0。

  2. Brier Score
     类似 MSE，衡量概率校准程度，越小越好。
     随机猜测 Brier ≈ 0.67；好模型 < 0.55。

  3. Calibration（校准曲线）
     预测概率 0.7 的比赛，实际胜率是否真的约 70%？
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve


def evaluate(y_true: np.ndarray, y_proba: np.ndarray,
             label_map: dict = None) -> dict:
    """
    计算多分类预测的评估指标。

    参数：
        y_true:    真实标签（整数编码）
        y_proba:   预测概率矩阵，shape (n, 3)
        label_map: {0: 'W', 1: 'D', 2: 'L'}

    返回：指标字典
    """
    label_map = label_map or {0: "W", 1: "D", 2: "L"}

    ll = log_loss(y_true, y_proba)

    # Brier Score：对每个类别单独计算后取均值
    brier_scores = {}
    for idx, name in label_map.items():
        y_bin = (y_true == idx).astype(int)
        brier_scores[f"brier_{name}"] = brier_score_loss(y_bin, y_proba[:, idx])
    avg_brier = np.mean(list(brier_scores.values()))

    # 准确率（作为参考）
    y_pred = np.argmax(y_proba, axis=1)
    accuracy = (y_pred == y_true).mean()

    metrics = {
        "log_loss": round(ll, 4),
        "avg_brier": round(avg_brier, 4),
        "accuracy": round(accuracy, 4),
        **{k: round(v, 4) for k, v in brier_scores.items()},
    }

    print("📊 评估结果：")
    print(f"   Log-loss  : {metrics['log_loss']}  （随机猜测≈1.10）")
    print(f"   Brier Score: {metrics['avg_brier']}  （随机猜测≈0.67）")
    print(f"   Accuracy  : {metrics['accuracy']}  （参考值，足球通常50-55%）")
    return metrics


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray,
                     class_idx: int = 0, class_name: str = "Win",
                     save_path: str = None):
    """
    画概率校准曲线。
    理想情况下曲线应贴近对角线（预测概率=实际频率）。
    """
    y_bin = (y_true == class_idx).astype(int)
    prob_true, prob_pred = calibration_curve(y_bin, y_proba[:, class_idx], n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="完美校准")
    ax.plot(prob_pred, prob_true, "o-", label=f"模型（{class_name}）")
    ax.set_xlabel("预测概率")
    ax.set_ylabel("实际频率")
    ax.set_title(f"校准曲线 — {class_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 校准曲线已保存至 {save_path}")
    plt.show()


def compare_models(results: dict):
    """
    对比多个模型的指标。

    参数：
        results: {"模型名": metrics_dict, ...}
    """
    df = pd.DataFrame(results).T
    print("\n🏆 模型对比：")
    print(df[["log_loss", "avg_brier", "accuracy"]].to_string())
    return df

