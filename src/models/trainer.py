"""
src/models/trainer.py
----------------------
模型训练与预测模块。

支持两个模型：
  - LogisticRegression（基准，可解释）
  - XGBoost（进阶，精度更高）

输出：
  - 模型文件（outputs/model_*.pkl）
  - 2026 赛程预测（outputs/predictions_2026.csv）
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.builder import FEATURE_COLS

OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"W": 0, "D": 1, "L": 2}  # 主队视角：胜/平/负


# ─────────────────────────────────────────────────────────────
# 数据准备
# ─────────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    """提取特征矩阵 X 和标签 y，自动丢弃缺失值行"""
    clean = df[FEATURE_COLS + ["result"]].dropna()
    X = clean[FEATURE_COLS].values
    y = clean["result"].map(LABEL_MAP).values
    print(f"📐 训练样本：{len(X)} 场（丢弃了 {len(df)-len(clean)} 行缺失值）")
    print(f"   标签分布：{ {k: (y==v).sum() for k,v in LABEL_MAP.items()} }")
    return X, y


# ─────────────────────────────────────────────────────────────
# 模型定义
# ─────────────────────────────────────────────────────────────

def build_logistic():
    """
    Logistic Regression pipeline。
    StandardScaler 归一化对 LR 非常重要（特征量纲不同）。
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            C=1.0,           # 正则化强度，越小越正则
            random_state=42,
        )),
    ])


def build_xgboost():
    """
    XGBoost 分类器。
    不需要归一化，但需要调参（这里用了合理的默认值）。
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )


# ─────────────────────────────────────────────────────────────
# 训练 & 交叉验证
# ─────────────────────────────────────────────────────────────

def train_and_evaluate(X, y, model_name: str = "logistic"):
    """
    训练模型并用 5-fold 交叉验证评估。

    返回训练好的模型。
    """
    model = build_logistic() if model_name == "logistic" else build_xgboost()

    print(f"\n🏋️  训练模型：{model_name}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Log-loss（负对数似然，越小越好）
    ll_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
    print(f"   CV Log-loss：{-ll_scores.mean():.4f} ± {ll_scores.std():.4f}")

    # 准确率（作为参考，足球预测准确率通常 50~55%）
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"   CV Accuracy：{acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

    # 用全量数据训练最终模型
    model.fit(X, y)

    # 保存模型
    model_path = OUTPUTS_DIR / f"model_{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 模型已保存至 {model_path}")

    return model


# ─────────────────────────────────────────────────────────────
# 预测 2026 赛程
# ─────────────────────────────────────────────────────────────

def predict_matches(model, fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    对 fixtures 中的比赛生成预测概率。

    fixtures 需包含 FEATURE_COLS 中的特征列。
    输出新增：pred_prob_W, pred_prob_D, pred_prob_L, pred_result
    """
    X = fixtures[FEATURE_COLS].fillna(0).values  # 缺失值用0填充（实际应更谨慎）
    probs = model.predict_proba(X)               # shape: (n, 3)

    inv_map = {v: k for k, v in LABEL_MAP.items()}
    result = fixtures.copy()
    result["pred_prob_W"] = probs[:, LABEL_MAP["W"]].round(3)
    result["pred_prob_D"] = probs[:, LABEL_MAP["D"]].round(3)
    result["pred_prob_L"] = probs[:, LABEL_MAP["L"]].round(3)
    result["pred_result"] = np.argmax(probs, axis=1)
    result["pred_result"] = result["pred_result"].map(inv_map)

    output_path = OUTPUTS_DIR / "predictions_2026.csv"
    result.to_csv(output_path, index=False)
    print(f"\n🔮 预测完成！结果已保存至 {output_path}")
    return result


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.kaggle_loader import load_results, add_result_label, filter_world_cup
    from src.data.scraper import fetch_elo_ratings
    from src.features.builder import build_features

    # 1. 加载数据
    matches = load_results()
    matches = add_result_label(matches)
    matches = filter_world_cup(matches)
    elo = fetch_elo_ratings()

    # 2. 构建特征
    df = build_features(matches, elo)

    # 3. 训练
    X, y = prepare_xy(df)
    lr_model = train_and_evaluate(X, y, model_name="logistic")
    xgb_model = train_and_evaluate(X, y, model_name="xgboost")

    # 4. 示例预测（用训练数据中最后10场模拟）
    sample = df.dropna(subset=FEATURE_COLS).tail(10)
    preds = predict_matches(xgb_model, sample)
    print(preds[["home_team", "away_team", "pred_prob_W", "pred_prob_D",
                  "pred_prob_L", "pred_result"]].to_string())

