"""
src/evaluation/live_tracker.py
-------------------------------
实时追踪预测准确性，并自动迭代优化模型。

工作流：
1. 从 API 获取最新比赛结果
2. 对比预测 vs 真实结果
3. 计算准确率、Brier Score、Log-loss
4. 分析哪些类型的比赛预测偏差大
5. 用新数据重新训练模型
"""

import json
import pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"
TRACKING_FILE = OUTPUTS_DIR / "prediction_tracking.json"


def fetch_live_results():
    """
    从 The Odds API 获取已完成的比赛结果。
    需要 API key（在 scraper.py 中配置）。
    """
    from src.data.scraper import ODDS_API_KEY
    
    url = (
        f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores"
        f"?apiKey={ODDS_API_KEY}&daysFrom=7"
    )
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ 获取实时结果失败: {e}")
        return pd.DataFrame()
    
    results = []
    for match in data:
        if match.get("completed"):
            results.append({
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "home_score": match["scores"][0]["score"],
                "away_score": match["scores"][1]["score"],
                "commence_time": match["commence_time"],
            })
    
    df = pd.DataFrame(results)
    if not df.empty:
        df["actual_result"] = df.apply(
            lambda r: "W" if r["home_score"] > r["away_score"] 
            else ("L" if r["home_score"] < r["away_score"] else "D"),
            axis=1
        )
    return df


def load_predictions():
    """加载之前的预测"""
    path = OUTPUTS_DIR / "predictions_2026.csv"
    if not path.exists():
        print("❌ 预测文件不存在，请先运行 trainer.py")
        return pd.DataFrame()
    return pd.read_csv(path)


def evaluate_predictions(actual_df, predictions_df):
    """
    对比预测 vs 真实结果。
    
    返回：
    - accuracy: 预测结果是否正确（W/D/L）
    - log_loss: 概率预测质量
    - brier_score: 概率校准程度
    - detailed_df: 每场比赛的详细对比
    """
    # Merge predictions with actual results
    merged = actual_df.merge(
        predictions_df,
        on=["home_team", "away_team"],
        how="inner",
        suffixes=("", "_pred")
    )
    
    if merged.empty:
        print("⚠️  没有匹配的预测-实际结果对")
        return {}
    
    # Map labels to integers
    label_map = {"W": 0, "D": 1, "L": 2}
    merged["actual_int"] = merged["actual_result"].map(label_map)
    merged["pred_int"] = merged["pred_result"].map(label_map)
    
    # 1. Accuracy (结果是否正确)
    accuracy = accuracy_score(merged["actual_int"], merged["pred_int"])
    
    # 2. Log-loss (概率预测质量)
    y_true = merged["actual_int"].values
    y_proba = merged[["pred_prob_W", "pred_prob_D", "pred_prob_L"]].values
    ll = log_loss(y_true, y_proba)
    
    # 3. Brier Score (概率校准)
    brier_scores = []
    for idx, name in label_map.items():
        y_bin = (y_true == idx).astype(int)
        brier_scores.append(brier_score_loss(y_bin, y_proba[:, idx]))
    avg_brier = np.mean(brier_scores)
    
    # 4. Detailed analysis
    merged["correct"] = merged["actual_int"] == merged["pred_int"]
    merged["confidence"] = merged[["pred_prob_W", "pred_prob_D", "pred_prob_L"]].max(axis=1)
    
    # Analyze by confidence level
    high_conf = merged[merged["confidence"] > 0.6]
    low_conf = merged[merged["confidence"] <= 0.6]
    
    results = {
        "accuracy": round(accuracy, 4),
        "log_loss": round(ll, 4),
        "avg_brier": round(avg_brier, 4),
        "total_matches": len(merged),
        "correct_predictions": merged["correct"].sum(),
        "high_confidence_accuracy": round(high_conf["correct"].mean(), 4) if len(high_conf) > 0 else None,
        "low_confidence_accuracy": round(low_conf["correct"].mean(), 4) if len(low_conf) > 0 else None,
        "timestamp": datetime.now().isoformat(),
    }
    
    return results, merged


def track_predictions(results, merged_df):
    """
    保存追踪记录（累积所有评估结果）。
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Load existing tracking data
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new results
    history.append(results)
    
    # Save
    with open(TRACKING_FILE, "w") as f:
        json.dump(history, f, indent=2)
    
    # Save detailed comparison
    merged_df.to_csv(OUTPUTS_DIR / "prediction_comparison.csv", index=False)
    
    print(f"✅ 追踪记录已保存: {TRACKING_FILE}")
    print(f"   详细对比: {OUTPUTS_DIR / 'prediction_comparison.csv'}")


def retrain_with_live_data(model_name="xgboost"):
    """
    用最新数据重新训练模型（迭代优化）。
    
    流程：
    1. 获取最新 ELO
    2. 获取最新赔率
    3. 重新构建特征
    4. 重新训练
    """
    from src.data.scraper import fetch_elo_ratings, fetch_odds, odds_to_implied_prob
    from src.data.kaggle_loader import load_results, add_result_label, filter_world_cup
    from src.features.builder import build_features
    from src.models.trainer import prepare_xy, train_and_evaluate
    
    print("\n🔄 开始用实时数据重新训练模型...")
    
    # 1. 获取最新数据
    elo = fetch_elo_ratings()
    odds = fetch_odds()
    if not odds.empty:
        odds = odds_to_implied_prob(odds)
    
    # 2. 加载历史数据 + 最新结果
    matches = load_results()
    matches = add_result_label(matches)
    matches = filter_world_cup(matches)
    
    # 3. 构建特征
    df = build_features(matches, elo, odds)
    
    # 4. 重新训练
    X, y = prepare_xy(df)
    model = train_and_evaluate(X, y, model_name=model_name)
    
    print(f"✅ 模型已更新: {model_name}")
    return model


def generate_accuracy_report():
    """
    生成预测报告（可视化）。
    """
    if not TRACKING_FILE.exists():
        print("⚠️  没有追踪记录")
        return
    
    with open(TRACKING_FILE, "r") as f:
        history = json.load(f)
    
    if not history:
        print("⚠️  追踪记录为空")
        return
    
    print("\n" + "="*60)
    print("📊 预测准确性追踪报告")
    print("="*60)
    
    for i, record in enumerate(history, 1):
        print(f"\n📅 评估 #{i} ({record.get('timestamp', 'N/A')[:10]})")
        print(f"   准确率: {record['accuracy']*100:.1f}% ({record['correct_predictions']}/{record['total_matches']})")
        print(f"   Log-loss: {record['log_loss']:.4f} (越小越好，随机≈1.10)")
        print(f"   Brier Score: {record['avg_brier']:.4f} (越小越好，随机≈0.67)")
        
        if record.get("high_confidence_accuracy"):
            print(f"   高置信度准确率: {record['high_confidence_accuracy']*100:.1f}%")
        if record.get("low_confidence_accuracy"):
            print(f"   低置信度准确率: {record['low_confidence_accuracy']*100:.1f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--retrain":
        # 重新训练模式
        retrain_with_live_data()
    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        # 报告模式
        generate_accuracy_report()
    else:
        # 默认：评估当前预测
        print("🔍 获取最新比赛结果...")
        actual = fetch_live_results()
        
        if actual.empty:
            print("⚠️  没有新的比赛结果")
            sys.exit(0)
        
        print(f"✅ 找到 {len(actual)} 场已完成的比赛\n")
        
        print("📊 加载预测...")
        predictions = load_predictions()
        
        if predictions.empty:
            sys.exit(1)
        
        print("📈 评估预测准确性...")
        results, merged = evaluate_predictions(actual, predictions)
        
        if results:
            print("\n" + "="*60)
            print("📊 评估结果")
            print("="*60)
            print(f"准确率: {results['accuracy']*100:.1f}%")
            print(f"Log-loss: {results['log_loss']:.4f}")
            print(f"Brier Score: {results['avg_brier']:.4f}")
            print(f"正确预测: {results['correct_predictions']}/{results['total_matches']}")
            print("="*60)
            
            track_predictions(results, merged)
            
            # Ask if user wants to retrain
            if results['accuracy'] < 0.6:  # 准确率低于60%
                print("\n⚠️  准确率较低，建议重新训练模型")
                print("运行: python -m src.evaluation.live_tracker --retrain")
