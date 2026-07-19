#!/usr/bin/env python3
"""
evaluate_and_improve.py
-----------------------
完整流程：拉取最新数据 → 评估预测 → 迭代优化

用法：
    python3 evaluate_and_improve.py          # 评估当前预测
    python3 evaluate_and_improve.py --train  # 评估 + 重新训练
"""

import sys
import os
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import urllib3
urllib3.disable_warnings()

sys.path.insert(0, str(Path(__file__).parent))

from src.data.scraper import ODDS_API_KEY

# Team code -> full name mapping (eloratings uses ISO codes)
ELO_CODE_MAP = {
    "ES": "Spain", "AR": "Argentina", "FR": "France", "EN": "England",
    "PT": "Portugal", "CO": "Colombia", "BR": "Brazil", "NL": "Netherlands",
    "DE": "Germany", "EC": "Ecuador", "HR": "Croatia", "UY": "Uruguay",
    "MX": "Mexico", "US": "USA", "CH": "Switzerland", "MA": "Morocco",
    "SN": "Senegal", "JP": "Japan", "KR": "South Korea", "NO": "Norway",
    "AT": "Austria", "AU": "Australia", "CA": "Canada", "TR": "Turkey",
    "CZ": "Czech Republic", "SE": "Sweden", "GH": "Ghana", "CI": "Ivory Coast",
    "PY": "Paraguay", "TN": "Tunisia", "BA": "Bosnia & Herzegovina",
    "ZA": "South Africa", "IR": "Iran", "DZ": "Algeria", "NZ": "New Zealand",
    "PA": "Panama", "SA": "Saudi Arabia", "UZ": "Uzbekistan", "JO": "Jordan",
    "EG": "Egypt", "CD": "DR Congo", "IQ": "Iraq", "CV": "Cape Verde",
    "QA": "Qatar", "HT": "Haiti", "CW": "Curaçao", "BE": "Belgium",
    "SC": "Scotland",
}

# Current App.js groups
GROUPS = {
    "A": ["Mexico","South Africa","South Korea","Czech Republic"],
    "B": ["Canada","Bosnia & Herzegovina","Qatar","Switzerland"],
    "C": ["Brazil","Morocco","Haiti","Scotland"],
    "D": ["USA","Paraguay","Australia","Turkey"],
    "E": ["Germany","Curaçao","Ivory Coast","Ecuador"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Egypt","Iran","New Zealand"],
    "H": ["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    "I": ["France","Senegal","Iraq","Norway"],
    "J": ["Argentina","Algeria","Austria","Jordan"],
    "K": ["Portugal","DR Congo","Uzbekistan","Colombia"],
    "L": ["England","Croatia","Ghana","Panama"],
}


def fetch_scores():
    """获取已完成的比赛结果"""
    all_matches = []
    for days in [1, 2, 3]:
        url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores"
        params = {"apiKey": ODDS_API_KEY, "daysFrom": days}
        try:
            r = requests.get(url, params=params, timeout=15, verify=False)
            if r.status_code == 200:
                all_matches.extend(r.json())
        except:
            pass

    # Deduplicate
    seen = set()
    unique = []
    for m in all_matches:
        mid = m.get("id", "")
        if mid not in seen:
            seen.add(mid)
            unique.append(m)

    completed = [m for m in unique if m.get("completed")]
    results = []
    for m in completed:
        hs = int(m["scores"][0]["score"])
        as_ = int(m["scores"][1]["score"])
        results.append({
            "home": m["home_team"],
            "away": m["away_team"],
            "home_score": hs,
            "away_score": as_,
            "date": m.get("commence_time", "")[:10],
            "actual": "W" if hs > as_ else ("L" if hs < as_ else "D"),
        })
    return results


def fetch_odds():
    """获取赔率数据"""
    url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    try:
        r = requests.get(url, params=params, timeout=15, verify=False)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []


def get_predictions():
    """从 App.js 提取硬编码的预测数据"""
    app_js = Path(__file__).parent / "src" / "App.js"
    content = app_js.read_text()

    # Extract ODDS_DATA
    import re
    match = re.search(r"const ODDS_DATA = (\[[\s\S]*?\]);", content)
    if not match:
        return []

    odds_str = match.group(1)
    # Parse JS array to Python
    odds_str = odds_str.replace("{home:", '{"home":')
    odds_str = odds_str.replace(",away:", ',"away":')
    odds_str = odds_str.replace(",ph:", ',"ph":')
    odds_str = odds_str.replace(",pd:", ',"pd":')
    odds_str = odds_str.replace(",pa:", ',"pa":')
    odds_str = odds_str.replace("}", "}")
    # Fix trailing commas
    import re as re2
    odds_str = re2.sub(r",\s*]", "]", odds_str)
    odds_str = re2.sub(r",\s*}", "}", odds_str)

    predictions = json.loads(odds_str)
    return predictions


def evaluate(predictions, actual_results):
    """评估预测准确性"""
    # Build lookup
    pred_map = {}
    for p in predictions:
        key = (p["home"], p["away"])
        pred_map[key] = p
        # Also reverse
        key_r = (p["away"], p["home"])
        pred_map[key_r] = {
            "home": p["away"], "away": p["home"],
            "ph": p["pa"], "pd": p["pd"], "pa": p["ph"]
        }

    correct = 0
    total = 0
    log_loss_sum = 0
    details = []

    for r in actual_results:
        key = (r["home"], r["away"])
        pred = pred_map.get(key)
        if not pred:
            continue

        total += 1
        actual = r["actual"]  # W/D/L from home perspective

        # Predicted result
        if pred["ph"] > pred["pd"] and pred["ph"] > pred["pa"]:
            pred_result = "W"
        elif pred["pa"] > pred["ph"] and pred["pa"] > pred["pd"]:
            pred_result = "L"
        else:
            pred_result = "D"

        is_correct = (pred_result == actual)
        if is_correct:
            correct += 1

        # Log loss
        label_map = {"W": 0, "D": 1, "L": 2}
        actual_idx = label_map[actual]
        probs = [pred["ph"], pred["pd"], pred["pa"]]
        # Clip to avoid log(0)
        probs = [max(p, 0.01) for p in probs]
        ll = -np.log(probs[actual_idx])
        log_loss_sum += ll

        confidence = max(pred["ph"], pred["pd"], pred["pa"])
        details.append({
            "home": r["home"], "away": r["away"],
            "score": f"{r['home_score']}-{r['away_score']}",
            "date": r["date"],
            "actual": actual,
            "predicted": pred_result,
            "correct": is_correct,
            "confidence": round(confidence, 3),
            "pred_probs": f"W:{pred['ph']:.0%} D:{pred['pd']:.0%} L:{pred['pa']:.0%}",
        })

    if total == 0:
        return {"message": "没有匹配的预测-实际结果对"}

    accuracy = correct / total
    avg_log_loss = log_loss_sum / total

    # Confidence analysis
    high_conf = [d for d in details if d["confidence"] > 0.5]
    low_conf = [d for d in details if d["confidence"] <= 0.5]

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_log_loss": round(avg_log_loss, 4),
        "high_conf_total": len(high_conf),
        "high_conf_correct": sum(1 for d in high_conf if d["correct"]),
        "high_conf_accuracy": round(sum(1 for d in high_conf if d["correct"]) / max(len(high_conf), 1), 4),
        "low_conf_total": len(low_conf),
        "low_conf_correct": sum(1 for d in low_conf if d["correct"]),
        "details": sorted(details, key=lambda x: x["date"]),
    }


def main():
    print("=" * 60)
    print("⚽ 世界杯预测系统 - 评估与迭代")
    print("=" * 60)

    # 1. 获取比赛结果
    print("\n📡 获取最新比赛结果...")
    results = fetch_scores()
    print(f"   已完成比赛: {len(results)} 场")

    if not results:
        print("   暂无已完成的比赛结果")
        return

    for r in sorted(results, key=lambda x: x["date"]):
        print(f"   {r['date']} {r['home']} {r['home_score']}-{r['away_score']} {r['away']}")

    # 2. 获取预测
    print("\n📊 加载预测数据...")
    predictions = get_predictions()
    print(f"   预测场次: {len(predictions)}")

    # 3. 评估
    print("\n📈 评估预测准确性...")
    evaluation = evaluate(predictions, results)

    if "message" in evaluation:
        print(f"   ⚠️ {evaluation['message']}")
        return

    print(f"\n{'='*60}")
    print(f"📊 评估结果")
    print(f"{'='*60}")
    print(f"   总场次: {evaluation['total']}")
    print(f"   正确预测: {evaluation['correct']}/{evaluation['total']}")
    print(f"   准确率: {evaluation['accuracy']*100:.1f}%")
    print(f"   平均 Log-loss: {evaluation['avg_log_loss']:.4f}")
    print(f"   (随机猜测 log-loss ≈ 1.10，好模型 < 1.0)")

    if evaluation["high_conf_total"] > 0:
        print(f"\n   高置信度 (>50%): {evaluation['high_conf_correct']}/{evaluation['high_conf_total']} = {evaluation['high_conf_accuracy']*100:.1f}%")
    if evaluation["low_conf_total"] > 0:
        low_acc = evaluation["low_conf_correct"] / max(evaluation["low_conf_total"], 1)
        print(f"   低置信度 (≤50%): {evaluation['low_conf_correct']}/{evaluation['low_conf_total']} = {low_acc*100:.1f}%")

    # 4. 详细对比
    print(f"\n{'='*60}")
    print("📋 逐场对比")
    print(f"{'='*60}")
    for d in evaluation["details"]:
        mark = "✅" if d["correct"] else "❌"
        print(f"   {mark} {d['date']} {d['home']} {d['score']} {d['away']}")
        print(f"      实际: {d['actual']} | 预测: {d['predicted']} | {d['pred_probs']}")

    # 5. 分析偏差
    wrong = [d for d in evaluation["details"] if not d["correct"]]
    if wrong:
        print(f"\n{'='*60}")
        print("🔍 错误预测分析")
        print(f"{'='*60}")
        for d in wrong:
            print(f"   ❌ {d['home']} {d['score']} {d['away']}")
            print(f"      预测了 {d['predicted']}，实际是 {d['actual']}")
            print(f"      概率: {d['pred_probs']}")

    # 6. 保存评估结果
    os.makedirs("outputs", exist_ok=True)
    eval_file = Path("outputs/evaluation.json")

    # Load existing history
    history = []
    if eval_file.exists():
        with open(eval_file) as f:
            history = json.load(f)

    eval_record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total": evaluation["total"],
        "correct": evaluation["correct"],
        "accuracy": evaluation["accuracy"],
        "avg_log_loss": evaluation["avg_log_loss"],
    }
    history.append(eval_record)
    with open(eval_file, "w") as f:
        json.dump(history, f, indent=2)

    # Save detailed results
    with open("outputs/live_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评估结果已保存到 outputs/evaluation.json")

    # 7. Retrain if requested
    if "--train" in sys.argv and evaluation["accuracy"] < 0.7:
        print(f"\n🔄 准确率 < 70%，建议重新训练...")
        print("   运行: python3 -m src.models.trainer")


if __name__ == "__main__":
    main()
