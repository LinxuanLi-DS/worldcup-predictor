"""
src/models/improved_predictor.py
----------------------------------
改进版预测模型，针对当前问题优化：
1. 提高平局概率（小组赛阶段）
2. 降低过度自信
3. 加入世界杯小组赛特殊因子
"""

import numpy as np
import pandas as pd
from pathlib import Path


def elo_to_probability(home_elo, away_elo, is_group_stage=True):
    """
    基于 ELO 计算胜平负概率。
    
    改进点：
    - 小组赛阶段提高平局概率（历史约27-30%）
    - 使用更保守的概率分布
    """
    # ELO 期望胜率
    elo_diff = home_elo - away_elo
    expected_home = 1 / (1 + 10 ** (-elo_diff / 400))
    
    # 基础平局概率
    base_draw = 0.27
    
    # 小组赛阶段调整（更保守，平局更多）
    if is_group_stage:
        base_draw = 0.35  # 实际小组赛平局率约35-44%
        
        # 强队 vs 弱队时，平局概率降低
        if abs(elo_diff) > 250:
            base_draw = 0.25
        elif abs(elo_diff) > 150:
            base_draw = 0.30
    
    # 计算胜平负
    prob_home = expected_home * (1 - base_draw)
    prob_away = (1 - expected_home) * (1 - base_draw)
    prob_draw = base_draw
    
    # 归一化
    total = prob_home + prob_draw + prob_away
    return {
        "ph": round(prob_home / total, 3),
        "pd": round(prob_draw / total, 3),
        "pa": round(prob_away / total, 3),
    }


def predict_match(home_team, away_team, elo_dict, is_group_stage=True):
    """预测单场比赛"""
    home_elo = elo_dict.get(home_team, 1600)
    away_elo = elo_dict.get(away_team, 1600)
    
    return elo_to_probability(home_elo, away_elo, is_group_stage)


def calibrate_probabilities(ph, pd, pa, match_context=None):
    """
    概率校准：降低极端概率，增加不确定性。
    
    改进点：
    - 限制最大概率到 85%（避免过度自信）
    - 提高最低概率到 8%（给冷门留空间）
    """
    # 限制范围
    ph = np.clip(ph, 0.08, 0.85)
    pd = np.clip(pd, 0.10, 0.40)
    pa = np.clip(pa, 0.08, 0.85)
    
    # 归一化
    total = ph + pd + pa
    return {
        "ph": round(ph / total, 3),
        "pd": round(pd / total, 3),
        "pa": round(pa / total, 3),
    }


def update_app_predictions(elo_dict, groups):
    """
    更新 App.js 中的所有预测。
    """
    import re
    import json
    
    app_js_path = Path(__file__).parents[2] / "src" / "App.js"
    content = app_js_path.read_text()
    
    # 生成所有小组赛预测
    odds_data = []
    for group, teams in groups.items():
        # 每队打3场（对阵组内其他3队）
        for i, home in enumerate(teams):
            for away in teams[i+1:]:
                pred = predict_match(home, away, elo_dict, is_group_stage=True)
                pred = calibrate_probabilities(pred["ph"], pred["pd"], pred["pa"])
                odds_data.append({
                    "home": home,
                    "away": away,
                    "ph": pred["ph"],
                    "pd": pred["pd"],
                    "pa": pred["pa"],
                })
    
    # 生成 JS 数组
    odds_js_items = []
    for row in odds_data:
        odds_js_items.append(
            f'{{home:"{row["home"]}",away:"{row["away"]}",'
            f'ph:{row["ph"]},pd:{row["pd"]},pa:{row["pa"]}}}'
        )
    odds_js = "[\n  " + ",\n  ".join(odds_js_items) + "\n]"
    
    # 替换 ODDS_DATA
    content = re.sub(
        r"const ODDS_DATA = \[[\s\S]*?\];",
        f"const ODDS_DATA = {odds_js};",
        content, count=1
    )
    
    # 更新 ELO
    elo_js = json.dumps(elo_dict, ensure_ascii=False)
    content = re.sub(
        r"const ELO = \{[^}]+\};",
        f"const ELO = {elo_js};",
        content, count=1, flags=re.DOTALL
    )
    
    app_js_path.write_text(content)
    print(f"✅ 已更新 {len(odds_data)} 场预测到 App.js")
    
    return odds_data


if __name__ == "__main__":
    import sys
    import json as json_mod
    sys.path.insert(0, str(Path(__file__).parents[2]))
    
    # Load corrected ELO mapping
    mapping_path = Path(__file__).parents[2] / "outputs" / "elo_mapping.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            elo_dict = {k: int(v) for k, v in json_mod.load(f).items()}
        print(f"Loaded {len(elo_dict)} team ELO mappings")
    else:
        from src.data.scraper import fetch_elo_ratings
        elo_df = fetch_elo_ratings()
        from evaluate_and_improve import ELO_CODE_MAP
        elo_dict = {}
        for _, row in elo_df.iterrows():
            code = row["team"]
            full_name = ELO_CODE_MAP.get(code)
            if full_name:
                elo_dict[full_name] = int(row["elo_rating"])
    
    # 加载分组
    from evaluate_and_improve import GROUPS
    
    # 更新预测
    predictions = update_app_predictions(elo_dict, GROUPS)
    
    print(f"\n📊 预测示例:")
    for p in predictions[:5]:
        print(f"   {p['home']} vs {p['away']}")
        print(f"      W:{p['ph']:.0%} D:{p['pd']:.0%} L:{p['pa']:.0%}")
