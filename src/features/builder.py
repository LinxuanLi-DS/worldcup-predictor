"""
src/features/builder.py
------------------------
特征工程模块：将原始数据转换为每场比赛的特征向量。

每行 = 一场比赛（从"主队视角"）
最终输出特征：
    elo_diff          主队ELO - 客队ELO
    home_form         主队近N场积分（胜=3, 平=1, 负=0）
    away_form         客队近N场积分
    home_goals_avg    主队近N场场均进球
    away_goals_avg    客队近N场场均进球
    h2h_win_rate      历史H2H主队胜率
    prob_home         赔率隐含概率（主队胜）
    prob_draw         赔率隐含概率（平）
    prob_away         赔率隐含概率（客队胜）
    rest_days_diff    主队休息天数 - 客队休息天数
    result            标签：W / D / L
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FORM_WINDOW = 10  # 近多少场计算状态


# ─────────────────────────────────────────────────────────────
# ELO 差值
# ─────────────────────────────────────────────────────────────

def add_elo_diff(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """
    添加两队 ELO 差值特征。
    ELO 差越大，强队获胜概率越高。
    """
    elo_dict = dict(zip(elo["team"], elo["elo_rating"]))
    matches = matches.copy()
    matches["elo_home"] = matches["home_team"].map(elo_dict)
    matches["elo_away"] = matches["away_team"].map(elo_dict)
    matches["elo_diff"] = matches["elo_home"] - matches["elo_away"]
    return matches


# ─────────────────────────────────────────────────────────────
# 近期状态（Form）
# ─────────────────────────────────────────────────────────────

def _team_form(team: str, before_date: pd.Timestamp,
               all_matches: pd.DataFrame, window: int = FORM_WINDOW) -> dict:
    """
    计算某队在某场比赛之前最近 window 场的状态。

    返回：points（积分总和）、goals_avg（场均进球）
    """
    # 找出该队参与的所有比赛（主场或客场）
    mask = (
        ((all_matches["home_team"] == team) | (all_matches["away_team"] == team))
        & (all_matches["date"] < before_date)
    )
    recent = all_matches[mask].sort_values("date").tail(window)

    if recent.empty:
        return {"points": np.nan, "goals_avg": np.nan}

    points, goals = 0, 0
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            gf, ga = row["home_score"], row["away_score"]
        else:
            gf, ga = row["away_score"], row["home_score"]

        goals += gf
        if gf > ga:
            points += 3
        elif gf == ga:
            points += 1

    return {
        "points": points,
        "goals_avg": round(goals / len(recent), 2),
    }


def add_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    为每场比赛添加两队的近期状态特征。
    注意：这个函数较慢（逐行计算），真实使用时可以预计算或缓存。
    """
    print("⏳ 正在计算近期状态特征（可能需要几分钟）...")
    matches = matches.copy()

    home_forms = matches.apply(
        lambda r: _team_form(r["home_team"], r["date"], matches), axis=1
    )
    away_forms = matches.apply(
        lambda r: _team_form(r["away_team"], r["date"], matches), axis=1
    )

    matches["home_form"] = [f["points"] for f in home_forms]
    matches["home_goals_avg"] = [f["goals_avg"] for f in home_forms]
    matches["away_form"] = [f["points"] for f in away_forms]
    matches["away_goals_avg"] = [f["goals_avg"] for f in away_forms]
    return matches


# ─────────────────────────────────────────────────────────────
# 历史 Head-to-Head
# ─────────────────────────────────────────────────────────────

def add_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    添加历史 H2H 主队胜率特征。
    只用该场比赛之前的历史数据，避免数据泄露。
    """
    matches = matches.copy()
    h2h_rates = []

    for _, row in matches.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["date"]
        past = matches[
            (matches["date"] < date) &
            (
                ((matches["home_team"] == home) & (matches["away_team"] == away)) |
                ((matches["home_team"] == away) & (matches["away_team"] == home))
            )
        ]
        if len(past) == 0:
            h2h_rates.append(0.33)  # 无历史数据时用均匀分布
            continue

        wins = 0
        for _, p in past.iterrows():
            if p["home_team"] == home and p["result"] == "W":
                wins += 1
            elif p["away_team"] == home and p["result"] == "L":
                wins += 1
        h2h_rates.append(round(wins / len(past), 3))

    matches["h2h_win_rate"] = h2h_rates
    return matches


# ─────────────────────────────────────────────────────────────
# 赔率隐含概率
# ─────────────────────────────────────────────────────────────

def add_odds_features(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    将赔率隐含概率 join 到比赛数据中。
    odds 需要包含：home_team, away_team, prob_home, prob_draw, prob_away
    """
    merged = matches.merge(
        odds[["home_team", "away_team", "prob_home", "prob_draw", "prob_away"]],
        on=["home_team", "away_team"],
        how="left",
    )
    return merged


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "elo_diff",
    "home_form", "away_form",
    "home_goals_avg", "away_goals_avg",
    "h2h_win_rate",
    "prob_home", "prob_draw", "prob_away",
]


def build_features(matches: pd.DataFrame, elo: pd.DataFrame,
                   odds: pd.DataFrame = None) -> pd.DataFrame:
    """
    完整特征工程流程。

    参数：
        matches: 含 result 标签的比赛数据
        elo:     ELO 评分表
        odds:    赔率隐含概率表（可选）
    """
    df = add_elo_diff(matches, elo)
    df = add_form_features(df)
    df = add_h2h_features(df)
    if odds is not None and not odds.empty:
        df = add_odds_features(df, odds)

    output_path = PROCESSED_DIR / "features.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ 特征构建完成，已保存至 {output_path}")
    print(f"   总样本数：{len(df):,}，特征列：{FEATURE_COLS}")
    return df


if __name__ == "__main__":
    from src.data.kaggle_loader import load_results, add_result_label, filter_world_cup
    from src.data.scraper import fetch_elo_ratings

    matches = load_results()
    matches = add_result_label(matches)
    matches = filter_world_cup(matches)
    elo = fetch_elo_ratings()
    df = build_features(matches, elo)
    print(df[FEATURE_COLS + ["result"]].dropna().head())

