"""
src/data/scraper.py
--------------------
实时抓取两类数据：
  1. ELO 排名  —— eloratings.net
  2. 赔率数据  —— 通过 The Odds API（免费 tier 500次/月）

The Odds API 注册地址：https://the-odds-api.com/
免费 key 每月 500 次请求，足够世界杯赛程使用。
"""

import time
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── 替换成你自己的 key ──────────────────────────────
ODDS_API_KEY = "17bcb8e9537eca76185488e582ca1f6f"
# ───────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# 1. ELO 排名
# ─────────────────────────────────────────────────────────────

def fetch_elo_ratings() -> pd.DataFrame:
    """
    从 eloratings.net 抓取当前国家队 ELO 分。

    返回：
        DataFrame with columns [team, elo_rating]
    """
    print("🌐 正在抓取 ELO 排名...")
    url = "https://www.eloratings.net/World.tsv"
    try:
        df = pd.read_csv(url, sep="\t", header=None,
                         usecols=[2, 3],
                         names=["team", "elo_rating"])
        df = df[["team", "elo_rating"]].copy()
        output_path = RAW_DIR / "elo_ratings.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ ELO 数据：{len(df)} 支球队，已保存至 {output_path}")
        return df
    except Exception as e:
        print(f"❌ ELO 抓取失败：{e}")
        return pd.DataFrame(columns=["team", "elo_rating"])


# ─────────────────────────────────────────────────────────────
# 2. 赔率数据（The Odds API）
# ─────────────────────────────────────────────────────────────

def fetch_odds(sport: str = "soccer_fifa_world_cup") -> pd.DataFrame:
    """
    通过 The Odds API 获取世界杯赔率。

    返回字段：
        match_id, home_team, away_team, bookmaker,
        odds_home, odds_draw, odds_away, commence_time
    """
    print("🎲 正在抓取赔率数据...")
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        f"?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ 赔率抓取失败：{e}")
        return pd.DataFrame()

    rows = []
    for match in data:
        for bookmaker in match.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                rows.append({
                    "match_id": match["id"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "commence_time": match["commence_time"],
                    "bookmaker": bookmaker["key"],
                    "odds_home": outcomes.get(match["home_team"]),
                    "odds_draw": outcomes.get("Draw"),
                    "odds_away": outcomes.get(match["away_team"]),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️  没有获取到赔率数据（可能赛程未开放）")
        return df

    # 取各赔率网站的均值，减少单一庄家偏差
    df = (
        df.groupby(["match_id", "home_team", "away_team", "commence_time"])
        [["odds_home", "odds_draw", "odds_away"]]
        .mean()
        .reset_index()
    )

    output_path = RAW_DIR / "odds.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ 赔率数据：{len(df)} 场比赛，已保存至 {output_path}")
    return df


def odds_to_implied_prob(df: pd.DataFrame) -> pd.DataFrame:
    """
    将赔率转换为归一化隐含概率（去除庄家抽水）。

    公式：
        raw_prob = 1 / odds
        margin   = sum(raw_probs)   # 通常 > 1，即庄家优势
        fair_prob = raw_prob / margin
    """
    df = df.copy()
    df["raw_home"] = 1 / df["odds_home"]
    df["raw_draw"] = 1 / df["odds_draw"]
    df["raw_away"] = 1 / df["odds_away"]
    df["margin"] = df["raw_home"] + df["raw_draw"] + df["raw_away"]

    df["prob_home"] = df["raw_home"] / df["margin"]
    df["prob_draw"] = df["raw_draw"] / df["margin"]
    df["prob_away"] = df["raw_away"] / df["margin"]

    return df.drop(columns=["raw_home", "raw_draw", "raw_away", "margin"])


if __name__ == "__main__":
    elo_df = fetch_elo_ratings()
    print(elo_df.head())

    odds_df = fetch_odds()
    if not odds_df.empty:
        odds_df = odds_to_implied_prob(odds_df)
        print(odds_df.head())

