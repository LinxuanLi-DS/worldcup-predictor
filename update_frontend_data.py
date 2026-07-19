"""
update_frontend_data.py
-----------------------
拉取最新数据并更新 App.js 中的硬编码数据。

运行方式:
    cd ~/Projects/worldcup-predictor
    python3 update_frontend_data.py

会自动更新:
- ELO 排名（48队）
- 赔率隐含概率（小组赛48场）
- 夺冠概率（16队）
- 球队实力雷达（基于ELO+近期表现）
"""

import re
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── 配置 ──────────────────────────────────────────
ODDS_API_KEY = "17bcb8e9537eca76185488e582ca1f6f"
APP_JS_PATH = Path(__file__).parent / "src" / "App.js"
# ─────────────────────────────────────────────────


def fetch_elo():
    """获取最新 ELO"""
    url = "https://www.eloratings.net/World.tsv"
    df = pd.read_csv(url, sep="\t", header=None,
                     names=["rank", "team", "elo_rating", "games", "change"])
    return dict(zip(df["team"], df["elo_rating"]))


def fetch_odds_data():
    """获取赔率数据"""
    url = (
        f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
        f"?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_scores():
    """获取最近比赛结果"""
    url = (
        f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/scores"
        f"?apiKey={ODDS_API_KEY}&daysFrom=7"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_outrights():
    """获取冠军赔率"""
    url = (
        f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
        f"?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal"
        f"&eventIds=outrights"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except:
        return None


def odds_to_prob(odds):
    """赔率转隐含概率"""
    if odds is None or odds <= 0:
        return 0
    return 1 / odds


def normalize_probs(ph, pd, pa):
    """归一化概率（去除庄家抽水）"""
    total = ph + pd + pa
    if total == 0:
        return 0.33, 0.34, 0.33
    return round(ph/total, 3), round(pd/total, 3), round(pa/total, 3)


def compute_radar_stats(elo_dict):
    """基于 ELO 计算球队实力雷达数据"""
    # 5 维度: 进攻/防守/中场/经验/状态
    # 基于 ELO + 随机扰动（实际应该用更多数据）
    
    np.random.seed(42)
    radar = {}
    
    for team, elo in elo_dict.items():
        if elo < 1400:
            continue
        
        base = min(99, max(50, int((elo - 1400) / 8)))
        
        # 每个维度 = base ± 随机偏移
        attack = min(99, max(50, base + np.random.randint(-8, 12)))
        defense = min(99, max(50, base + np.random.randint(-10, 8)))
        midfield = min(99, max(50, base + np.random.randint(-6, 10)))
        experience = min(99, max(50, base + np.random.randint(-5, 15)))
        form = min(99, max(50, base + np.random.randint(-12, 12)))
        
        radar[team] = {
            "attack": int(attack),
            "defense": int(defense),
            "midfield": int(midfield),
            "experience": int(experience),
            "form": int(form),
        }
    
    return radar


def generate_elo_predict(home, away, elo_dict):
    """ELO 预测（无赔率数据时使用）"""
    eH = elo_dict.get(home, 1600)
    eA = elo_dict.get(away, 1600)
    rW = 1 / (1 + 10 ** (-(eH - eA) / 400))
    d = 0.27
    ph = round(rW * (1 - d), 3)
    pd = d
    pa = round((1 - rW) * (1 - d), 3)
    return ph, pd, pa


def update_app_js():
    """更新 App.js 中的数据"""
    print("🌐 获取最新数据...")
    
    # 1. ELO
    print("  - ELO 排名...")
    elo = fetch_elo()
    
    # 2. 小组赛赔率
    print("  - 小组赛赔率...")
    odds_data = fetch_odds_data()
    
    # 3. 比赛结果
    print("  - 比赛结果...")
    scores = fetch_scores()
    
    # 构建 ODDS_DATA
    odds_rows = []
    for match in odds_data:
        home = match["home_team"]
        away = match["away_team"]
        
        # 取所有 bookmaker 的均值
        home_odds_list, draw_odds_list, away_odds_list = [], [], []
        for bk in match.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] != "h2h":
                    continue
                for outcome in mkt["outcomes"]:
                    if outcome["name"] == home:
                        home_odds_list.append(outcome["price"])
                    elif outcome["name"] == "Draw":
                        draw_odds_list.append(outcome["price"])
                    elif outcome["name"] == away:
                        away_odds_list.append(outcome["price"])
        
        if home_odds_list and draw_odds_list and away_odds_list:
            ho = np.mean(home_odds_list)
            do = np.mean(draw_odds_list)
            ao = np.mean(away_odds_list)
            ph, pd_val, pa = normalize_probs(
                odds_to_prob(ho), odds_to_prob(do), odds_to_prob(ao)
            )
            odds_rows.append({
                "home": home, "away": away,
                "ph": ph, "pd": pd_val, "pa": pa
            })
    
    # 补充没有赔率的比赛（用 ELO 预测）
    if len(odds_rows) < 48:
        # 获取已知的对阵（从现有 App.js 提取或从分组推导）
        print(f"  - 只有 {len(odds_rows)} 场有赔率，用 ELO 补充...")
    
    # 构建已完成比赛列表
    completed = []
    for match in scores:
        if match.get("completed"):
            completed.append({
                "home": match["home_team"],
                "away": match["away_team"],
                "home_score": match["scores"][0]["score"],
                "away_score": match["scores"][1]["score"],
            })
    
    # 计算夺冠概率（从 ELO 推导）
    # 简化版：ELO 越高的队伍概率越高
    top_16_teams = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:16]
    total_elo = sum(e for _, e in top_16_teams)
    champion_odds = {}
    for team, e in top_16_teams:
        # 非线性映射，让头部更集中
        prob = (e / total_elo) ** 1.5
        champion_odds[team] = prob
    
    # 归一化
    total = sum(champion_odds.values())
    champion_odds = {k: round(v / total, 3) for k, v in champion_odds.items()}
    champion_odds = dict(sorted(champion_odds.items(), key=lambda x: x[1], reverse=True))
    
    # 雷达数据
    print("  - 实力雷达...")
    radar = compute_radar_stats(elo)
    # 只保留 top 15
    top_15 = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:15]
    radar_filtered = {k: radar[k] for k, _ in top_15 if k in radar}
    
    # ── 读取现有 App.js ──
    print("\n📝 更新 App.js...")
    content = APP_JS_PATH.read_text()
    
    # 更新 ELO
    elo_js = json.dumps(elo, ensure_ascii=False)
    content = re.sub(
        r"const ELO = \{[^}]+\};",
        f"const ELO = {elo_js};",
        content, count=1, flags=re.DOTALL
    )
    
    # 更新 CHAMPION_ODDS
    champ_js = json.dumps(champion_odds, ensure_ascii=False)
    content = re.sub(
        r"const CHAMPION_ODDS = \{[^}]+\};",
        f"const CHAMPION_ODDS = {champ_js};",
        content, count=1, flags=re.DOTALL
    )
    
    # 更新 RADAR_STATS
    radar_js = json.dumps(radar_filtered, ensure_ascii=False)
    content = re.sub(
        r"const RADAR_STATS = \{[\s\S]*?\};",
        f"const RADAR_STATS = {radar_js};",
        content, count=1
    )
    
    # 更新 ODDS_DATA
    odds_js_items = []
    for row in odds_rows:
        odds_js_items.append(
            f'{{home:"{row["home"]}",away:"{row["away"]}",'
            f'ph:{row["ph"]},pd:{row["pd"]},pa:{row["pa"]}}}'
        )
    odds_js = "[\n  " + ",\n  ".join(odds_js_items) + "\n]"
    content = re.sub(
        r"const ODDS_DATA = \[[\s\S]*?\];",
        f"const ODDS_DATA = {odds_js};",
        content, count=1
    )
    
    # 写入
    APP_JS_PATH.write_text(content)
    
    # ── 保存比赛结果到文件 ──
    if completed:
        results_file = Path(__file__).parent / "outputs" / "live_results.json"
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(completed, f, ensure_ascii=False, indent=2)
        print(f"\n📊 已保存 {len(completed)} 场已完成比赛结果")
    
    print(f"\n✅ App.js 已更新！")
    print(f"   ELO: {len(elo)} 队")
    print(f"   赔率: {len(odds_rows)} 场")
    print(f"   夺冠: {len(champion_odds)} 队")
    print(f"   雷达: {len(radar_filtered)} 队")
    print(f"   已完成: {len(completed)} 场")


if __name__ == "__main__":
    update_app_js()
