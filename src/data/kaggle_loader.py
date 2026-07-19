"""
src/data/kaggle_loader.py
--------------------------
从 Kaggle 加载历史国际足球比赛数据。

数据集：martj42/international-football-results
包含：1872年至今的所有国际A级赛事结果

使用前需配置 Kaggle API：
  1. 去 https://www.kaggle.com/settings 生成 API Token
  2. 将 kaggle.json 放到 ~/.kaggle/kaggle.json
"""

import os
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset():
    """下载 Kaggle 数据集到 data/raw/"""
    print("📥 正在下载 Kaggle 数据集...")
    os.system(
        "kaggle datasets download -d martj42/international-football-results "
        f"-p {RAW_DIR} --unzip"
    )
    print(f"✅ 下载完成，文件保存至 {RAW_DIR}")


def load_results() -> pd.DataFrame:
    """
    加载比赛结果数据。

    返回字段：
        date, home_team, away_team, home_score, away_score,
        tournament, city, country, neutral
    """
    path = RAW_DIR / "results.csv"
    if not path.exists():
        download_kaggle_dataset()

    df = pd.read_csv(path, parse_dates=["date"])
    print(f"📊 加载了 {len(df):,} 场比赛记录（{df['date'].min().year}~{df['date'].max().year}）")
    return df


def load_shootouts() -> pd.DataFrame:
    """加载点球大战结果（用于淘汰赛平局处理）"""
    path = RAW_DIR / "shootouts.csv"
    if not path.exists():
        download_kaggle_dataset()
    return pd.read_csv(path, parse_dates=["date"])


def filter_world_cup(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 FIFA World Cup 正式赛事"""
    wc = df[df["tournament"] == "FIFA World Cup"].copy()
    print(f"🏆 世界杯比赛场次：{len(wc):,} 场")
    return wc


def filter_recent(df: pd.DataFrame, years: int = 20) -> pd.DataFrame:
    """只保留最近 N 年的比赛，减少久远数据的噪声"""
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    return df[df["date"] >= cutoff].copy()


def add_result_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加比赛结果标签（从主队视角）：
        W = 主队赢, D = 平局, L = 主队输
    """
    df = df.copy()
    df["result"] = "D"
    df.loc[df["home_score"] > df["away_score"], "result"] = "W"
    df.loc[df["home_score"] < df["away_score"], "result"] = "L"
    return df


if __name__ == "__main__":
    df = load_results()
    df = add_result_label(df)
    print(df.head())
    print("\n结果分布：")
    print(df["result"].value_counts())

