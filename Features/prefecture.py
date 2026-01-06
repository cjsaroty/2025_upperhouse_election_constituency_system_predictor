import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import os
import re

# --- 日本語フォント設定 ---
sns.set(font=["Meiryo"])

# --- データ読み込み ---
file_path = "Data/2025_upperhouse_election_constituency_system_cleaning.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} が見つかりません。")

df = pd.read_excel(file_path, engine="openpyxl")

# --- 当落を数値化 ---
df["当落フラグ"] = df["当落"].map({
    "当選": 1, "落選": 0,
    "当": 1, "落": 0
})

# --- 都道府県名と()内数字を分離 ---
df["区数"] = df["都道府県"].str.extract(r"\((\d+)\)").astype(float)
df["都道府県名"] = df["都道府県"].str.replace(r"\(\d+\)", "", regex=True)

# --- 欠損除外 ---
df = df.dropna(subset=["都道府県名", "区数", "当落フラグ"])

# --- 都道府県 × 区数 ごとの当選確率 ---
pref_win_rate = (
    df
    .groupby(["都道府県名", "区数"])["当落フラグ"]
    .agg(mean="mean", count="count")
    .reset_index()
)

# --- 並び替え（平均当選確率順） ---
pref_win_rate = pref_win_rate.sort_values("mean", ascending=False)

# --- グラフ描画 ---
plt.figure(figsize=(16, 8))

sns.barplot(
    data=pref_win_rate,
    x="都道府県名",
    y="mean",
    hue="区数",
    dodge=False,
    palette="Set2",
    width=0.75   # ← 棒を太くする
)

# --- パーセント表示 ---
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.ylabel("当選確率", fontsize=12)
plt.xlabel("都道府県", fontsize=12)
plt.title("都道府県・区数別 当選確率", fontsize=15)

# --- x軸ラベル回転 ---
plt.xticks(rotation=45, ha="right")

# --- 凡例 ---
plt.legend(title="議席数", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()
