import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import os
import re
from scipy.stats import chi2_contingency

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
df["区数"] = df["都道府県"].str.extract(r"\((\d+)\)").astype(int)
df["都道府県名"] = df["都道府県"].str.replace(r"\(\d+\)", "", regex=True)

# --- 欠損除外 ---
df = df.dropna(subset=["都道府県名", "区数", "当落フラグ"])

# =================================================
# カイ二乗検定
# =================================================

# クロス集計表（区数 × 当落）
contingency = pd.crosstab(df["区数"], df["当落フラグ"])

chi2, p, dof, expected = chi2_contingency(contingency)

print("=== カイ二乗検定結果 ===")
print("クロス集計表（区数 × 当落）")
print(contingency)
print(f"\nカイ二乗統計量 = {chi2:.3f}")
print(f"p値 = {p:.3f}")
print(f"自由度 = {dof}")
print("\n期待度数")
print(expected)

if p < 0.05:
    print("\n→ 区数と当落には統計的に有意な関係があります（p < 0.05）")
else:
    print("\n→ 区数と当落に統計的に有意な関係は確認できません（p ≥ 0.05）")

# =================================================
# グラフ描画
# =================================================

pref_win_rate = (
    df
    .groupby(["都道府県名", "区数"])["当落フラグ"]
    .agg(mean="mean", count="count")
    .reset_index()
    .sort_values("mean", ascending=False)
)

plt.figure(figsize=(16, 8))

sns.barplot(
    data=pref_win_rate,
    x="都道府県名",
    y="mean",
    hue="区数",
    dodge=False,
    palette="Set2",
    width=0.75
)

plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.ylabel("当選確率", fontsize=12)
plt.xlabel("都道府県", fontsize=12)
plt.title("都道府県・区数別 当選確率", fontsize=15)

plt.xticks(rotation=45, ha="right")
plt.legend(title="議席数", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()
