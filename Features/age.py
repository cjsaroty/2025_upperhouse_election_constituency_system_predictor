import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency  # 追加

sns.set(font=["Meiryo"])

# Excelファイル読み込み
file_path = "Data/2025_upperhouse_election_constituency_system_cleaning.xlsx"
df = pd.read_excel(file_path)

# 当落列の正規化
df["当落"] = df["当落"].astype(str).str.strip()

# 当選フラグ（当選=1、それ以外=0）
df["当選フラグ"] = df["当落"].isin(["当選", "当"]).astype(int)

# 年齢を30～90歳に制限
df = df[(df["年齢"] >= 30) & (df["年齢"] <= 90)]

# ========================
# ▼ カイ二乗検定
# 年齢を10歳刻みの年代に変換
df["年代"] = (df["年齢"] // 10) * 10

# クロス集計表（年代 × 当選）
contingency = pd.crosstab(df["年代"], df["当選フラグ"])
print("▼ クロス集計表（年代 × 当選）")
print(contingency)

# χ²検定
chi2, p, dof, expected = chi2_contingency(contingency)
print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（年齢層と当選は独立ではない）")
else:
    print("→ 有意差なし（年齢層と当選は独立とみなせる）")
# ========================

# 1歳幅
ages = range(30, 91)

# 年齢ごとの当選確率
summary = (
    df.groupby("年齢")["当選フラグ"]
      .mean()
      .reindex(ages)
)

# グラフ作成
fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(
    summary.index,
    summary.values,
    width=0.8,
    color="skyblue",
    edgecolor="black"
)

# 上側に余白を追加
ax.set_ylim(0, 1.05)

# 軸・タイトル
ax.set_xlabel("年齢")
ax.set_ylabel("当選確率")
ax.set_title("年齢別 当選確率（1歳幅・30〜90歳）")

ax.set_xticks(range(30, 91, 2))
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
