import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency

sns.set(font=["Meiryo"])

# ==================================================
# データ読み込み
# ==================================================
df = pd.read_excel(
    "./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx",
    engine="openpyxl"
)

# 当落を数値化（当選=1, 落選=0）
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# ==================================================
# カイ二乗検定
# ==================================================
# クロス集計表（地方議会経験 × 当落）
contingency = pd.crosstab(
    df["地方議会経験フラグ"],
    df["当落フラグ"]
)

print("▼ クロス集計表（地方議会経験 × 当落）")
print(contingency)

# カイ二乗検定
chi2, p, dof, expected = chi2_contingency(contingency)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（地方議会経験と当選は独立ではない）")
else:
    print("→ 有意差なし（地方議会経験と当選は独立とみなせる）")

# ==================================================
# 当選率の可視化（既存処理）
# ==================================================
win_rate_by_local_council = (
    df.groupby("地方議会経験フラグ")["当落フラグ"]
    .mean()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x=win_rate_by_local_council.index,
    y=win_rate_by_local_council.values,
    palette="coolwarm",
    ax=ax
)

ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.1)
ax.set_xticklabels(["地方議会経験あり", "地方議会経験なし"])
ax.set_xlabel("地方議会経験の有無", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=10)

plt.title("地方議会経験の有無と当選確率の関係", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
