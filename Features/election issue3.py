import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency

sns.set(font=["Meiryo"])

# ==================================================
# 1. データ読み込み
# ==================================================
df = pd.read_excel(
    "./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx",
    engine="openpyxl"
)

# ==================================================
# 2. 当落を数値化（当選=1, 落選=0）
# ==================================================
df["当落フラグ"] = df["当落"].map({
    "当選": 1,
    "落選": 0,
    "当": 1,
    "落": 0
})

# 欠損除外（安全対策）
df = df.dropna(subset=["争点3位", "当落フラグ"])

# ==================================================
# 3. 争点1位ごとの当選確率・人数
# ==================================================
issue1_stats = (
    df.groupby("争点3位")["当落フラグ"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "当選確率", "count": "人数"})
      .sort_values("当選確率", ascending=False)
)

print("\n▼ 争点3位ごとの当選確率・人数")
print(issue1_stats)

# ==================================================
# 4. 可視化（争点1位 × 当選確率）
# ==================================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    x=issue1_stats.index,
    y=issue1_stats["当選確率"],
    ax=ax,
    palette="viridis"
)

ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.05)
ax.set_xlabel("争点3位の政策", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)
ax.set_title("争点3位と当選確率の関係", fontsize=14, pad=15)

# 値ラベル
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=10)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# ==================================================
# 5. カイ二乗検定（争点1位 × 当落）
# ==================================================
contingency = pd.crosstab(df["争点3位"], df["当落フラグ"])

print("\n▼ クロス集計表（争点3位 × 当落）")
print(contingency)

chi2, p, dof, expected = chi2_contingency(contingency)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（争点3位と当選は独立ではない）")
else:
    print("→ 有意差なし（争点3位と当選は独立とみなせる）")

# ==================================================
# 6. 期待度数の確認（重要）
# ==================================================
expected_df = pd.DataFrame(
    expected,
    index=contingency.index,
    columns=contingency.columns
)

print("\n▼ 期待度数（5未満が多い場合は注意）")
print(expected_df.round(2))
