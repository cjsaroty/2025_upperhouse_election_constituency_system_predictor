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
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# ==================================================
# 3. 党派 × 当落のクロス集計
# ==================================================
party_outcome = pd.crosstab(df["党派"], df["当落"])

print("\n▼ クロス集計（党派 × 当落）")
print(party_outcome)

# ==================================================
# 4. カイ二乗検定
# ==================================================
chi2, p, dof, expected = chi2_contingency(party_outcome)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（党派と当落は独立ではない）")
else:
    print("→ 有意差なし（党派と当落は独立とみなせる）")

# 期待度数（参考）
expected_df = pd.DataFrame(
    expected,
    index=party_outcome.index,
    columns=party_outcome.columns
)

print("\n▼ 期待度数（5未満が多い場合は解釈に注意）")
print(expected_df.round(2))

# ==================================================
# 5. 党派ごとの当選率
# ==================================================
party_win_rate = df.groupby("党派")["当落フラグ"].mean().sort_values(ascending=False)

# ==================================================
# 6. 棒グラフ（当選確率）
# ==================================================
fig, ax1 = plt.subplots(figsize=(12, 6))

sns.barplot(
    x=party_win_rate.index,
    y=party_win_rate.values,
    ax=ax1
)

ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax1.set_ylim(0, 1)
ax1.set_ylabel("当選確率 (%)", fontsize=12)

# 値ラベル
for container in ax1.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax1.bar_label(container, labels=labels, padding=2, fontsize=9)

plt.title("党派別当選確率", fontsize=14)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()
