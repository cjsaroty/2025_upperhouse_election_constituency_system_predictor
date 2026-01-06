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

secretary_col = "秘書経験フラグ"

# ==================================================
# 3. クロス集計（秘書経験 × 当落）
# ==================================================
contingency = pd.crosstab(df[secretary_col], df["当落"])

print("\n▼ クロス集計（秘書経験 × 当落）")
print(contingency)

# ==================================================
# 4. カイ二乗検定
# ==================================================
chi2, p, dof, expected = chi2_contingency(contingency)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（秘書経験の有無と当落は独立ではない）")
else:
    print("→ 有意差なし（秘書経験の有無と当落は独立とみなせる）")

# 期待度数（参考）
expected_df = pd.DataFrame(
    expected,
    index=contingency.index,
    columns=contingency.columns
)

print("\n▼ 期待度数（5未満が多い場合は解釈に注意）")
print(expected_df.round(2))

# ==================================================
# 5. 秘書経験ごとの当選率
# ==================================================
win_rate_by_secretary = (
    df.groupby(secretary_col)["当落フラグ"]
    .mean()
    .sort_index()
)

# ==================================================
# 6. 棒グラフ描画
# ==================================================
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(
    x=win_rate_by_secretary.index,
    y=win_rate_by_secretary.values,
    ax=ax
)

ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.1)

ax.set_xticklabels(["秘書経験あり", "秘書経験なし"])
ax.set_xlabel("秘書経験の有無", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# 値ラベル
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=10)

plt.title("秘書経験の有無と当選確率の関係", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
