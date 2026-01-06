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

# ==================================================
# 3. クロス集計（参議院当選回数 × 当落）
# ==================================================
cross_tab = pd.crosstab(df["参議院当選回数"], df["当落"])

print("\n▼ クロス集計（参議院当選回数 × 当落）")
print(cross_tab)

# ==================================================
# 4. カイ二乗検定
# ==================================================
chi2, p, dof, expected = chi2_contingency(cross_tab)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（参議院当選回数と当落は独立ではない）")
else:
    print("→ 有意差なし（参議院当選回数と当落は独立とみなせる）")

# 期待度数（確認用）
expected_df = pd.DataFrame(
    expected,
    index=cross_tab.index,
    columns=cross_tab.columns
)

print("\n▼ 期待度数（5未満が多い場合は注意）")
print(expected_df.round(2))

# ==================================================
# 5. 当選率の計算（可視化用）
# ==================================================
win_rate_by_count = (
    df.groupby("参議院当選回数")["当落フラグ"]
      .mean()
      .sort_index()
)

# ==================================================
# 6. 棒グラフ描画（当選確率）
# ==================================================
fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    x=win_rate_by_count.index,
    y=win_rate_by_count.values,
    palette="viridis",
    ax=ax
)

ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.1)

ax.set_xlabel("参議院選挙での過去当選回数", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# 値ラベル
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=9)

plt.title(
    "参議院選挙での過去当選回数と当選確率との関係性",
    fontsize=14,
    pad=30
)

plt.tight_layout()
plt.show()
