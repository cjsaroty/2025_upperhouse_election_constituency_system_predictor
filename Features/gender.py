import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# 2. クロス集計（性別 × 当落）
# ==================================================
cross_tab = pd.crosstab(df["性別"], df["当落"])

print("\n▼ クロス集計表（性別 × 当落）")
print(cross_tab)

# ==================================================
# 3. カイ二乗検定
# ==================================================
chi2, p, dof, expected = chi2_contingency(cross_tab)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（性別と当落は独立ではない）")
else:
    print("→ 有意差なし（性別と当落は独立とみなせる）")

# 期待度数（確認用）
expected_df = pd.DataFrame(
    expected,
    index=cross_tab.index,
    columns=cross_tab.columns
)

print("\n▼ 期待度数（5未満が多い場合は注意）")
print(expected_df.round(2))

# ==================================================
# 4. 当選率を計算（%）
# ==================================================
cross_tab["当選率(%)"] = cross_tab["当"] / cross_tab.sum(axis=1) * 100

print("\n▼ 性別ごとの当選率 (%)")
print(cross_tab["当選率(%)"].round(2))

# ==================================================
# 5. 当選率の棒グラフ
# ==================================================
cross_tab["当選率(%)"].plot(
    kind="bar",
    color=["skyblue", "salmon"],
    figsize=(6, 4)
)

plt.ylabel("当選率 (%)")
plt.xlabel("性別")
plt.xticks([0, 1], ["男性", "女性"], rotation=0)
plt.ylim(0, 100)
plt.title("男女ごとの当選率比較")
plt.tight_layout()
plt.show()
