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
# 2. クロス集計（元現新 × 当落）
# ==================================================
status = pd.crosstab(df["元現新"], df["当落"])

print("\n▼ クロス集計（元現新 × 当落）")
print(status)

# ==================================================
# 3. カイ二乗検定
# ==================================================
chi2, p, dof, expected = chi2_contingency(status)

print("\n▼ カイ二乗検定結果")
print(f"χ²値 = {chi2:.3f}")
print(f"自由度 = {dof}")
print(f"p値 = {p:.5f}")

alpha = 0.05
if p < alpha:
    print("→ 有意差あり（元現新と当落は独立ではない）")
else:
    print("→ 有意差なし（元現新と当落は独立とみなせる）")

# 期待度数（参考）
expected_df = pd.DataFrame(
    expected,
    index=status.index,
    columns=status.columns
)

print("\n▼ 期待度数（5未満が多い場合は注意）")
print(expected_df.round(2))

# ==================================================
# 4. 割合に変換（当選率・落選率）
# ==================================================
status_norm = status.div(status.sum(axis=1), axis=0)

# ==================================================
# 5. 積み上げ棒グラフ
# ==================================================
status_norm.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6),
    color=["skyblue", "salmon"]
)

plt.title("元現新と当落の関係", fontsize=14)
plt.ylabel("割合")
plt.xlabel("元現新")
plt.legend(title="当落")
plt.xticks(rotation=0)

plt.ylim(0, 1.1)
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{int(y*100)}%")
)

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sns.set(font=["Meiryo"])

