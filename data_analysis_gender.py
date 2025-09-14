import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font=["Meiryo"])

df = pd.read_excel("sangiinn_2025_cleaning.xlsx", engine="openpyxl")

# クロス集計
cross_tab = pd.crosstab(df["性別"], df["当落"])

# 当選率を計算 (%)
cross_tab["当選率(%)"] = cross_tab["当"] / cross_tab.sum(axis=1) * 100

# 当選率の棒グラフ
cross_tab["当選率(%)"].plot(
    kind="bar",
    color=["skyblue", "salmon"],
    figsize=(6,4)
)

plt.ylabel("当選率 (%)")
plt.xlabel("性別")
plt.xticks([0,1], ["男性", "女性"], rotation=0)
plt.ylim(0,100)
plt.title("男女ごとの当選率比較")
plt.tight_layout()
plt.show()