import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

sns.set(font=["Meiryo"])

file_path = r"C:\Users\owner\OneDrive\デスクトップ\2025_upperhouse_election_predictor\Data\2025_upperhouse_election_constituency_system_cleaning.xlsx"

df = pd.read_excel(file_path)

#当落ごとに年齢の統計量を集計
age_stats = df.groupby("当落")["年齢"].agg(["mean", "median", "std"])
age_stats = age_stats.rename(columns={
    "mean": "平均年齢",
    "median": "中央値",
    "std": "標準偏差"})

#グラフ作成
fig, ax = plt.subplots(figsize=(6,4))

#平均年齢の棒グラフ（標準偏差をエラーバーとして表示）
ax.bar(age_stats.index, age_stats["平均年齢"], yerr=age_stats["標準偏差"], capsize=1, color=["skyblue","salmon"],width=0.3,label="平均年齢（エラーバー：平均 ±1σ（σ = 標準偏差））")

#中央値も線で表示
ax.plot(age_stats.index, age_stats["中央値"], color='green', marker='o', linestyle='--', label='中央値')

#凡例パッチを自作（skyblueとsalmon両方に同じラベル）
blue_patch = mpatches.Patch(color="skyblue", label="平均年齢（エラーバー：平均 ±1σ（σ = 標準偏差））")
salmon_patch = mpatches.Patch(color="salmon", label="平均年齢（エラーバー：平均 ±1σ（σ = 標準偏差））")
median_line = mlines.Line2D([], [], color='green', marker='o', linestyle='--', label='中央値')

ax.legend(handles=[blue_patch, salmon_patch, median_line], loc="best")

# 軸ラベル・タイトル
ax.set_ylabel("年齢")
ax.set_title("当落別年齢統計")

plt.tight_layout()
plt.show()