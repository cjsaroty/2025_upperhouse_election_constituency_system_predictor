import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

sns.set(font=["Meiryo"])

# データ読み込み
df = pd.read_excel(
    "C:/Users/frontier-Python/Desktop/2025_upperhouse_election_constituency_system_predictor/Data/2025_upperhouse_election_constituency_system_cleaning.xlsx",
    engine="openpyxl"
)

# 当落を数値化（当選=1, 落選=0）
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# 衆議院の当選回数を取得
df["衆議院の当選回数"] = df["衆議院の当選回数"]

# 過去当選回数ごとの当選率を計算
win_rate_by_count = df.groupby("衆議院の当選回数")["当落フラグ"].mean().sort_index()

# グラフ描画
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=win_rate_by_count.index, y=win_rate_by_count.values, palette="viridis", ax=ax)

# y軸をパーセント表示
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.1)

ax.set_xlabel("過去の衆議院当選回数", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# 棒グラフの値ラベルを追加
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=9)

# タイトル設定
plt.title("過去の衆議院当選回数と当落の関係", fontsize=14, pad=30)

# レイアウト調整
plt.tight_layout()
plt.show()
