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

# 参議院の過去当選回数ごとの当選率を計算
win_rate_by_count = df.groupby("参議院の当選回数")["当落フラグ"].mean().sort_index()

# グラフ描画
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=win_rate_by_count.index, y=win_rate_by_count.values, palette="viridis", ax=ax)

# y軸をパーセント表示
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# y軸範囲を少し広げて上側にスペースを作る
ax.set_ylim(0, 1.1)

# x軸とy軸ラベル
ax.set_xlabel("参議院選挙での過去当選回数", fontsize=12)
ax.set_ylabel("当選確率 (%)", fontsize=12)

# 棒グラフの値ラベルを追加
for container in ax.containers:
    labels = [f"{v.get_height()*100:.1f}%" for v in container]
    ax.bar_label(container, labels=labels, padding=2, fontsize=9)

# タイトル設定（上側マージンを追加するため pad を指定）
plt.title("参議院選挙での過去当選回数と当選確率との関係性", fontsize=14, pad=30)

# レイアウト調整
plt.tight_layout()

plt.show()
