import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font=["Meiryo"])

df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")

#当選確率とステータス(元職・現職・新人)の関係性
status = pd.crosstab(df["元現新"], df["当落"])
print(status)

status_norm = status.div(status.sum(axis=1), axis=0)  # 行ごとに割合
status_norm.plot(kind='bar', stacked=True, figsize=(8,6), color=["skyblue", "salmon"])
plt.title("元現新と当落の関係", fontsize=14)
plt.ylabel("当選率")
plt.xlabel("元現新")
plt.legend(title="当落")
plt.xticks(rotation=0)

plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))

plt.tight_layout()
plt.show()