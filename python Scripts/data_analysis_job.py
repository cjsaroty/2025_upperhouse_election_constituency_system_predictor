import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font=["Meiryo"])

df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", engine="openpyxl")

job_outcome = pd.crosstab(df['職業(分類)'], df['当落'])

order = [job for job in job_outcome.index if job != 'その他'] + ['その他']
job_outcome = job_outcome.reindex(order)

#当選率の折れ線グラフ
job_outcome['当選率'] = job_outcome['当'] / job_outcome.sum(axis=1)

# 棒グラフに変更
plt.figure(figsize=(12,6))
sns.lineplot(x=job_outcome.index, y=job_outcome['当選率'], marker='o')
plt.title('職業ごとの当選率')
plt.xlabel('職業')
plt.ylabel('当選率(%)')
plt.ylim(0,1)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
