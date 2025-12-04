import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from statsmodels.formula.api import logit
import statsmodels.api as sm

sns.set(font=["Meiryo"])

# データ読み込み
df = pd.read_excel(
    "C:/Users/frontier-Python/Desktop/2025_upperhouse_election_constituency_system_predictor/Data/2025_upperhouse_election_constituency_system_cleaning.xlsx",
    engine="openpyxl"
)

# 当落を数値化
df["当落フラグ"] = df["当落"].map({"当選": 1, "落選": 0, "当": 1, "落": 0})

# 争点の列
issue_columns = ["争点1位", "争点2位", "争点3位"]

# 争点ごとの当選確率（信頼区間付）
issue_win_rates = pd.DataFrame()

for col in issue_columns:
    temp = df.groupby(col)["当落フラグ"].agg(['mean', 'count'])
    temp['se'] = (temp['mean'] * (1 - temp['mean']) / temp['count'])**0.5
    temp['lower'] = temp['mean'] - 1.96 * temp['se']
    temp['upper'] = temp['mean'] + 1.96 * temp['se']

    temp = temp.reset_index()
    temp["順位"] = col
    temp = temp.rename(columns={"mean": "当選確率", col: "政策"})

    issue_win_rates = pd.concat([issue_win_rates, temp], axis=0)

# ---- グラフ（95%信頼区間付き） ----
plt.figure(figsize=(14, 10))

sns.pointplot(
    data=issue_win_rates,
    x="当選確率",
    y="政策",
    hue="順位",
    join=False,
    dodge=0.4,
    errorbar=None
)

# 手動でエラーバー（CI95%）
for i, row in issue_win_rates.iterrows():
    plt.errorbar(
        x=row["当選確率"],
        y=i,  # y位置を i 行に対応させる
        xerr=[[row["当選確率"] - row["lower"]], [row["upper"] - row["当選確率"]]],
        fmt='none',
        capsize=3,
        color='gray'
    )

plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.title("争点にした政策と当選確率（95%信頼区間付き）", fontsize=16)
plt.xlabel("当選確率", fontsize=13)
plt.ylabel("政策", fontsize=13)
plt.tight_layout()
plt.show()

# ---- ロジスティック回帰（政策の効果を数値化） ----
# 争点を1つの列に整形
df_long = df.melt(
    id_vars=["当落フラグ"],
    value_vars=issue_columns,
    var_name="順位",
    value_name="政策"
)

# ダミー変数化
df_dummy = pd.get_dummies(df_long, columns=["政策"], drop_first=True)

# モデル式作成（参考：政策A が基準 → 他の政策との差を見る）
formula = "当落フラグ ~ " + " + ".join([c for c in df_dummy.columns if c.startswith("政策_")])

model = logit(formula, df_dummy).fit()

print(model.summary())
