import pandas as pd

df = pd.read_excel("sangiinn_2025.xlsx", engine="openpyxl")

# 最大行数を None に設定
pd.set_option("display.max_rows", None)

print(df)

#列名の表示('当落', '候補者氏名', '年齢', '党派', '元現新', '職業', '議席数', '得票数'のカラムが入っているか確認)
print(df.columns)

#行名の表示(今回の選挙では比例代表での立候補者を除いた選挙区での候補者数が350なので、行が350あることを確認)
print(df.index)

#当選者の数(今回の選挙では定数が125(選挙区＋比例代表)-50(比例代表)=75定数(選挙区)なので数値が75であることを確認)
num_win = (df["当落"] == "当").sum()
print(num_win)