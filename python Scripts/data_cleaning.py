import pandas as pd

df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system.xlsx", engine="openpyxl")

#欠損地の個数を確認
print(df.isnull().values.any())

# 例：df["党派"]列から先頭と末尾にある半角スペース、改行を削除
df["党派"] = df["党派"].str.strip()
# ユニークな党派を確認
print(df["党派"].unique())

#得票数は予測モデルが結果に直結する情報を参照してしまいデータリークを起こす可能性があるので削除する
df = df.drop(columns=["得票数"])

#職業列の全角スペース、改行を削除
df["職業"] = df["職業"].str.replace("　", "", regex=False)
df["職業"] = df["職業"].str.replace("\n", "", regex=False) 

#職種と職種の数
print("職種:", df["職業"].unique())
num_unique = df['職業'].nunique()
print(num_unique)

#統計的な示唆を出しやすくするため、カテゴリを集約
def categorize_job(job):
    if any(x in job for x in ["議員","団体役員","旅行代理店経営","秘書","党","政党","支部","政治団体","政党役員""選挙区支部長"]):
        return "政治家・政党関係"
    elif any(x in job for x in ["会社役員","経営","代表取締役","社長","会社顧問","有限会社レムフク","有限会社丸光代表","ビッグバン株式会社","医療法人ひまわり",
"株式会社ＡＭＳ代表","観光ＰＲ会社代表","株式会社マグナデザインネット チーフサイエンティスト","株式会社リュウタ","株式会社ソシエテ"]):
        return "企業経営者・役員"
    elif any(x in job for x in ["造園業　個人事業","自営業","個人事業主","元飲食店経営","会社代表","株式会社良心塾","有限 会社雅趣代表","英語教室代表","不動産賃貸業"]):
        return "自営業"
    elif any(x in job for x in ["農業","農家","農林業"]):
        return "農林業"
    elif any(x in job for x in ["医師","歯科医師","介護職","外科医","救命医","看護師","助産師","福祉","カウンセラー","医療法人ひまわり"]):
        return "医療・福祉"
    elif any(x in job for x in ["弁護士","司法書士","法律","行政書士","税理士","土地家屋調査士","熊本経営サポート"]):
        return "法律・士業"
    elif any(x in job for x in ["教授","教","塾","教育","学習塾経営","小中学校教員"]):
        return "教育・研究"
    elif any(x in job for x in ["ジャーナリスト","広報代行業","コンセプター","デザイン会社社員"]):
        return "マスコミュニケーション"
    elif any(x in job for x in ["航空会社社員"]):
        return "航空会社社員"
    elif any(x in job for x in ["水道工事","内装業","建築業","プランニング代表"]):
        return "建設・施工業"
    elif any(x in job for x in ["製造業"]):
        return "製造業"
    elif any(x in job for x in ["投資","ファイナンシャルプランナー","個人投資家","保険代理業従業員"]):
        return "金融業"
    elif any(x in job for x in ["音楽","シンガー","ダンス","コメディアン","イベント","プロレス","格闘","YouTuber","作家","スイミングアドバイザー","空手道指導者"]):
        return "芸術・エンタメ・スポーツ"
    elif any(x in job for x in ["IT","エンジニア","システム"]):
        return "ITエンジニア"
    elif any(x in job for x in ["コンサル","中小企業診断士"]):
        return "コンサルタント"
    elif any(x in job for x in ["清掃"]):
        return "清掃業"
    elif any(x in job for x in ["警備員"]):
        return "警備業"
    elif any(x in job for x in ["ネイリスト"]):
        return "美容業"
    elif any(x in job for x in ["観光","旅行"]):
        return "観光・旅行"
    elif any(x in job for x in ["小売"]):
        return "小売業"
    elif any(x in job for x in ["飲食"]):
        return "飲食業"
    elif any(x in job for x in ["団体職員","会社員","パート","アルバイト","有限会社中本農園","シルバー人材センターアルバイト"]):
        return "会社員・社員"
    elif any(x in job for x in ["主婦"]):
        return "主婦"
    elif any(x in job for x in ["無職"]):
        return "無職"
    else:
        return "その他"

#職業列に集約した職業カテゴリを適用し、職業(分類)の列を追加
df["職業(分類)"] = df["職業"].apply(categorize_job)

# 職業列を削除
df = df.drop(columns=["職業"])

# 新しいExcelファイルとして保存
df.to_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx", index=False)

# Excelファイルを読み込む
df = pd.read_excel("./Data/2025_upperhouse_election_constituency_system_cleaning.xlsx")

# 確認
print(df.head(30))