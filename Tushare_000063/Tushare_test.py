import os
import tushare as ts

token = os.getenv('TUSHARE_TOKEN')
ts.set_token(token)
pro = ts.pro_api()

# 尝试获取一点数据，比如获取股票列表
try:
    df = pro.stock_basic()
    print("Tushare 初始化成功！前几行数据：")
    print(df.head())
except Exception as e:
    print("初始化失败：", e)