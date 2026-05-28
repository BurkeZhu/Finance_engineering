from dotenv import load_dotenv
import os
import tushare as ts

# 加载 .env 文件（如果存在）
load_dotenv()
# 从环境变量读取，不是硬编码
token = os.getenv('TUSHARE_TOKEN')

if not token:
    raise ValueError(
        "环境变量 TUSHARE_TOKEN 未设置！\n"
        "本地开发：在项目根目录创建 .env 文件写入 TUSHARE_TOKEN=你的token\n"
        "GitHub Actions：在仓库 Settings → Secrets 添加 TUSHARE_TOKEN"
    )

ts.set_token(token)
pro = ts.pro_api()

# 测试连接
try:
    df = pro.stock_basic()
    print("Tushare 初始化成功！")
    print(df.head())
except Exception as e:
    print("初始化失败：", e)