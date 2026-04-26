import pandas as pd
import mplfinance as mpf
from ta.trend import MACD
import numpy as np

# --- 1. 数据读取与清洗 ---
# 读取你上传的文件
df = pd.read_csv('000063_中兴通讯_历史行情.csv')

# 1.1 转换日期并设为索引
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)

# 1.2 关键修复：重命名列 (mplfinance 默认寻找 Open, High, Low, Close, Volume)
# 根据你的CSV内容进行映射
df.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'vol': 'Volume'  # 很多报错是因为找不到 'Volume' 列
}, inplace=True)

# 确保数据按时间顺序
df.sort_index(inplace=True)

# --- 2. 计算技术指标 ---
# 2.1 计算均线
df['MA5'] = df['Close'].rolling(5).mean()
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['MA60'] = df['Close'].rolling(60).mean()

# 2.2 计算 MACD
macd_indicator = MACD(df['Close'])
df['MACD_Line'] = macd_indicator.macd()  # DIF线
df['MACD_Signal'] = macd_indicator.macd_signal()  # DEA线
df['MACD_Hist'] = macd_indicator.macd_diff()  # 柱状图

# --- 3. 构建图形元素 (关键修改点) ---
# 3.1 K线图上的均线
# 将均线合并为一个列表
apd_lines = [
    mpf.make_addplot(df['MA5'], color='white', width=1),
    mpf.make_addplot(df['MA10'], color='yellow', width=1),
    mpf.make_addplot(df['MA20'], color='purple', width=1),  # 用purple代替magenta避免潜在颜色问题
    mpf.make_addplot(df['MA60'], color='green', width=1),
]

# 3.2 解决报错的核心：将 MACD 的线和柱子分开绘制，并拆分线条
# 这样每一行 make_addplot 只处理单一颜色，彻底避开列表报错
apd_macd = [
    # 拆分线条：分别绘制，避免 color=['blue', 'orange'] 这种写法
    mpf.make_addplot(df['MACD_Line'], panel=1, color='blue', width=1.5, secondary_y=False),
    mpf.make_addplot(df['MACD_Signal'], panel=1, color='orange', width=1.5, secondary_y=False),

    # 柱状图：为了简单稳定，先统一颜色（或者用下面的进阶红绿法）
    mpf.make_addplot(df['MACD_Hist'], type='bar', panel=1, color='gray', alpha=0.5, secondary_y=False)
]

# 合并所有图形元素
all_addplots = apd_lines + apd_macd

# --- 4. 绘图 ---
# 创建风格 (如果报字体错，暂时注释掉 rc 参数)
style = mpf.make_mpf_style(base_mpf_style='yahoo')

mpf.plot(
    df,
    type='candle',
    addplot=all_addplots,  # 使用合并后的图形列表
    volume=True,  # 开启成交量
    volume_panel=2,  # 成交量放在最下面 (Panel 2)
    title='000063 中兴通讯 - K线+均线+MACD',
    style=style,
    figsize=(14, 10),
    tight_layout=True
)