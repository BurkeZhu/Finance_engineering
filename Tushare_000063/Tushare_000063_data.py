import tushare as ts
import os
import pandas as pd

# --- 配置区 ---
TOKEN = os.getenv('TUSHARE_TOKEN')
CSV_FILE = '000063_中兴通讯_历史行情.csv'  # 你的本地文件名
TS_CODE = '000063.SZ'  # 股票代码

# 1. 初始化接口
ts.set_token(TOKEN)
pro = ts.pro_api()


def get_latest_date_from_csv(filename):
    """读取本地CSV文件，返回最新的交易日期"""
    if not os.path.exists(filename):
        print(f"⚠️ 本地文件 {filename} 不存在，将创建新文件。")
        return None, None

    df_local = pd.read_csv(filename, encoding='utf-8-sig')
    if 'trade_date' in df_local.columns:
        # 将字符串转换为日期对象以便比较
        df_local['trade_date'] = pd.to_datetime(df_local['trade_date'])
        latest_date = df_local['trade_date'].max()
        return latest_date, df_local
    else:
        print("❌ 本地文件中未找到日期列。")
        return None, None


def fetch_and_update_data():
    # 2. 获取本地最新的日期
    latest_date, existing_df = get_latest_date_from_csv(CSV_FILE)

    if latest_date is None:
        # 如果是第一次运行，设置一个较早开始日期
        start_date = '20250101'
        print("首次运行，从2025年开始获取数据...")
    else:
        # 如果不是第一次，从“最新日期的下一天”开始获取
        # Tushare 日期格式为 YYYYMMDD，需要转换
        start_date = (latest_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
        print(f"上次数据截止到: {latest_date.date()}，将从 {start_date} 开始获取新数据。")

    # 3. 获取今天的日期作为结束日期
    end_date = pd.Timestamp.today().strftime('%Y%m%d')

    # 4. 调用接口获取新数据
    df_new = pro.daily(ts_code=TS_CODE, start_date=start_date, end_date=end_date)

    if df_new.empty:
        print("ℹ️ 没有获取到新数据（可能是非交易日，或者数据已是最新的）。")
        return

    # 5. 数据清洗（转换日期格式，重命名等）
    df_new['trade_date'] = pd.to_datetime(df_new['trade_date'], format='%Y%m%d')

    # 重命名列（可选）
    # df_new.rename(columns={
    #     'open': '开盘价',
    #     'high': '最高价',
    #     'low': '最低价',
    #     'close': '收盘价',
    #     'vol': '成交量',
    #     'amount': '成交额',
    #     'pct_chg': '涨跌幅(%)'
    # }, inplace=True)

    # 6. 合并数据
    if existing_df is not None and 'trade_date' in existing_df.columns:
        # 将旧数据的日期也转换为 datetime 类型以便合并
        existing_df['trade_date'] = pd.to_datetime(existing_df['trade_date'])
        # 合并旧数据和新数据
        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        combined_df = df_new

    # 7. 去重并排序（以防万一）
    # 按日期排序，保留最新的记录（防止接口偶尔返回重复数据）
    combined_df.sort_values('trade_date', inplace=True)
    combined_df.drop_duplicates(subset=['trade_date'], keep='last', inplace=True)

    # 8. 保存回文件
    combined_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ 更新成功！共 {len(combined_df)} 行数据。最新日期: {combined_df['trade_date'].max().date()}")


# --- 执行更新 ---
if __name__ == "__main__":
    fetch_and_update_data()