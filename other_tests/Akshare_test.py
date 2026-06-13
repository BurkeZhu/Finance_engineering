"""
AKShare 增强版查询脚本
解决 RemoteDisconnected / 反爬问题
"""

import os
import time
import random
import akshare as ak
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# ========== 全局反爬配置 ==========
# AKShare 底层使用 requests，可以通过设置全局请求头伪装浏览器
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 创建带重试机制的 Session（在 akshare 底层生效）
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,  # 指数退避: 1s, 2s, 4s, 8s...
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

# 设置全局请求头（模拟 Chrome）
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
})

# 将 session 注入 akshare（部分接口生效）
ak.requests_session = session


def safe_query(func, *args, min_delay=3, max_delay=6, **kwargs):
    """
    带随机延迟和指数退避的查询函数
    """
    # 每次查询前强制随机等待（反爬核心）
    sleep_time = random.uniform(min_delay, max_delay)
    print(f"  ⏳ 预等待 {sleep_time:.1f}s 以降低请求频率...")
    time.sleep(sleep_time)

    try:
        print(f"  🚀 调用: {func.__name__}")
        df = func(*args, **kwargs)
        print(f"  ✅ 成功 | 返回 {len(df)} 行, {len(df.columns)} 列")
        return df
    except Exception as e:
        print(f"  ❌ 失败: {str(e)[:100]}")
        return pd.DataFrame()


def demo_history_kline_em():
    """东方财富历史 K 线（数据全，但反爬严格）"""
    print("\n" + "="*60)
    print("示例A: 东方财富历史K线 (000063 中兴通讯)")
    print("="*60)
    
    df = safe_query(
        ak.stock_zh_a_hist,
        symbol="000063",
        period="daily",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq",
        min_delay=4,  # 东财接口多等一会
        max_delay=8
    )
    if not df.empty:
        print(df.tail(3))
    return df


def demo_history_kline_sina():
    """新浪历史 K 线（反爬较松，但需自行复权）"""
    print("\n" + "="*60)
    print("示例B: 新浪历史K线 (000063 中兴通讯) —— 备用源")
    print("="*60)
    
    # 新浪接口参数不同，返回的是未复权数据
    df = safe_query(
        ak.stock_zh_a_daily,
        symbol="sz000063",  # 新浪格式需加市场前缀
        start_date="20240101",
        end_date="20241231",
        min_delay=2,
        max_delay=4
    )
    if not df.empty:
        print(df.tail(3))
    return df


def demo_realtime_quote():
    """实时行情（数据量大，间隔要更长）"""
    print("\n" + "="*60)
    print("示例C: 实时行情快照")
    print("="*60)
    
    df = safe_query(
        ak.stock_zh_a_spot_em,
        min_delay=5,  # 全市场数据量大，必须多等
        max_delay=10
    )
    if not df.empty:
        cols = [c for c in ['代码', '名称', '最新价', '涨跌幅'] if c in df.columns]
        print(df[cols].head(5))
    return df


def demo_stock_list():
    """A股列表（通常最稳定）"""
    print("\n" + "="*60)
    print("示例D: A股基础信息")
    print("="*60)
    
    df = safe_query(ak.stock_info_a_code_name, min_delay=1, max_delay=2)
    if not df.empty:
        print(df.head())
    return df


if __name__ == "__main__":
    print("AKShare 增强版反爬脚本启动")
    print(f"版本: {ak.__version__}")
    print("提示: 若仍频繁失败，建议错峰使用或搭配 Tushare 作为主力数据源\n")
    
    # 按顺序运行（间隔会自动叠加）
    demo_stock_list()
    demo_history_kline_em()      # 东财源，可能失败
    # demo_history_kline_sina()  # 新浪备用源
    # demo_realtime_quote()      # 数据量大，容易触发反爬
    
    print("\n" + "="*60)
    print("查询完成。如遇频繁断开，建议: 1)延长等待 2)分时段运行 3)换Tushare")