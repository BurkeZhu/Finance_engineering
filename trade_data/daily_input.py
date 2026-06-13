#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_input.py - 每日交易录入模板
使用方法：
1. 复制此文件，修改日期和交易记录
2. 运行：python daily_input.py
"""

from trade_daily_manager import TradeManager

# ==================== 每日配置区 ====================
# 改成今天的日期
TODAY = '2026-06-16'

# 初始化（表已存在时会自动跳过，不会重复建表）
tm = TradeManager()

# ==================== 录入当日成交 ====================
# 有几笔写几笔，没有交易就留空或注释掉

# 示例1：买入红利质量ETF
# oid1 = tm.add_order(TODAY, '159209.SZ', 'BUY', 1.1680, 3000, 'CORE_ETF')
# tm.fill_order(oid1, 1.1670, 3000, commission=0.88)

# 示例2：卖出中证1000ETF（如需）
# oid2 = tm.add_order(TODAY, '159845.SZ', 'SELL', 2.4200, 1000, 'SATELLITE')
# tm.fill_order(oid2, 2.4180, 1000, commission=0.60)

# 示例3：买入黄金ETF（如需）
# oid3 = tm.add_order(TODAY, '159934.SZ', 'BUY', 5.9200, 500, '卫星配置')
# tm.fill_order(oid3, 5.9150, 500, commission=0.74)

# ==================== 更新收盘价 ====================
# 6只自选全部更新（即使没交易，价格也在变）
tm.update_close_price(TODAY, '159209.SZ', 1.1720)   # 红利质量
tm.update_close_price(TODAY, '159845.SZ', 2.4250)     # 中证1000
tm.update_close_price(TODAY, '159934.SZ', 5.9100)    # 黄金
tm.update_close_price(TODAY, '511030.SH', 1.0290)    # 公司债
tm.update_close_price(TODAY, '511360.SH', 1.0100)    # 短融
tm.update_close_price(TODAY, '159338.SZ', 1.0650)     # 中证A500

# ==================== 计算日终盈亏 ====================
tm.calc_daily_pnl(TODAY)

# ==================== 查看与导出 ====================
print("\n📦 当前持仓:")
print(tm.get_position(TODAY).to_string(index=False))

print("\n📈 当日成交:")
print(tm.get_trade_history(start_date=TODAY, end_date=TODAY).to_string(index=False))

print("\n💰 当日汇总:")
print(tm.get_daily_summary(start_date=TODAY, end_date=TODAY).to_string(index=False))

# 导出Excel（可选）
tm.export_daily_report(TODAY, f'trade_report_{TODAY}.xlsx')
