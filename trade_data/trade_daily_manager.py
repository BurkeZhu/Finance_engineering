#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Daily Manager - 每日交易数据更新接口 (完整版)
基于真实自选池 (watchlist) 的持仓与盈亏管理
自动建表，拿到即用
"""

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

# ==================== 配置区 ====================
DB_URL = "mysql+pymysql://trade_user:TradeDB2026!@localhost:3306/trade_db?charset=utf8mb4"
engine = create_engine(DB_URL, echo=False, future=True)
Session = sessionmaker(bind=engine)

# ==================== 核心类 ====================

class TradeManager:
    """
    每日交易数据管理接口

    使用示例:
        tm = TradeManager()  # 自动建表

        # 1. 录入一笔买入委托并成交
        order_id = tm.add_order('2026-06-13', '159209.SZ', 'BUY', 1.1560, 2000, 'CORE_ETF')
        tm.fill_order(order_id, 1.1550, 2000, commission=0.58)

        # 2. 日终更新收盘价并计算盈亏
        tm.update_close_price('2026-06-13', '159209.SZ', 1.1620)
        tm.calc_daily_pnl('2026-06-13')

        # 3. 查询当前持仓
        df = tm.get_position()
        print(df)
    """

    def __init__(self, auto_init: bool = True):
        self.session = Session()
        if auto_init:
            self.init_tables()

    def __del__(self):
        if hasattr(self, 'session') and self.session:
            self.session.close()

    # ---------- 0. 自动建表 ----------

    def init_tables(self) -> None:
        """自动创建所有表（如果不存在）"""

        # 1. 自选池表
        self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS watchlist (
                code         VARCHAR(20) PRIMARY KEY COMMENT '标的代码',
                name         VARCHAR(50) NOT NULL COMMENT 'ETF名称',
                market       ENUM('SH', 'SZ') NOT NULL COMMENT '交易所',
                category     ENUM('BOND', 'CASH', 'COMMODITY', 'EQUITY') NOT NULL COMMENT '资产类别',
                strategy_tag VARCHAR(50) COMMENT '策略标签',
                added_date   DATE DEFAULT (CURRENT_DATE) COMMENT '加入自选日期',
                notes        VARCHAR(200) COMMENT '备注'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        # 2. 委托表
        self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id        INT AUTO_INCREMENT PRIMARY KEY,
                trade_date      DATE NOT NULL,
                code            VARCHAR(20) NOT NULL COMMENT '标的代码',
                direction       ENUM('BUY', 'SELL') NOT NULL,
                order_price     DECIMAL(10,4),
                order_volume    INT NOT NULL,
                order_status    ENUM('PENDING', 'FILLED', 'PARTIAL', 'CANCELLED') DEFAULT 'PENDING',
                strategy_tag    VARCHAR(50) COMMENT '策略标签',
                create_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_order_date_code (trade_date, code)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        # 3. 成交表
        self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id        INT AUTO_INCREMENT PRIMARY KEY,
                order_id        INT,
                trade_date      DATE NOT NULL,
                code            VARCHAR(20) NOT NULL,
                direction       ENUM('BUY', 'SELL') NOT NULL,
                trade_price     DECIMAL(10,4) NOT NULL,
                trade_volume    INT NOT NULL,
                trade_amount    DECIMAL(15,4),
                commission      DECIMAL(10,4) DEFAULT 0 COMMENT '佣金',
                tax             DECIMAL(10,4) DEFAULT 0 COMMENT '印花税',
                create_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (order_id) REFERENCES orders(order_id),
                INDEX idx_trade_date (trade_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        # 4. 日终持仓盈亏表
        self.session.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                snapshot_date   DATE NOT NULL,
                code            VARCHAR(20) NOT NULL,
                hold_volume     INT NOT NULL DEFAULT 0,
                hold_cost       DECIMAL(15,4) DEFAULT 0 COMMENT '持仓成本',
                close_price     DECIMAL(10,4) COMMENT '当日收盘价',
                market_value    DECIMAL(15,4),
                daily_pnl       DECIMAL(15,4) COMMENT '当日浮动盈亏',
                cum_pnl         DECIMAL(15,4) COMMENT '累计实现盈亏',
                PRIMARY KEY (snapshot_date, code)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        self.session.commit()
        print("✅ 数据库表初始化完成（watchlist + orders + trades + daily_pnl）")

    def init_watchlist(self, etfs: list = None) -> None:
        """
        初始化自选池数据

        Parameters:
            etfs: 自定义标的列表，格式 [(code, name, market, category, strategy_tag, notes), ...]
                  默认使用用户截图中的 6 只 ETF
        """
        if etfs is None:
            etfs = [
                ('511030.SH', '公司债ETF平安', 'SH', 'BOND', '债券底仓', '信用债ETF'),
                ('511360.SH', '短融ETF海富通', 'SH', 'CASH', '现金替代', '短久期现金ETF'),
                ('159934.SZ', '黄金ETF易方达', 'SZ', 'COMMODITY', '卫星配置', '避险资产'),
                ('159209.SZ', '红利质量ETF招商', 'SZ', 'EQUITY', 'CORE_ETF', '红利+质量双因子'),
                ('159845.SZ', '中证1000ETF华夏', 'SZ', 'EQUITY', 'SATELLITE', '小盘成长'),
                ('159338.SZ', '中证A500ETF', 'SZ', 'EQUITY', 'CORE_ETF', 'A500宽基'),
            ]

        for e in etfs:
            exists = self.session.execute(
                text("SELECT 1 FROM watchlist WHERE code = :code"),
                {"code": e[0]}
            ).fetchone()
            if not exists:
                self.session.execute(text("""
                    INSERT INTO watchlist (code, name, market, category, strategy_tag, notes)
                    VALUES (:code, :name, :market, :category, :strategy_tag, :notes)
                """), {
                    "code": e[0], "name": e[1], "market": e[2], "category": e[3],
                    "strategy_tag": e[4], "notes": e[5]
                })

        self.session.commit()
        print(f"✅ 自选池已初始化，共 {len(etfs)} 只 ETF")

    # ---------- 1. 委托与成交 ----------

    def add_order(self, trade_date: str, code: str, direction: str, 
                  order_price: float, order_volume: int, 
                  strategy_tag: str = None, order_status: str = 'PENDING') -> int:
        """
        录入委托单

        Parameters:
            trade_date: 交易日期 'YYYY-MM-DD'
            code: 标的代码，如 '159209.SZ'
            direction: 'BUY' 或 'SELL'
            order_price: 委托价格
            order_volume: 委托数量（整数）
            strategy_tag: 策略标签，如 'CORE_ETF' / 'SATELLITE' / '债券底仓'
            order_status: 默认 'PENDING'，成交后调用 fill_order() 更新

        Returns:
            order_id: 委托编号
        """
        # 校验标的在自选池中
        exists = self.session.execute(
            text("SELECT 1 FROM watchlist WHERE code = :code"),
            {"code": code}
        ).fetchone()
        if not exists:
            raise ValueError(f"标的 {code} 不在自选池中，请先添加到 watchlist")

        sql = """
        INSERT INTO orders (trade_date, code, direction, order_price, order_volume, 
                            order_status, strategy_tag)
        VALUES (:trade_date, :code, :direction, :order_price, :order_volume, 
                :order_status, :strategy_tag)
        """
        result = self.session.execute(text(sql), {
            "trade_date": trade_date,
            "code": code,
            "direction": direction,
            "order_price": order_price,
            "order_volume": order_volume,
            "order_status": order_status,
            "strategy_tag": strategy_tag
        })
        self.session.commit()
        order_id = result.lastrowid
        print(f"✅ 委托已录入: {direction} {code} {order_volume}股 @ {order_price} (ID={order_id})")
        return order_id

    def fill_order(self, order_id: int, trade_price: float, trade_volume: int,
                   commission: float = 0, tax: float = 0) -> None:
        """
        将委托标记为成交，并写入成交明细

        Parameters:
            order_id: add_order() 返回的委托编号
            trade_price: 实际成交价格
            trade_volume: 实际成交数量（可能小于委托数量）
            commission: 佣金（元），默认0
            tax: 印花税（元），ETF通常为0，默认0
        """
        # 获取委托信息
        order = self.session.execute(
            text("SELECT * FROM orders WHERE order_id = :order_id"),
            {"order_id": order_id}
        ).mappings().fetchone()

        if not order:
            raise ValueError(f"委托 {order_id} 不存在")

        # 更新委托状态
        self.session.execute(
            text("UPDATE orders SET order_status = 'FILLED' WHERE order_id = :order_id"),
            {"order_id": order_id}
        )

        # 写入成交
        trade_amount = Decimal(str(trade_price)) * Decimal(str(trade_volume))
        trade_amount = trade_amount.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

        sql = """
        INSERT INTO trades (order_id, trade_date, code, direction, trade_price, 
                            trade_volume, trade_amount, commission, tax)
        VALUES (:order_id, :trade_date, :code, :direction, :trade_price, 
                :trade_volume, :trade_amount, :commission, :tax)
        """
        self.session.execute(text(sql), {
            "order_id": order_id,
            "trade_date": order["trade_date"],
            "code": order["code"],
            "direction": order["direction"],
            "trade_price": trade_price,
            "trade_volume": trade_volume,
            "trade_amount": float(trade_amount),
            "commission": commission,
            "tax": tax
        })
        self.session.commit()
        print(f"✅ 成交已录入: {order['direction']} {order['code']} {trade_volume}股 @ {trade_price} "
              f"(佣金{commission}, 印花税{tax})")

    def cancel_order(self, order_id: int) -> None:
        """撤销未成交委托"""
        self.session.execute(
            text("UPDATE orders SET order_status = 'CANCELLED' WHERE order_id = :order_id"),
            {"order_id": order_id}
        )
        self.session.commit()
        print(f"🚫 委托 {order_id} 已撤销")

    # ---------- 2. 日终收盘与盈亏计算 ----------

    def update_close_price(self, snapshot_date: str, code: str, close_price: float) -> None:
        """
        更新某只标的在某日的收盘价
        """
        sql = """
        INSERT INTO daily_pnl (snapshot_date, code, close_price)
        VALUES (:snapshot_date, :code, :close_price)
        ON DUPLICATE KEY UPDATE close_price = :close_price
        """
        self.session.execute(text(sql), {
            "snapshot_date": snapshot_date,
            "code": code,
            "close_price": close_price
        })
        self.session.commit()
        print(f"📌 收盘价更新: {code} @ {close_price} ({snapshot_date})")

    def calc_daily_pnl(self, snapshot_date: str) -> None:
        """
        计算指定日期的日终持仓与盈亏
        """
        # 获取所有有交易的标的
        codes = self.session.execute(text("""
            SELECT DISTINCT code FROM trades WHERE trade_date <= :d
        """), {"d": snapshot_date}).fetchall()
        codes = [c[0] for c in codes]

        for code in codes:
            # 累计买入
            buy_result = self.session.execute(text("""
                SELECT COALESCE(SUM(trade_volume), 0), COALESCE(SUM(trade_price * trade_volume), 0)
                FROM trades WHERE code = :c AND trade_date <= :d AND direction = 'BUY'
            """), {"c": code, "d": snapshot_date}).fetchone()

            # 累计卖出
            sell_result = self.session.execute(text("""
                SELECT COALESCE(SUM(trade_volume), 0), COALESCE(SUM(trade_price * trade_volume), 0)
                FROM trades WHERE code = :c AND trade_date <= :d AND direction = 'SELL'
            """), {"c": code, "d": snapshot_date}).fetchone()

            total_buy_vol = int(buy_result[0] or 0)
            total_buy_amt = Decimal(str(buy_result[1] or 0))
            total_sell_vol = int(sell_result[0] or 0)
            total_sell_amt = Decimal(str(sell_result[1] or 0))

            hold_volume = total_buy_vol - total_sell_vol

            if hold_volume <= 0:
                self.session.execute(text("""
                    DELETE FROM daily_pnl WHERE snapshot_date = :d AND code = :c
                """), {"d": snapshot_date, "c": code})
                continue

            # 加权平均持仓成本
            hold_cost = (total_buy_amt / Decimal(str(total_buy_vol))).quantize(
                Decimal('0.0001'), rounding=ROUND_HALF_UP
            ) if total_buy_vol > 0 else Decimal('0')

            # 获取当日收盘价
            close_row = self.session.execute(text("""
                SELECT close_price FROM daily_pnl 
                WHERE snapshot_date = :d AND code = :c
            """), {"d": snapshot_date, "c": code}).fetchone()

            close_price = Decimal(str(close_row[0])) if close_row else Decimal('0')

            if close_price == 0:
                print(f"⚠️ {code} 未录入 {snapshot_date} 收盘价，跳过盈亏计算")
                continue

            market_value = (Decimal(str(hold_volume)) * close_price).quantize(
                Decimal('0.0001'), rounding=ROUND_HALF_UP
            )
            daily_pnl = (Decimal(str(hold_volume)) * (close_price - hold_cost)).quantize(
                Decimal('0.0001'), rounding=ROUND_HALF_UP
            )
            cum_pnl = (total_sell_amt - (hold_cost * Decimal(str(total_sell_vol)))).quantize(
                Decimal('0.0001'), rounding=ROUND_HALF_UP
            ) if total_sell_vol > 0 else Decimal('0')

            sql = """
            INSERT INTO daily_pnl (snapshot_date, code, hold_volume, hold_cost, close_price, 
                                   market_value, daily_pnl, cum_pnl)
            VALUES (:snapshot_date, :code, :hold_volume, :hold_cost, :close_price,
                    :market_value, :daily_pnl, :cum_pnl)
            ON DUPLICATE KEY UPDATE
                hold_volume = VALUES(hold_volume),
                hold_cost = VALUES(hold_cost),
                close_price = VALUES(close_price),
                market_value = VALUES(market_value),
                daily_pnl = VALUES(daily_pnl),
                cum_pnl = VALUES(cum_pnl)
            """
            self.session.execute(text(sql), {
                "snapshot_date": snapshot_date,
                "code": code,
                "hold_volume": hold_volume,
                "hold_cost": float(hold_cost),
                "close_price": float(close_price),
                "market_value": float(market_value),
                "daily_pnl": float(daily_pnl),
                "cum_pnl": float(cum_pnl)
            })

        self.session.commit()
        print(f"✅ 日终持仓盈亏已计算: {snapshot_date}")

    # ---------- 3. 查询接口 ----------

    def get_position(self, snapshot_date: str = None) -> pd.DataFrame:
        """
        查询持仓

        Parameters:
            snapshot_date: 指定日期，默认最新日期
        """
        if snapshot_date is None:
            snapshot_date = self.session.execute(
                text("SELECT MAX(snapshot_date) FROM daily_pnl")
            ).scalar()

        sql = """
        SELECT 
            d.snapshot_date, d.code, w.name, w.category, w.strategy_tag,
            d.hold_volume, d.hold_cost, d.close_price, d.market_value,
            d.daily_pnl, d.cum_pnl
        FROM daily_pnl d
        LEFT JOIN watchlist w ON d.code = w.code
        WHERE d.snapshot_date = :snapshot_date AND d.hold_volume > 0
        ORDER BY d.market_value DESC
        """
        df = pd.read_sql(text(sql), engine, params={"snapshot_date": snapshot_date})
        return df

    def get_trade_history(self, start_date: str = None, end_date: str = None, 
                          code: str = None) -> pd.DataFrame:
        """查询成交历史"""
        conditions = []
        params = {}

        if start_date:
            conditions.append("t.trade_date >= :start_date")
            params["start_date"] = start_date
        if end_date:
            conditions.append("t.trade_date <= :end_date")
            params["end_date"] = end_date
        if code:
            conditions.append("t.code = :code")
            params["code"] = code

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
        SELECT 
            t.trade_date, t.code, w.name, t.direction,
            t.trade_price, t.trade_volume, t.trade_amount,
            t.commission, t.tax, o.strategy_tag
        FROM trades t
        LEFT JOIN orders o ON t.order_id = o.order_id
        LEFT JOIN watchlist w ON t.code = w.code
        {where_clause}
        ORDER BY t.trade_date DESC, t.code
        """
        df = pd.read_sql(text(sql), engine, params=params)
        return df

    def get_daily_summary(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """按日统计交易汇总"""
        conditions = []
        params = {}

        if start_date:
            conditions.append("trade_date >= :start_date")
            params["start_date"] = start_date
        if end_date:
            conditions.append("trade_date <= :end_date")
            params["end_date"] = end_date

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
        SELECT 
            trade_date,
            SUM(CASE WHEN direction='BUY' THEN trade_amount ELSE 0 END) AS buy_amount,
            SUM(CASE WHEN direction='SELL' THEN trade_amount ELSE 0 END) AS sell_amount,
            SUM(commission) AS total_commission,
            SUM(tax) AS total_tax,
            COUNT(*) AS trade_count
        FROM trades
        {where_clause}
        GROUP BY trade_date
        ORDER BY trade_date DESC
        """
        df = pd.read_sql(text(sql), engine, params=params)
        return df

    def export_daily_report(self, snapshot_date: str, filepath: str = None) -> str:
        """
        导出指定日期的完整日报到 Excel
        """
        if filepath is None:
            filepath = f"trade_report_{snapshot_date}.xlsx"

        pos = self.get_position(snapshot_date)
        trades = self.get_trade_history(start_date=snapshot_date, end_date=snapshot_date)
        summary = self.get_daily_summary(start_date=snapshot_date, end_date=snapshot_date)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            pos.to_excel(writer, sheet_name='持仓', index=False)
            trades.to_excel(writer, sheet_name='当日成交', index=False)
            summary.to_excel(writer, sheet_name='当日汇总', index=False)

        print(f"📊 日报已导出: {filepath}")
        return filepath

    # ---------- 4. 修改接口 ----------

    def update_order(self, order_id: int, **kwargs) -> None:
        """
        修改委托记录

        可修改字段: trade_date, code, direction, order_price, order_volume, strategy_tag

        示例:
            tm.update_order(1, order_price=1.1580, order_volume=2500)
        """
        allowed_fields = {'trade_date', 'code', 'direction', 'order_price', 
                           'order_volume', 'strategy_tag', 'order_status'}

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            print("⚠️ 没有可更新的字段")
            return

        set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
        sql = f"UPDATE orders SET {set_clause} WHERE order_id = :order_id"
        updates["order_id"] = order_id

        self.session.execute(text(sql), updates)
        self.session.commit()
        print(f"✅ 委托 {order_id} 已更新: {updates}")

    def update_trade(self, trade_id: int, **kwargs) -> None:
        """
        修改成交记录

        可修改字段: trade_price, trade_volume, commission, tax
        注意: 修改后 trade_amount 不会自动重算，如需更新请手动传入

        示例:
            tm.update_trade(1, trade_price=1.1570, commission=0.88)
        """
        allowed_fields = {'trade_price', 'trade_volume', 'commission', 'tax', 'trade_amount'}

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            print("⚠️ 没有可更新的字段")
            return

        set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
        sql = f"UPDATE trades SET {set_clause} WHERE trade_id = :trade_id"
        updates["trade_id"] = trade_id

        self.session.execute(text(sql), updates)
        self.session.commit()
        print(f"✅ 成交 {trade_id} 已更新: {updates}")

    def update_close_price(self, snapshot_date: str, code: str, close_price: float) -> None:
        """
        更新收盘价（覆盖原有方法，支持修改已有记录）
        """
        sql = """
        INSERT INTO daily_pnl (snapshot_date, code, close_price)
        VALUES (:snapshot_date, :code, :close_price)
        ON DUPLICATE KEY UPDATE close_price = :close_price
        """
        self.session.execute(text(sql), {
            "snapshot_date": snapshot_date,
            "code": code,
            "close_price": close_price
        })
        self.session.commit()
        print(f"📌 收盘价更新: {code} @ {close_price} ({snapshot_date})")

    def delete_order(self, order_id: int, cascade: bool = True) -> None:
        """
        删除委托记录

        Parameters:
            cascade: 是否级联删除关联的成交记录（默认 True）
        """
        if cascade:
            # 先删关联成交
            self.session.execute(
                text("DELETE FROM trades WHERE order_id = :order_id"),
                {"order_id": order_id}
            )

        self.session.execute(
            text("DELETE FROM orders WHERE order_id = :order_id"),
            {"order_id": order_id}
        )
        self.session.commit()
        print(f"🗑️ 委托 {order_id} 已删除{'(含关联成交)' if cascade else ''}")

    def delete_trade(self, trade_id: int) -> None:
        """删除单条成交记录"""
        self.session.execute(
            text("DELETE FROM trades WHERE trade_id = :trade_id"),
            {"trade_id": trade_id}
        )
        self.session.commit()
        print(f"🗑️ 成交 {trade_id} 已删除")

    def recalc_pnl(self, snapshot_date: str) -> None:
        """
        重新计算指定日期的持仓盈亏

        适用场景: 修改了历史成交记录后，需要重新计算该日及之后的持仓
        """
        # 删除该日期的旧计算结果
        self.session.execute(
            text("DELETE FROM daily_pnl WHERE snapshot_date = :d"),
            {"d": snapshot_date}
        )
        self.session.commit()

        # 重新计算
        self.calc_daily_pnl(snapshot_date)
        print(f"🔄 {snapshot_date} 持仓盈亏已重新计算")
# ==================== 使用示例 ====================

# def demo():
#     """
#     演示：如何录入 2026-06-13 的真实交易并计算日终盈亏
#     """
#     tm = TradeManager()      # 自动建表
#     tm.init_watchlist()      # 初始化自选池（已有则跳过）

#     # --- 假设今日发生以下交易 ---
#     today = '2026-06-13'

#     # 1. 买入红利质量ETF 2000股
#     oid1 = tm.add_order(today, '159209.SZ', 'BUY', 1.1560, 2000, 'CORE_ETF')
#     tm.fill_order(oid1, 1.1550, 2000, commission=0.58)

#     # 2. 卖出中证1000ETF 1000股
#     oid2 = tm.add_order(today, '159845.SZ', 'SELL', 2.4100, 1000, 'SATELLITE')
#     tm.fill_order(oid2, 2.4120, 1000, commission=0.60, tax=0)

#     # 3. 买入黄金ETF 500股
#     oid3 = tm.add_order(today, '159934.SZ', 'BUY', 5.8900, 500, '卫星配置')
#     tm.fill_order(oid3, 5.8850, 500, commission=0.74)

#     # --- 日终更新收盘价 ---
#     tm.update_close_price(today, '159209.SZ', 1.1620)
#     tm.update_close_price(today, '159845.SZ', 2.4080)
#     tm.update_close_price(today, '159934.SZ', 5.9050)
#     tm.update_close_price(today, '511030.SH', 1.0280)
#     tm.update_close_price(today, '511360.SH', 1.0090)
#     tm.update_close_price(today, '159338.SZ', 1.0610)

#     # --- 计算日终持仓盈亏 ---
#     tm.calc_daily_pnl(today)

#     # --- 查询与导出 ---
#     print("\n📦 当前持仓:")
#     print(tm.get_position(today).to_string(index=False))

#     print("\n📈 当日成交:")
#     print(tm.get_trade_history(start_date=today, end_date=today).to_string(index=False))

#     print("\n💰 当日汇总:")
#     print(tm.get_daily_summary(start_date=today, end_date=today).to_string(index=False))

#     # 导出 Excel
#     tm.export_daily_report(today)

# if __name__ == "__main__":
#    demo()
