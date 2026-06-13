from sqlalchemy import create_engine, text

# 使用你刚才创建的 trade_user 和密码
DB_URL = "mysql+pymysql://trade_user:TradeDB2026!@localhost:3306/trade_db?charset=utf8mb4"

try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ MySQL 连接成功：", result.scalar())
except Exception as e:
    print("❌ 连接失败：", e)