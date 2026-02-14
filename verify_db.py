"""Verify database integrity"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('btc_pipeline.db')

# Check row counts
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM candles")
candles = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM features")
features = c.fetchone()[0]

print(f"Database Status:")
print(f"  Candles: {candles}")
print(f"  Features: {features}")

# Check sample data
df = pd.read_sql_query("""
    SELECT c.timestamp, c.close, f.adx_14, f.bb_upper, f.bb_lower
    FROM candles c
    JOIN features f ON datetime(c.timestamp) = f.timestamp
    LIMIT 5
""", conn)

print(f"\nSample data:")
print(df)

conn.close()
