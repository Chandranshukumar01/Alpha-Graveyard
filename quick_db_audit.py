"""Quick database audit"""
import sqlite3
conn = sqlite3.connect('btc_pipeline.db')
c = conn.cursor()

# Check tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in c.fetchall()]

print("=== DATABASE AUDIT ===")
print(f"Tables found: {tables}")

for table in tables:
    try:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"  {table}: {count} rows")
    except:
        print(f"  {table}: ERROR")

# Check features table for NULLs
if 'features' in tables:
    c.execute("SELECT ema_20, atr_14, adx_14 FROM features LIMIT 5")
    sample = c.fetchall()
    print(f"\nFeatures sample (first 5 rows):")
    for row in sample:
        print(f"  ema_20={row[0]}, atr_14={row[1]}, adx_14={row[2]}")
    
    # Count NULLs
    c.execute("SELECT COUNT(*) FROM features WHERE ema_20 IS NULL")
    null_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM features")
    total = c.fetchone()[0]
    print(f"\nFeatures with NULL ema_20: {null_count}/{total}")

conn.close()
