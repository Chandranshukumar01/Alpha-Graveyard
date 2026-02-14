"""Check archived database"""
import sqlite3
conn = sqlite3.connect('_archive/btc_pipeline.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [t[0] for t in c.fetchall()])

# Check row counts
try:
    c.execute("SELECT COUNT(*) FROM candles")
    print("Candles:", c.fetchone()[0])
except:
    print("No candles table")

try:
    c.execute("SELECT COUNT(*) FROM features")
    print("Features:", c.fetchone()[0])
except:
    print("No features table")

conn.close()
