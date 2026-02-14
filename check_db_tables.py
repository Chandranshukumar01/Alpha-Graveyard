"""Check database tables"""
import sqlite3
conn = sqlite3.connect('btc_pipeline.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [t[0] for t in c.fetchall()])
conn.close()
