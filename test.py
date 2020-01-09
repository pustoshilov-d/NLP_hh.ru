import sqlite3

conn = sqlite3.connect("data.db")
cursor = conn.cursor()
cursor.execute("""select field_name, spec_name from sentences where spec_id =? limit 1""",(8,))
print(cursor.fetchall())
