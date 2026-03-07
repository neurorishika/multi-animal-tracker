import sqlite3
import sys


def check_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT file_path, label FROM images LIMIT 10")
    rows = c.fetchall()
    print(f"First 10 images in {db_path}:")
    for r in rows:
        print(f"  Path: '{r[0]}', Label: {r[1]}")

    c.execute("SELECT COUNT(*) FROM images WHERE label IS NOT NULL")
    count = c.fetchone()[0]
    print(f"Total labeled images: {count}")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_db(sys.argv[1])
    else:
        print("Usage: python check_db_paths.py <db_path>")
