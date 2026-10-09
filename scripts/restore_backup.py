"""Restore a SQLite backup. Stop the stack first:  ./scripts/stop.sh
Usage: python scripts/restore_backup.py backups/market_YYYYMMDDTHHMMSSZ.sqlite3.gz [data/market.sqlite3]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.workers.backup import restore_sqlite  # noqa: E402

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(2)
target = sys.argv[2] if len(sys.argv) > 2 else "data/market.sqlite3"
restore_sqlite(Path(sys.argv[1]), target)
print(f"restored {sys.argv[1]} -> {target}")
