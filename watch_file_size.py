
import sys
import time
from pathlib import Path

def format_size(size_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file>")
    sys.exit(1)

file_path = Path(sys.argv[1])

while True:
    try:
        size = file_path.stat().st_size
        text = f"{file_path}: {size} bytes ({format_size(size)})"
    except FileNotFoundError:
        text = f"{file_path}: file not found"

    print("\r" + text + " " * 20, end="", flush=True)
    time.sleep(1)
