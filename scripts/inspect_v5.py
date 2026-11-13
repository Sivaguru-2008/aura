import re
from pathlib import Path

v5_path = Path("docs/MANUSCRIPT_v5.0.md")
text = v5_path.read_text(encoding="utf-8")
print(f"Total lines: {len(text.splitlines())}, total chars: {len(text)}")

print("\n--- Headers ---")
for i, line in enumerate(text.splitlines(), 1):
    if line.strip().startswith("#"):
        print(f"L{i}: {line}")

print("\n--- Tables ---")
for i, line in enumerate(text.splitlines(), 1):
    if re.match(r"^\s*Table\s+\d+", line, re.IGNORECASE) or re.match(r"^\s*\|\s*#?\s*\|", line):
        print(f"L{i}: {line[:80]}")
