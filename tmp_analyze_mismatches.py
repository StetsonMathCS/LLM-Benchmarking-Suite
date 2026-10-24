import csv
import pathlib

p = pathlib.Path(r"reports/comparison/opus_qwen_mismatches.csv")
rows = []
with p.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if (row.get("mismatch_type") or "").strip().lower() != "non_extreme":
            continue
        try:
            d = float(row["diff"])
            float(row["opus_score"])
            float(row["qwen_score"])
        except Exception:
            continue
        row["_abs"] = abs(d)
        rows.append(row)

rows.sort(key=lambda x: x["_abs"], reverse=True)
for row in rows[:30]:
    print(row["index"], row["benchmark"], row["opus_score"], row["qwen_score"], row["diff"])
print("count", len(rows))
