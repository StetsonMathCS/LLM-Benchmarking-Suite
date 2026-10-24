import csv
import pathlib
from collections import defaultdict

p = pathlib.Path(r"reports/comparison/opus_qwen_mismatches.csv")
by_benchmark = defaultdict(list)

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
        by_benchmark[row["benchmark"]].append(row)

for b in by_benchmark:
    by_benchmark[b].sort(key=lambda x: x["_abs"], reverse=True)

eligible = {b: rows for b, rows in by_benchmark.items() if len(rows) >= 2}
best_benchmark = max(eligible, key=lambda b: eligible[b][0]["_abs"] + eligible[b][1]["_abs"])
selected = eligible[best_benchmark][:2]

print("best_benchmark", best_benchmark)
for row in selected:
    print(row["index"], row["benchmark"], row["opus_score"], row["qwen_score"], row["diff"])
