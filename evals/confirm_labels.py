"""Mark the label files as reviewed by the human.

Run this ONLY after reading every row of the label files and fixing the labels you disagree
with: python -m evals.confirm_labels
"""

import csv

from src.config import ROOT, load_config

FILES = ["fresh_headlines.csv", "judge_claims.csv"]
REVIEWED = "human_reviewed"


def confirm(name: str) -> int:
    """Set labeled_by to human_reviewed on every labeled row of one file; return the count."""
    path = ROOT / load_config()["paths"]["labels_dir"] / name
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["labeled_by"]:
            row["labeled_by"] = REVIEWED
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sum(1 for row in rows if row["labeled_by"] == REVIEWED)


def main() -> None:
    """Confirm every label file and say what to run next."""
    for name in FILES:
        print(f"{name}: {confirm(name)} rows marked as {REVIEWED}")
    print("Now run: python -m evals.run_all")


if __name__ == "__main__":
    main()
