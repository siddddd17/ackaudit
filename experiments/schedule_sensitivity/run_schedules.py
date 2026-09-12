import argparse
from pathlib import Path

from ackaudit._hashseed import pin
from experiments.schedule_sensitivity.schedules import run, report


def main():
    p = argparse.ArgumentParser(
        description="Compare backward-memory simulation under schedules A/B/C."
    )
    p.add_argument("--outdir", default="out_schedules")
    p.add_argument("--graphs", nargs="*", default=None)
    p.add_argument(
        "--budgets",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    a = p.parse_args()

    rows = run(a.outdir, budgets=a.budgets, names=a.graphs)
    if not rows:
        raise SystemExit("no graphs captured; not writing a report")

    text = report(rows)
    print(text)
    Path(a.outdir, "report.txt").write_text(text)


if __name__ == "__main__":
    pin()
    main()
