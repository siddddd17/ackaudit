from ackaudit._hashseed import pin
pin()

import argparse, logging
from pathlib import Path
from ackaudit.capture import capture_all, MODELS
from ackaudit.analyze import report, degeneracy_report
from ackaudit.schedule import DEFAULT_SCHEDULE, SCHEDULES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="out")
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--schedule", choices=SCHEDULES, default=DEFAULT_SCHEDULE)
    p.add_argument("--budgets", nargs="*", type=float,
                   default=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    capture_all(a.outdir, names=a.models, budgets=a.budgets, schedule=a.schedule)
    text = (f"schedule: {a.schedule}   PYTHONHASHSEED: 0\n\n"
            + report(a.outdir) + "\n" + degeneracy_report(a.outdir))
    print(text)
    Path(a.outdir, "report.txt").write_text(text)

if __name__ == "__main__":
    main()
