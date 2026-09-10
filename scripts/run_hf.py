from ackaudit._hashseed import pin
pin()

import argparse, logging
from pathlib import Path
from ackaudit.capture import capture_all
from ackaudit.analyze import report, degeneracy_report
from ackaudit.schedule import DEFAULT_SCHEDULE, SCHEDULES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="out_hf")
    p.add_argument("--models", nargs="*", default=["llama", "bert", "vit"])
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--schedule", choices=SCHEDULES, default=DEFAULT_SCHEDULE)
    p.add_argument("--budgets", nargs="*", type=float,
                   default=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    captured = capture_all(a.outdir, names=a.models, budgets=a.budgets, scale=a.scale, schedule=a.schedule)
    if not any(v.records for v in captured.values()):
        raise SystemExit(
            f"no graphs captured from {a.models}; not writing a report. "
            "check the capture errors above"
        )
    text = (f"schedule: {a.schedule}   PYTHONHASHSEED: 0   scale: {a.scale}\n\n"
            + report(a.outdir) + "\n" + degeneracy_report(a.outdir))
    print(text)
    Path(a.outdir, "report.txt").write_text(text)

if __name__ == "__main__":
    main()
