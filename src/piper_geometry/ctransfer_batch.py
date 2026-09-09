"""
Piper Geometry: Batch runner for concept_transfer_test.py.

Runs concept_transfer_test.py as a FRESH SUBPROCESS per trial, not
in-process, because a CUDA device-side assert (observed reliably on the
random-rotation condition, twice, with two different random rotation
matrices) leaves the CUDA context corrupted for the rest of that process -
there is no way to recover and safely continue running more trials once
that happens. A subprocess-per-trial design means one crash only costs
that one trial; the next trial starts a genuinely clean CUDA context
regardless of whether the previous one crashed.

Cycles through all six concept-dictionary domains rather than repeating
the same physics passage every time, so a batch surfaces both run-to-run
stochastic variation (sampling temperature, the random-rotation control's
unseeded matrix draw) and any domain-dependence in the pattern seen so
far (calibrated rotation numerically stable but incoherent; random
rotation crashing outright).

Usage: python3 src/piper_geometry/ctransfer_batch.py [num_trials]
(defaults to 50)
"""

import subprocess
import sys
import random
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = Path(__file__).resolve().parent / "concept_transfer_test.py"
BATCH_LOG_PATH = REPO_ROOT / "obsidian" / "Experiments" / "ctransfer_batch_log.md"

DOMAINS = ["physics", "computer_science", "philosophy", "mathematics", "cognitive_science", "biology"]

# How much of a crashed trial's stderr to show live, so a crash is visible
# without flooding the console across a long batch - the full traceback
# is still whatever the subprocess itself printed to the terminal.
CRASH_STDERR_TAIL_LINES = 6


def run_batch(num_trials: int = 50) -> list:
    BATCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i in range(num_trials):
        domain = random.choice(DOMAINS)
        print(f"\n[Batch] === Trial {i + 1}/{num_trials} - domain={domain} ===")

        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--domain", domain],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )

        crashed = proc.returncode != 0
        status = "CRASHED" if crashed else "OK"
        print(f"[Batch] Trial {i + 1} -> {status} (exit code {proc.returncode})")

        if crashed:
            tail = "\n".join(proc.stderr.strip().splitlines()[-CRASH_STDERR_TAIL_LINES:])
            print(f"[Batch]   stderr tail:\n{tail}")

        results.append({
            "trial": i + 1,
            "domain": domain,
            "status": status,
            "returncode": proc.returncode,
        })

    ok_count = sum(1 for r in results if r["status"] == "OK")
    crash_count = num_trials - ok_count
    print(f"\n[Batch] Complete: {ok_count}/{num_trials} succeeded, {crash_count} crashed.")

    _append_batch_log(results, ok_count, num_trials)
    return results


def _append_batch_log(results: list, ok_count: int, num_trials: int) -> None:
    with open(BATCH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## Batch run {datetime.now().isoformat()}\n\n")
        f.write(f"{ok_count}/{num_trials} succeeded, {num_trials - ok_count} crashed.\n\n")
        for r in results:
            f.write(f"- Trial {r['trial']}: domain={r['domain']} status={r['status']} (exit code {r['returncode']})\n")
        # Domain breakdown - is crashing correlated with a particular domain, or spread evenly?
        f.write("\n### By domain\n")
        for domain in DOMAINS:
            domain_results = [r for r in results if r["domain"] == domain]
            if not domain_results:
                continue
            domain_ok = sum(1 for r in domain_results if r["status"] == "OK")
            f.write(f"- {domain}: {domain_ok}/{len(domain_results)} succeeded\n")
    print(f"[Batch] Log appended to {BATCH_LOG_PATH}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_batch(n)
