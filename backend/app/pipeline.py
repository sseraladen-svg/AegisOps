"""One command from empty database to investigated incidents.

    python -m app.pipeline                    # full run, needs ANTHROPIC_API_KEY
    python -m app.pipeline --offline          # same, scripted agent instead of Claude
    python -m app.pipeline --skip-seed        # keep the data, rerun detection onward
    python -m app.pipeline --no-investigate   # stop after the frozen numbers

Stages run in order and each one's output is the next one's input:

    seed -> naive -> isolation_forest -> lstm_autoencoder -> eval -> investigate

The investigate stage only picks up incidents that have no investigation yet,
so rerunning the pipeline over an existing database costs nothing for work
already done. A fresh seed makes every incident new, which is why the stage is
capped: 400-odd incidents is not a batch anyone meant to send to a model.
"""
import argparse
import sys
import time
import traceback
from datetime import datetime, timezone

from app.config import DETECTORS, RESULTS_PATH
from app.db import session_scope
from app.models import Incident, Investigation

DEFAULT_MAX_INVESTIGATIONS = 5


class StageFailed(Exception):
    """A stage failed; downstream stages cannot be trusted and are skipped."""


def _run_stage(name, function, results, verbose=True):
    if verbose:
        print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
    started = time.monotonic()
    try:
        detail = function()
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - the summary must survive any stage
        detail = f"{type(exc).__name__}: {exc}"
        status = "failed"
        if verbose:
            traceback.print_exc()
    elapsed = round(time.monotonic() - started, 1)
    results.append({"stage": name, "status": status, "seconds": elapsed, "detail": detail})
    if verbose:
        print(f"--- {name}: {status} in {elapsed}s")
    if status == "failed":
        raise StageFailed(name)
    return detail


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage_seed():
    from app import seed
    seed.main()
    with session_scope() as session:
        return {"incidents_cleared": session.query(Incident).count()}


def stage_detect_naive():
    from app import detector_naive
    detector_naive.main()
    return _incident_count('naive')


def stage_detect_isolation_forest():
    from app import detector_isolation_forest
    detector_isolation_forest.main()
    return _incident_count('isolation_forest')


def stage_train_lstm():
    from app import detector_lstm_autoencoder
    summary = detector_lstm_autoencoder.run(verbose=True)
    return {"threshold": round(summary["threshold"], 5),
            "flagged_windows": summary["flagged_windows"],
            "incidents": summary["incident_windows"]}


def stage_eval():
    from app import eval_harness
    from app.db import get_session
    session = get_session()
    try:
        data = eval_harness.build_results(session)
    finally:
        session.close()
    eval_harness.write_results(data)
    return {
        "path": str(RESULTS_PATH),
        "detectors": {name: {"recall": stats["recall"], "incidents": stats["incidents"],
                             "false_positives": stats["false_positives"]}
                      for name, stats in data["detectors"].items()},
    }


def _incident_count(detector_source):
    with session_scope() as session:
        return {"incidents": session.query(Incident).filter(
            Incident.detector_source == detector_source).count()}


def find_uninvestigated(session, detector_source, limit):
    """Highest-scoring incidents from one detector that have no report yet."""
    investigated = {i for (i,) in session.query(Investigation.incident_id).distinct().all()}
    candidates = session.query(Incident).filter(
        Incident.detector_source == detector_source,
    ).order_by(Incident.anomaly_score.desc()).all()
    return [inc for inc in candidates if inc.id not in investigated][:limit]


def stage_investigate(detector_source, limit, offline, use_sample):
    from app.investigator import build_client, investigate, select_sample_incidents

    client = build_client(offline=offline)
    if offline:
        print("OFFLINE: responses come from app/offline_agent.py, not from Claude.")

    summaries = []
    with session_scope() as session:
        if use_sample:
            # The curated sample spans all three tiers plus false positives,
            # which is what makes a demo run show something worth looking at.
            targets = [inc for inc, _ in select_sample_incidents(session, detector_source, n=limit)]
            already = {i for (i,) in session.query(Investigation.incident_id).distinct().all()}
            targets = [t for t in targets if t.id not in already]
        else:
            targets = find_uninvestigated(session, detector_source, limit)

        if not targets:
            print("  nothing new to investigate")
            return {"investigated": 0, "skipped_existing": True}

        print(f"  investigating {len(targets)} incident(s): "
              + ", ".join(f"#{t.id}" for t in targets))
        for incident in targets:
            summaries.append(investigate(session, incident, client, verbose=True))

    warnings = sum(len(s["validation_warnings"]) for s in summaries)
    return {
        "investigated": len(summaries),
        "validation_warnings": warnings,
        "incident_ids": [s["incident_id"] for s in summaries],
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-seed', action='store_true',
                        help="keep existing telemetry; rerun detection onward")
    parser.add_argument('--no-investigate', action='store_true',
                        help="stop after writing eval/results.json")
    parser.add_argument('--offline', action='store_true',
                        help="use the scripted stub instead of the Claude API")
    parser.add_argument('--detector', default='lstm_autoencoder', choices=DETECTORS,
                        help="which detector's incidents to investigate")
    parser.add_argument('--max-investigations', type=int, default=DEFAULT_MAX_INVESTIGATIONS,
                        help=f"cap on incidents sent to the agent (default {DEFAULT_MAX_INVESTIGATIONS})")
    parser.add_argument('--all-incidents', action='store_true',
                        help="investigate by score rather than the curated tier sample")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    print(f"pipeline started {started.isoformat(timespec='seconds')}")

    # Fail fast on a missing credential rather than after the ~2 minutes of
    # training that precedes the stage that needs it.
    if not args.no_investigate:
        from app.investigator import build_client
        try:
            build_client(offline=args.offline)
        except RuntimeError as exc:
            raise SystemExit(f"\ncannot run the investigate stage: {exc}") from exc

    results = []
    try:
        if args.skip_seed:
            print("\n=== seed: skipped (--skip-seed)")
            results.append({"stage": "seed", "status": "skipped", "seconds": 0, "detail": None})
        else:
            _run_stage("seed", stage_seed, results)

        _run_stage("detect:naive", stage_detect_naive, results)
        _run_stage("detect:isolation_forest", stage_detect_isolation_forest, results)
        _run_stage("detect:lstm_autoencoder", stage_train_lstm, results)
        _run_stage("eval", stage_eval, results)

        if args.no_investigate:
            print("\n=== investigate: skipped (--no-investigate)")
            results.append({"stage": "investigate", "status": "skipped",
                            "seconds": 0, "detail": None})
        else:
            _run_stage(
                "investigate",
                lambda: stage_investigate(args.detector, args.max_investigations,
                                          args.offline, not args.all_incidents),
                results)
    except StageFailed as exc:
        print(f"\npipeline aborted: stage '{exc}' failed; later stages were not run")

    _print_summary(results, started)
    return 1 if any(r["status"] == "failed" for r in results) else 0


def _print_summary(results, started):
    total = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    print("\n" + "=" * 68)
    print(f"{'stage':<28} {'status':<9} {'seconds':>8}")
    print("-" * 68)
    for row in results:
        print(f"{row['stage']:<28} {row['status']:<9} {row['seconds']:>8}")
    print("-" * 68)
    print(f"{'total':<28} {'':<9} {total:>8}")

    evaluated = next((r for r in results if r["stage"] == "eval" and r["status"] == "ok"), None)
    if evaluated:
        print("\ndetector comparison (recall by tier: obvious / drift / correlated):")
        for name, stats in evaluated["detail"]["detectors"].items():
            recall = " / ".join(f"{v:.2f}" for v in stats["recall"])
            print(f"  {name:<20} recall {recall}   "
                  f"{stats['incidents']:>3} incidents, {stats['false_positives']:>3} false positives")

    investigated = next((r for r in results if r["stage"] == "investigate"
                         and r["status"] == "ok"), None)
    if investigated and investigated["detail"].get("investigated"):
        detail = investigated["detail"]
        print(f"\ninvestigated {detail['investigated']} incident(s) "
              f"({detail['validation_warnings']} validation warning(s)): "
              + ", ".join(f"#{i}" for i in detail["incident_ids"]))

    print("\nnext: `make api` then open http://localhost:8000/docs")


if __name__ == "__main__":
    sys.exit(main())
