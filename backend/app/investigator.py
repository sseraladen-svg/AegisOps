"""The investigation agent: one incident in, one structured report out.

Flow, in two phases:

  1. A tool-use loop. Claude is given the incident and the four tools from
     app/agent_tools.py and works until it stops calling tools (or hits
     AGENT_MAX_ITERATIONS). Every request, tool call, and tool result is
     recorded in a trace.
  2. One final call with no tools and a structured output schema, which turns
     the conversation into a validated InvestigationReport.

Splitting the phases is deliberate. Asking for structured output *during* the
tool loop means every intermediate turn is pushed toward emitting a final
answer, which cuts investigations short; and a report produced in the same
turn as a tool call is harder to attribute in the trace. The extra call is
cheap next to the loop it summarises.

This uses a manual loop rather than the SDK's tool runner because the trace is
a product here, not debug output: it's persisted to investigations.tool_calls_json
and rendered step-by-step in the frontend, so the loop needs to own the exact
ordering and timing of every call rather than reconstruct it afterwards.

Run with:
    python -m app.investigator --incident-id 42
    python -m app.investigator --sample          # 5 incidents across all tiers
    python -m app.investigator --sample --offline  # no API key needed
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import List, Literal

from pydantic import BaseModel, Field

from app.agent_tools import TOOL_SCHEMAS, ToolError, dispatch
from app.config import AGENT_MAX_ITERATIONS, AGENT_MAX_TOKENS, AGENT_MODEL
from app.db import session_scope
from app.detector_utils import load_ground_truth, windows_overlap
from app.models import Incident, Investigation, Service

TOOL_NAMES = tuple(schema["name"] for schema in TOOL_SCHEMAS)

# Enabling the server-side refusal fallback means a safety-classifier refusal
# is routed to a capable fallback model instead of returning an empty report
# mid-batch. Drop both lines to opt out.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Evidence(BaseModel):
    claim: str = Field(description="One specific factual claim about this incident.")
    source_tool: Literal[TOOL_NAMES] = Field(  # type: ignore[valid-type]
        description="The tool whose result establishes this claim.")
    detail: str = Field(
        description="The concrete values from that tool result — numbers, "
                    "timestamps, or a quoted log line. Not a paraphrase.")


class InvestigationReport(BaseModel):
    hypothesis: str = Field(description="The most likely explanation, in one or two sentences.")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1. Calibrated, not enthusiastic.")
    severity: Literal["low", "medium", "high"]
    evidence: List[Evidence] = Field(description="Every claim that supports the hypothesis.")
    ruled_out: List[str] = Field(description="Explanations considered and eliminated, with why.")
    recommended_action: str = Field(description="What the on-call engineer should do next.")
    issue_prepared: bool = Field(description="Whether file_github_issue was called.")
    issue_filed: bool = Field(description="Whether that call actually created an issue (its `filed` field).")


SYSTEM_PROMPT = """You are an on-call SRE investigating an anomaly that an \
automated detector flagged in a telemetry system. Three services emit four \
metrics each — cpu_usage, latency_ms, error_rate, request_rate — at 5-minute \
intervals, alongside application logs.

You have four tools: query_metrics, query_logs, query_similar_incidents, and \
file_github_issue. Investigate, then stop.

How to work:

- Start by confirming the anomaly is real: query_metrics on the flagged series \
for the incident window. The result includes the preceding equal-length window \
as a baseline, so you can state how far the metric actually moved.
- Look for a cause, not just a confirmation. The four metrics are causally \
linked — traffic drives CPU, CPU drives latency, latency drives errors. CPU \
that rose in proportion to request_rate is load; CPU that rose without it is a \
regression in work-per-request. Check the neighbouring metrics before concluding.
- Check other services over the same window. Some incidents are only visible \
across services: a small simultaneous move on two unrelated services usually \
means a shared dependency, and looking at one service alone will make it look \
like noise.
- Read the logs for the window. They are where a mechanism gets named.
- Use query_similar_incidents once you can describe the *pattern*. Past write-ups \
are hypotheses to test against this incident's data — never evidence about it. \
If the data doesn't match the past incident, say so and discard it.
- Call file_github_issue last, once, after you have a conclusion.

Rules about what you may assert:

- Every claim in your final report must come from a tool result you actually \
received in this conversation. If you did not call a tool that establishes a \
claim, you may not make it.
- Cite concrete values. "cpu_usage mean rose from 40.7 to 99.6, +144%" is a \
claim; "CPU was elevated" is not. Quote log lines rather than summarising them.
- If the evidence is thin, say so and lower your confidence. A calibrated 0.4 \
with the gap named is worth more than an unsupported 0.9. Reserve confidence \
above 0.8 for a mechanism you can point at a specific log line for.
- State what you ruled out and what observation eliminated it.
- file_github_issue returns a `filed` field. When it is false the issue was \
prepared but not created — report it that way. Never say an issue was filed \
unless `filed` was true.
- If the data does not support any confident explanation, the correct report \
says that and recommends what to collect next."""


# --------------------------------------------------------------------------
# incident brief
# --------------------------------------------------------------------------

def build_incident_brief(session, incident):
    """The user-turn framing for one incident.

    Deliberately does *not* include ground truth or the metric values — the
    agent has to go get those with tools, which is the whole point.
    """
    service = session.get(Service, incident.service_id)
    duration = (incident.ts_end - incident.ts_start).total_seconds() / 60.0
    return (
        f"An anomaly detector flagged the following incident. Investigate it.\n\n"
        f"  incident_id:     {incident.id}\n"
        f"  service:         {service.name if service else 'unknown'} "
        f"(service_id={incident.service_id})\n"
        f"  metric:          {incident.metric_name}\n"
        f"  window:          {incident.ts_start.isoformat()} to {incident.ts_end.isoformat()} "
        f"({duration:.0f} minutes)\n"
        f"  detector:        {incident.detector_source}\n"
        f"  anomaly_score:   {incident.anomaly_score:.4f}\n\n"
        f"Other services you can query: 1=auth-service, 2=payment-gateway, "
        f"3=user-profile. Metrics: cpu_usage, latency_ms, error_rate, request_rate.\n\n"
        f"Work out what happened and why, then produce your findings."
    )


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def _content_to_jsonable(content):
    """Model content blocks -> plain dicts for the persisted trace."""
    blocks = []
    for block in content:
        if getattr(block, 'type', None) == 'text':
            blocks.append({"type": "text", "text": block.text})
        elif getattr(block, 'type', None) == 'tool_use':
            blocks.append({"type": "tool_use", "id": block.id,
                           "name": block.name, "input": block.input})
        elif getattr(block, 'type', None) == 'thinking':
            # Thinking text is empty unless display is "summarized"; record only
            # that a thinking block occurred so the trace shape stays honest.
            blocks.append({"type": "thinking"})
    return blocks


def run_tool_loop(session, client, brief, model=AGENT_MODEL,
                  max_iterations=AGENT_MAX_ITERATIONS, verbose=True):
    """Phase 1. Returns (messages, trace, stop_state)."""
    messages = [{"role": "user", "content": brief}]
    trace = []
    tools_used = set()
    github_result = None
    stop_state = "completed"

    for iteration in range(1, max_iterations + 1):
        started = time.monotonic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=AGENT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
            thinking={"type": "adaptive"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        elapsed_ms = round((time.monotonic() - started) * 1000)

        # Populated only on refusal; guard before reading it.
        if response.stop_reason == "refusal":
            details = getattr(response, 'stop_details', None)
            trace.append({"step": len(trace) + 1, "type": "refusal",
                          "category": getattr(details, 'category', None),
                          "explanation": getattr(details, 'explanation', None)})
            stop_state = "refused"
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if getattr(b, 'type', None) == 'tool_use']
        text_blocks = [b.text for b in response.content if getattr(b, 'type', None) == 'text']

        trace.append({
            "step": len(trace) + 1,
            "type": "model_turn",
            "iteration": iteration,
            "latency_ms": elapsed_ms,
            "stop_reason": response.stop_reason,
            "text": "\n".join(text_blocks),
            "tool_calls": [{"id": b.id, "name": b.name, "input": b.input} for b in tool_uses],
            "usage": {
                "input_tokens": getattr(response.usage, 'input_tokens', None),
                "output_tokens": getattr(response.usage, 'output_tokens', None),
            },
        })

        if not tool_uses:
            break

        # Every tool_result for one assistant turn goes back in a single user
        # message; splitting them across messages teaches the model to stop
        # issuing parallel calls.
        results = []
        for block in tool_uses:
            tools_used.add(block.name)
            tool_started = time.monotonic()
            try:
                output = dispatch(session, block.name, block.input)
                is_error = False
            except ToolError as exc:
                output = {"error": str(exc)}
                is_error = True
            tool_ms = round((time.monotonic() - tool_started) * 1000)

            if block.name == 'file_github_issue' and not is_error:
                github_result = output

            trace.append({
                "step": len(trace) + 1,
                "type": "tool_result",
                "iteration": iteration,
                "tool_use_id": block.id,
                "name": block.name,
                "input": block.input,
                "latency_ms": tool_ms,
                "is_error": is_error,
                "output": output,
            })
            if verbose:
                mark = "!" if is_error else " "
                print(f"    [{iteration}]{mark} {block.name}({_short_args(block.input)})")

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(output, default=str),
                "is_error": is_error,
            })

        messages.append({"role": "user", "content": results})
    else:
        stop_state = "max_iterations"
        if verbose:
            print(f"    stopped: hit max_iterations={max_iterations}")

    return messages, trace, {
        "stop_state": stop_state,
        "tools_used": sorted(tools_used),
        "github_result": github_result,
    }


def _short_args(arguments):
    parts = []
    for key, value in list(arguments.items())[:3]:
        text = str(value)
        parts.append(f"{key}={text[:28] + '...' if len(text) > 28 else text}")
    return ", ".join(parts)


REPORT_REQUEST = (
    "Now write your findings. Use only what the tool results in this "
    "conversation actually showed — every evidence entry must name the tool "
    "that produced it and quote its concrete values."
)


def request_report(client, messages, model=AGENT_MODEL):
    """Phase 2: the conversation, minus tools, as a validated report."""
    response = client.beta.messages.parse(
        model=model,
        max_tokens=AGENT_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[*messages, {"role": "user", "content": REPORT_REQUEST}],
        output_format=InvestigationReport,
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    return response.parsed_output


# --------------------------------------------------------------------------
# validation — the mechanical version of "iterate until claims cite tools"
# --------------------------------------------------------------------------

NUMBER = re.compile(r"\d")


def validate_report(report, stop_state):
    """Check the report against what actually happened in the loop.

    A prompt asking for cited evidence is a request; this is the check. Every
    warning here is a claim the report makes that the trace does not support,
    which is exactly the failure mode that makes an agent report worse than no
    report at all.
    """
    warnings = []
    called = set(stop_state["tools_used"])
    github = stop_state["github_result"]

    if not report.evidence:
        warnings.append("report cites no evidence at all")

    for i, item in enumerate(report.evidence):
        if item.source_tool not in called:
            warnings.append(
                f"evidence[{i}] cites {item.source_tool}, which was never called "
                f"successfully (called: {sorted(called) or 'none'})")
        if not NUMBER.search(item.detail):
            warnings.append(
                f"evidence[{i}] detail contains no concrete value: {item.detail[:80]!r}")

    if report.issue_prepared and 'file_github_issue' not in called:
        warnings.append("report says an issue was prepared but file_github_issue was never called")
    actually_filed = bool(github and github.get('filed'))
    if report.issue_filed and not actually_filed:
        warnings.append("report claims an issue was filed; the tool result says it was not")

    if report.confidence > 0.8 and len(report.evidence) < 2:
        warnings.append(
            f"confidence {report.confidence} above 0.8 on {len(report.evidence)} "
            f"evidence item(s)")

    return warnings


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def investigate(session, incident, client, verbose=True):
    """Investigate one incident and persist the result. Returns a summary dict."""
    brief = build_incident_brief(session, incident)
    if verbose:
        print(f"  investigating incident {incident.id} "
              f"(service={incident.service_id} {incident.metric_name}, "
              f"{incident.detector_source})")

    messages, trace, stop_state = run_tool_loop(session, client, brief, verbose=verbose)

    if stop_state["stop_state"] == "refused":
        report, warnings = None, ["the model refused to answer"]
    else:
        report = request_report(client, messages)
        warnings = validate_report(report, stop_state)

    github = stop_state["github_result"] or {}
    investigation = Investigation(
        incident_id=incident.id,
        tool_calls_json=trace,
        hypothesis=report.hypothesis if report else None,
        confidence=report.confidence if report else None,
        evidence_json={
            "evidence": [e.model_dump() for e in report.evidence] if report else [],
            "ruled_out": report.ruled_out if report else [],
            "recommended_action": report.recommended_action if report else None,
            "severity": report.severity if report else None,
            "stop_state": stop_state["stop_state"],
            "tools_used": stop_state["tools_used"],
            "validation_warnings": warnings,
        },
        github_issue_url=github.get('issue_url'),
    )
    # One investigation per incident; a rerun replaces rather than accumulates.
    session.query(Investigation).filter(Investigation.incident_id == incident.id).delete(
        synchronize_session=False)
    session.add(investigation)

    if verbose:
        if report:
            print(f"    hypothesis: {report.hypothesis}")
            print(f"    confidence: {report.confidence:.2f}  severity: {report.severity}  "
                  f"evidence: {len(report.evidence)} item(s)")
        for warning in warnings:
            print(f"    WARNING: {warning}")

    return {
        "incident_id": incident.id,
        "detector_source": incident.detector_source,
        "service_id": incident.service_id,
        "metric_name": incident.metric_name,
        "hypothesis": report.hypothesis if report else None,
        "confidence": report.confidence if report else None,
        "severity": report.severity if report else None,
        "evidence_count": len(report.evidence) if report else 0,
        "tools_used": stop_state["tools_used"],
        "stop_state": stop_state["stop_state"],
        "steps": len(trace),
        "validation_warnings": warnings,
    }


def select_sample_incidents(session, detector_source='lstm_autoencoder', n=5):
    """Five incidents spanning all three difficulty tiers.

    Picks one true positive per tier first — an investigation of a real anomaly
    is what the report format is for — then tops up with the
    highest-scoring false positives, because an agent that invents a confident
    root cause for noise is the failure this whole exercise has to catch.
    """
    incidents = session.query(Incident).filter(
        Incident.detector_source == detector_source).order_by(
        Incident.anomaly_score.desc()).all()
    if not incidents:
        raise RuntimeError(
            f"no incidents from '{detector_source}' — run the detectors first")

    ground_truth = load_ground_truth(session)

    def tier_of(incident):
        for gt in ground_truth:
            if (incident.service_id, incident.metric_name) == (gt.service_id, gt.metric_name) \
                    and windows_overlap(incident.ts_start, incident.ts_end,
                                        gt.ts_start, gt.ts_end):
                return gt.difficulty_tier
        return None

    chosen, seen_tiers = [], set()
    for incident in incidents:
        tier = tier_of(incident)
        if tier and tier not in seen_tiers:
            seen_tiers.add(tier)
            chosen.append((incident, tier))
    for incident in incidents:
        if len(chosen) >= n:
            break
        if tier_of(incident) is None and all(incident.id != c.id for c, _ in chosen):
            chosen.append((incident, 'false_positive'))
    return chosen[:n]


def build_client(offline=False):
    if offline:
        from app.offline_agent import ScriptedClient
        return ScriptedClient()
    import anthropic
    if not (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN')):
        raise RuntimeError(
            "no Anthropic credentials found. Export ANTHROPIC_API_KEY, or run with "
            "--offline to exercise the loop against the scripted stub instead.")
    return anthropic.Anthropic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--incident-id', type=int, help="investigate one incident by id")
    group.add_argument('--sample', action='store_true',
                       help="investigate 5 incidents spanning all three tiers")
    parser.add_argument('--detector', default='lstm_autoencoder',
                        help="which detector's incidents to sample from")
    parser.add_argument('--offline', action='store_true',
                        help="use the scripted stub instead of the Claude API")
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    client = build_client(offline=args.offline)
    if args.offline:
        print("OFFLINE MODE: responses come from app/offline_agent.py, not from Claude.")

    summaries = []
    with session_scope() as session:
        if args.incident_id is not None:
            incident = session.get(Incident, args.incident_id)
            if incident is None:
                raise SystemExit(f"no incident with id {args.incident_id}")
            targets = [(incident, None)]
        else:
            targets = select_sample_incidents(session, args.detector, n=5)
            print(f"selected {len(targets)} incident(s) from '{args.detector}': "
                  + ", ".join(f"#{i.id}({t})" for i, t in targets))

        for incident, tier in targets:
            summary = investigate(session, incident, client, verbose=not args.quiet)
            summary["tier"] = tier
            summaries.append(summary)

    total_warnings = sum(len(s["validation_warnings"]) for s in summaries)
    print(f"\ninvestigated {len(summaries)} incident(s); "
          f"{total_warnings} validation warning(s) across all reports")
    print(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec='seconds'),
                      "investigations": summaries}, indent=2, default=str))


if __name__ == "__main__":
    main()
