"""A scripted stand-in for the Claude client, for running the investigation
loop with no API key.

This is a test harness, not a model. It follows a fixed investigation script
and assembles its report from the *real* tool results the loop hands back, so
it exercises everything around the model — tool dispatch and error handling,
parallel tool_result batching, trace capture, report validation, and database
persistence — deterministically and for free. It does no reasoning, and its
hypotheses are template text; never present its output as an agent's findings.

The surface implemented here is only what app/investigator.py calls:
client.beta.messages.create(...) and client.beta.messages.parse(...).
"""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.investigator import Evidence, InvestigationReport


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _Response:
    content: list
    stop_reason: str
    usage: _Usage = field(default_factory=_Usage)
    stop_details: None = None


@dataclass
class _Parsed:
    parsed_output: InvestigationReport


BRIEF = re.compile(
    r"service_id=(?P<service_id>\d+).*?"
    r"metric:\s+(?P<metric>\w+).*?"
    r"window:\s+(?P<start>\S+) to (?P<end>\S+)",
    re.S,
)


def _parse_brief(messages):
    text = messages[0]["content"]
    match = BRIEF.search(text)
    if not match:
        raise RuntimeError("offline stub could not parse the incident brief")
    return {
        "service_id": int(match["service_id"]),
        "metric_name": match["metric"],
        "ts_start": match["start"],
        "ts_end": match["end"],
    }


def _harvest_tool_results(messages):
    """Every tool result so far, as {tool_name: [parsed output, ...]}.

    The stub's report has to be grounded in real numbers for the validator to
    be exercised meaningfully, and these are where the real numbers are.
    """
    calls = {}
    names_by_id = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, 'type', None) == 'tool_use':
                names_by_id[block.id] = block.name
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                name = names_by_id.get(block["tool_use_id"], "unknown")
                try:
                    calls.setdefault(name, []).append(json.loads(block["content"]))
                except json.JSONDecodeError:
                    pass
    return calls


def _widen(ts_start, ts_end, minutes=30):
    """Look a little either side of the flagged window, as a real investigation would."""
    start = datetime.fromisoformat(ts_start) - timedelta(minutes=minutes)
    end = datetime.fromisoformat(ts_end) + timedelta(minutes=minutes)
    return start.isoformat(), end.isoformat()


class _Messages:
    """Stateless by construction: the turn number is derived from the
    conversation it is handed, never counted on the instance. One client is
    reused across every incident in a run, so instance state would leak from
    one investigation into the next — and would do so silently, since a stub
    that skips straight to its last turn still returns a well-formed response."""

    @staticmethod
    def _turn_number(messages):
        # [user] -> 1; [user, assistant, tool_results] -> 2; and so on.
        return (len(messages) + 1) // 2

    def create(self, *, messages, **_kwargs):
        brief = _parse_brief(messages)
        start, end = _widen(brief["ts_start"], brief["ts_end"])
        turn = self._turn_number(messages)

        if turn == 1:
            # Confirm the flagged metric, and check request_rate in the same
            # turn — two parallel calls, which is what the real loop must batch.
            blocks = [
                _TextBlock("Confirming the flagged series and checking whether traffic explains it."),
                _ToolUseBlock("stub_1a", "query_metrics", {
                    "service_id": brief["service_id"], "metric_name": brief["metric_name"],
                    "ts_start": start, "ts_end": end, "include_baseline": True}),
                _ToolUseBlock("stub_1b", "query_metrics", {
                    "service_id": brief["service_id"], "metric_name": "request_rate",
                    "ts_start": start, "ts_end": end, "include_baseline": True}),
            ]
        elif turn == 2:
            blocks = [
                _TextBlock("Reading logs for the window and looking for comparable past incidents."),
                _ToolUseBlock("stub_2a", "query_logs", {
                    "service_id": brief["service_id"], "ts_start": start,
                    "ts_end": end, "level": "ERROR", "limit": 20}),
                _ToolUseBlock("stub_2b", "query_similar_incidents", {
                    "query": f"{brief['metric_name']} anomaly with correlated log errors", "k": 3}),
            ]
        elif turn == 3:
            harvest = _harvest_tool_results(messages)
            metrics = (harvest.get("query_metrics") or [{}])[0]
            window = metrics.get("window") or {}
            blocks = [
                _TextBlock("Preparing an issue."),
                _ToolUseBlock("stub_3a", "file_github_issue", {
                    "title": (f"[{metrics.get('service_name', 'service')}] "
                              f"{brief['metric_name']} anomaly at {brief['ts_start']}"),
                    "body": (f"Detector flagged {brief['metric_name']} on "
                             f"{metrics.get('service_name', 'service')} between "
                             f"{brief['ts_start']} and {brief['ts_end']}. Window mean "
                             f"{window.get('mean')}, max {window.get('max')}.\n\n"
                             f"(Generated by the offline stub, not by a model.)"),
                    "labels": ["incident", "auto-generated"]}),
            ]
        else:
            blocks = [_TextBlock("Investigation complete.")]

        return _Response(
            content=blocks,
            stop_reason="tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn",
            usage=_Usage(input_tokens=100 * turn, output_tokens=50),
        )

    def parse(self, *, messages, **_kwargs):
        harvest = _harvest_tool_results(messages)
        metric_calls = harvest.get("query_metrics") or []
        log_calls = harvest.get("query_logs") or []
        similar_calls = harvest.get("query_similar_incidents") or []
        github_calls = harvest.get("file_github_issue") or []

        evidence, ruled_out = [], []
        confidence = 0.3

        flagged = next((m for m in metric_calls if m.get("metric_name") != "request_rate"), None)
        traffic = next((m for m in metric_calls if m.get("metric_name") == "request_rate"), None)

        if flagged and flagged.get("window"):
            window = flagged["window"]
            change = (flagged.get("baseline") or {}).get("pct_change_in_mean")
            evidence.append(Evidence(
                claim=f"{flagged['metric_name']} moved outside its baseline on "
                      f"{flagged['service_name']}.",
                source_tool="query_metrics",
                detail=(f"window mean {window['mean']}, max {window['max']}, "
                        f"p95 {window['p95']}; change vs preceding window: {change}%"),
            ))
            confidence += 0.2

        if traffic and traffic.get("baseline", {}).get("pct_change_in_mean") is not None:
            traffic_change = traffic["baseline"]["pct_change_in_mean"]
            evidence.append(Evidence(
                claim="Request rate over the same window did not move proportionally.",
                source_tool="query_metrics",
                detail=f"request_rate mean {traffic['window']['mean']}, "
                       f"change vs preceding window: {traffic_change}%",
            ))
            if abs(traffic_change) < 10:
                ruled_out.append(
                    f"Traffic-driven load: request_rate changed only {traffic_change}% "
                    f"over the same window.")
                confidence += 0.15

        if log_calls:
            logs = log_calls[0]
            counts = logs.get("level_counts", {})
            first = (logs.get("logs") or [{}])[0]
            evidence.append(Evidence(
                claim="The service logged errors during the window.",
                source_tool="query_logs",
                detail=(f"{counts.get('ERROR', 0)} ERROR of "
                        f"{logs.get('total_lines_in_window', 0)} lines; first: "
                        f"{first.get('message', 'none')!r} at {first.get('ts', 'n/a')}"),
            ))
            if counts.get("ERROR", 0) > 0:
                confidence += 0.1

        if similar_calls and similar_calls[0].get("matches"):
            top = similar_calls[0]["matches"][0]
            evidence.append(Evidence(
                claim=f"The pattern resembles past incident {top['id']}.",
                source_tool="query_similar_incidents",
                detail=f"cosine similarity {top['similarity']}: {top['title']}",
            ))

        github = github_calls[0] if github_calls else None
        hypothesis = (
            f"{flagged['metric_name']} on {flagged['service_name']} departed from its "
            f"baseline without a matching change in request rate, which points at a "
            f"work-per-request regression rather than load."
            if flagged else
            "Insufficient tool output to form a hypothesis."
        )

        return _Parsed(InvestigationReport(
            hypothesis=hypothesis + " (Scripted stub output — not model reasoning.)",
            confidence=round(min(confidence, 0.75), 2),
            severity="medium",
            evidence=evidence,
            ruled_out=ruled_out or ["Nothing eliminated: the stub does not reason."],
            recommended_action=(
                "Compare the window against the preceding deploy history for this "
                "service, and check whether the error log lines share a request_id."),
            issue_prepared=github is not None,
            issue_filed=bool(github and github.get("filed")),
        ))


class _Beta:
    def __init__(self):
        self.messages = _Messages()


class ScriptedClient:
    """Deterministic stand-in for anthropic.Anthropic()."""

    def __init__(self):
        self.beta = _Beta()
