# agentic-copilot

Anomaly detection over synthetic service telemetry, an LLM agent that
investigates what the detectors find, and a dashboard that shows its work.

Three services (`auth-service`, `payment-gateway`, `user-profile`) emit four
metrics each — `cpu_usage`, `latency_ms`, `error_rate`, `request_rate` — every
five minutes for fourteen days, plus a correlated application log stream. Three
labelled anomalies of increasing subtlety are injected into that data, and three
detectors compete to find them.

## Quick start

```bash
docker compose up --build     # generates data, detects, serves API + dashboard
```

Then open **http://localhost:5173** (dashboard) or
**http://localhost:8000/docs** (API).

Without Docker:

```bash
make pipeline OFFLINE=1   # install, generate, detect x3, score, investigate
make api                  # http://localhost:8000
make web                  # http://localhost:5173, in another shell
```

Drop `OFFLINE=1` once `ANTHROPIC_API_KEY` is set to run the real agent instead
of the scripted stub.

| Target | What it does |
|---|---|
| `make pipeline` | seed → detect ×3 → score → investigate, with per-stage timing |
| `make seed` | regenerate telemetry, logs, and ground truth |
| `make detect` / `make train` | run the detectors individually |
| `make eval` | rescore and rewrite `eval/results.json` |
| `make investigate` | investigate 5 incidents spanning all three tiers |
| `make api` / `make web` | backend and frontend dev servers |
| `make build` | lint and production-build the frontend |
| `make clean` | drop the venv, database, model artifacts, and `dist/` |

## The three detectors

| Detector | Sees | Threshold chosen by |
|---|---|---|
| `naive` | one series at a time | fixed z-score against a causal EWMA baseline |
| `isolation_forest` | one series at a time, rolling-window features | contamination grid **tuned against ground-truth labels** |
| `lstm_autoencoder` | all 12 (service × metric) channels at once | percentile of reconstruction error on an **unlabelled** validation split |

The three difficulty tiers exist to separate them:

- **`obvious_spike`** — a large CPU step change. Everything catches this.
- **`gradual_drift`** — latency ramping linearly over twelve hours. No single
  point is anomalous, so a threshold detector cannot see it.
- **`subtle_correlated`** — a small simultaneous CPU rise on two services. Each
  service alone looks like noise; only a model that sees services *together*
  has the information to catch it.

Frozen results (`eval/results.json`, regenerate with `make eval`):

| Detector | Recall (spike / drift / correlated) | Incidents fired | False positives |
|---|---|---|---|
| naive | 1.00 / 0.00 / 0.00 | 102 | 101 |
| isolation_forest | 1.00 / 1.00 / 1.00 | 259 | 251 |
| lstm_autoencoder | 1.00 / 1.00 / 1.00 | **47** | **42** |

The autoencoder reaches the same perfect recall as the Isolation Forest while
firing a fifth as many alarms — and it does so without ever looking at a label,
whereas the Isolation Forest's contamination is picked against the same ground
truth the table reports on. `results.json` records that asymmetry per detector
in a `tuning` field, because a comparison table that hid it would be misleading.

Absolute precision is low for every detector: there are four ground-truth
anomalies in fourteen days, so even a well-behaved detector's false positives
dominate the denominator. The comparison *between* detectors is the meaningful
number here, not any single detector's precision.

## The investigation agent

`app/investigator.py` takes one incident and produces a validated report —
hypothesis, calibrated confidence, cited evidence, ruled-out alternatives, and a
recommended action — by way of a Claude tool-use loop with four tools:

| Tool | Returns |
|---|---|
| `query_metrics` | window statistics plus the preceding equal-length window, with percent change |
| `query_logs` | log lines for a window, plus per-level counts for the whole window |
| `query_similar_incidents` | nearest of 15 past write-ups, from a FAISS index |
| `file_github_issue` | **dry run by default** — check the `filed` field |

Every request, tool call, and tool result is recorded in a trace persisted to
`investigations.tool_calls_json`, served at
`GET /api/incidents/{id}/investigation`, and rendered step by step on the
incident page.

Reports are then checked mechanically, not just asked for politely: every
evidence item must name a tool that was actually called, must contain a concrete
value, and cannot claim an issue was filed when the tool result says otherwise.
Violations are recorded on the investigation as `validation_warnings` and shown
in the UI.

`file_github_issue` never files for real unless `AGENT_GITHUB_MODE=live` **and**
`GITHUB_TOKEN` **and** `GITHUB_REPO` are all set. A dry run returns the issue it
would have created.

Without an API key, `--offline` swaps in `app/offline_agent.py`: a scripted
stand-in that follows a fixed investigation script and builds its report from
the *real* tool results. It exercises tool dispatch, parallel tool-result
batching, trace capture, report validation, and persistence — everything except
the model. Its output is clearly labelled and is not agent reasoning.

## The dashboard

Four pages, all reading the API:

- **Overview** — per-service status pill, sparkline, and incident count over a
  selectable window, plus a guided tour of the three anomalies showing which
  detectors caught each (computed from the incidents table, not asserted).
- **Timeline** — one metric over the full fourteen days with each detector's
  flagged windows in *its own lane* beneath the plot. Overlapping translucent
  bands composite into a colour nobody can decode; separate lanes make "the
  naive detector missed the drift" visible as a gap.
- **Incidents** — every flagged window, filterable, with the table view of
  everything the charts show.
- **Evaluation** — precision / recall / F1 grouped by difficulty tier, all three
  panels on a shared 0–1 axis so the tiny precision values aren't inflated by
  per-panel scaling, with a table view twin.

Charts are hand-built SVG with crosshair/hover tooltips and keyboard
inspection. The categorical palette is three validated slots (blue / orange /
aqua) that clear the all-pairs colour-vision and contrast floors in both light
and dark mode; a detector's colour follows the detector, so toggling one off
never repaints the others. The metric line itself wears ink rather than a
categorical hue, which keeps all three hues meaning "detector" and nothing else.

## API

`GET /docs` has the full OpenAPI document. The endpoints the dashboard uses:

```
GET  /health
GET  /api/services            GET /api/services/health
GET  /api/metrics             GET /api/logs
GET  /api/incidents           GET /api/incidents/{id}
GET  /api/incidents/{id}/investigation
GET  /api/eval/results        GET /api/ground-truth   GET /api/eval/demo-tour
GET  /api/tools               POST /api/tools/{query_logs,query_metrics,
                                   query_similar_incidents,file_github_issue}
```

The `/api/tools/*` endpoints call the same functions the agent calls in
process — one implementation, two doors onto it.

**Timestamps are naive ISO-8601 in UTC**, and "now" means the last sample in the
database, not wall-clock time. The frontend has a `toApiTs` helper for the
round trip; `new Date(...).toISOString()` would silently shift every query by
the viewer's UTC offset.

## Layout

```
backend/app/
  config.py                     paths and constants shared across modules
  db.py  models.py              SQLAlchemy engine, session scope, schema
  seed.py                       simulator, anomaly injector, log generator
  detector_utils.py             series loading, flag->window merging, incident writes
  detector_naive.py             z-score / EWMA baseline
  detector_isolation_forest.py  per-series Isolation Forest
  windowing.py                  multivariate (window, service, metric) tensors
  detector_lstm_autoencoder.py  LSTM autoencoder + percentile threshold
  eval_harness.py               tiered scoring -> eval/results.json
  knowledge_base.py             FAISS index over past incident write-ups
  agent_tools.py                the four tools + their wire schemas
  investigator.py               the Claude tool-use loop
  offline_agent.py              scripted stand-in for running the loop with no API key
  pipeline.py                   the one-command orchestration
  schemas.py  main.py           response models and the FastAPI surface
backend/data/past_incidents.json   15 past write-ups
frontend/src/
  api/                          client + fetch hook
  components/                   charts, tooltips, shared UI
  pages/                        Overview, Timeline, Incidents, IncidentDetail, Evaluation
eval/results.json                  the frozen detector comparison
```

Python scripts run as modules from `backend/` (`python -m app.seed`), which is
the same import root `uvicorn app.main:app` uses.

## Reproducibility

Fixed RNG seeds and a fixed `START_TIME` anchor, single-threaded torch, and a
deterministic train/validation split — `make pipeline` reproduces the numbers
above exactly, from an empty database, in about 25 seconds. The detectors are
scored only against `ground_truth_anomalies`, which nothing but `seed.py` ever
writes to.
