"""OpenAPI contract checks.

The frontend treats /openapi.json as the contract for what the API returns —
every page destructures fields it expects the document to promise. Declaring
response models is what makes that true, but nothing enforced it: a new
endpoint added without a `response_model` documents itself as returning an
untyped object, and the omission is invisible until a page renders blank.

These tests are the enforcement. They assert the document generates, that
every endpoint the dashboard calls exists and is typed, and that the schema
graph has no dangling $ref.
"""
import pytest

from app.config import DETECTORS, METRICS, TIERS

# Every path the React client calls, from frontend/src/api/client.js. If a
# rename breaks one of these, the dashboard breaks with it — so the rename
# should break this test first.
CLIENT_PATHS = [
    ("get", "/health"),
    ("get", "/api/services"),
    ("get", "/api/services/health"),
    ("get", "/api/metrics"),
    ("get", "/api/logs"),
    ("get", "/api/incidents"),
    ("get", "/api/incidents/{incident_id}"),
    ("get", "/api/incidents/{incident_id}/investigation"),
    ("get", "/api/eval/results"),
    ("get", "/api/ground-truth"),
    ("get", "/api/eval/demo-tour"),
]

# The four agent tools, over HTTP. Their response shape is deliberately
# untyped (it is the tool's own JSON), but their existence is still contract.
TOOL_PATHS = [
    ("post", "/api/tools/query_logs"),
    ("post", "/api/tools/query_metrics"),
    ("post", "/api/tools/query_similar_incidents"),
    ("post", "/api/tools/file_github_issue"),
]


@pytest.fixture(scope="module")
def openapi():
    from app.main import app
    return app.openapi()


def test_document_generates_and_declares_version(openapi):
    assert openapi["openapi"].startswith("3.")
    assert openapi["info"]["title"] == "agentic-copilot"
    assert openapi["info"]["version"]


@pytest.mark.parametrize("method,path", CLIENT_PATHS + TOOL_PATHS)
def test_every_endpoint_the_client_calls_exists(openapi, method, path):
    assert path in openapi["paths"], f"{path} is missing from the OpenAPI document"
    assert method in openapi["paths"][path], f"{path} does not accept {method.upper()}"


@pytest.mark.parametrize("method,path", CLIENT_PATHS)
def test_client_endpoints_declare_a_typed_success_response(openapi, method, path):
    """A `response_model`-less endpoint documents itself as `{}` — useless as a
    contract. The tool endpoints are exempt (see TOOL_PATHS); these are not."""
    operation = openapi["paths"][path][method]
    schema = (operation["responses"]["200"]["content"]["application/json"]["schema"])
    assert schema, f"{method.upper()} {path} documents an untyped 200 response"
    assert "$ref" in schema or schema.get("type") or "items" in schema or "anyOf" in schema


def test_no_dangling_component_references(openapi):
    """Every $ref resolves. A broken ref makes generated clients fail to build."""
    defined = set(openapi.get("components", {}).get("schemas", {}))

    referenced = set()

    def walk(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                referenced.add(ref.rsplit("/", 1)[-1])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(openapi)
    assert referenced <= defined, f"dangling $ref(s): {sorted(referenced - defined)}"


def test_path_parameters_are_declared(openapi):
    """A templated path segment with no matching parameter silently 404s."""
    for path, operations in openapi["paths"].items():
        expected = {seg[1:-1] for seg in path.split("/")
                    if seg.startswith("{") and seg.endswith("}")}
        if not expected:
            continue
        for method, operation in operations.items():
            declared = {p["name"] for p in operation.get("parameters", [])
                        if p.get("in") == "path"}
            assert expected <= declared, (
                f"{method.upper()} {path} does not declare {sorted(expected - declared)}")


def test_enumerations_match_the_configured_vocabulary(openapi):
    """The document's literals and config.py must not drift apart — the
    frontend hardcodes the same three detector ids and four metric names."""
    schemas = openapi["components"]["schemas"]
    # Pydantic inlines a bare Literal alias rather than emitting a named
    # component, so the enum is read off the property that uses it.
    detector = schemas["HealthOverview"]["properties"]["detector"]
    assert set(detector["enum"]) == set(DETECTORS)

    text = str(openapi)
    for metric in METRICS:
        assert metric in text, f"{metric} is absent from the OpenAPI document"
    for tier in TIERS:
        assert tier in text or tier in str(schemas), f"{tier} is undocumented"


def test_error_responses_are_documented_for_parameterised_paths(openapi):
    """Every endpoint that validates input should document its 422."""
    for method, path in CLIENT_PATHS:
        operation = openapi["paths"][path][method]
        if operation.get("parameters"):
            assert "422" in operation["responses"], (
                f"{method.upper()} {path} takes parameters but documents no 422")
