# AI Evaluation Engine Backend (v1) — Schema-Driven DeepEval Runner

This backend implements an end-to-end evaluation service for your AI Evaluation Engine UI.

It is **schema-driven**: the YAML metric schema is the source of truth for:
- which metrics exist (catalog)
- which DeepEval metric class to instantiate
- which test-case fields are required per metric
- which constructor/init params are required per metric
- threshold semantics (minimum/maximum is passing)
- constraints (e.g., image input/output requirements)

The backend accepts frontend inputs aligned to your UX flow:
**Object → Use Case → Context → Metrics → Test Cases → Run Controls**  
and returns decision-first outputs:
**Overall status + per-metric results + evidence artifact pointer**.

---

## File-by-file Purpose

### `app.py`
**Purpose:** FastAPI application entrypoint.

**Responsibilities:**
- Defines the HTTP API (e.g., `POST /v1/evaluate`)
- Validates requests at the API boundary (Pydantic models)
- Calls the evaluation runner
- Writes evidence artifacts
- Computes a v1 “overall status” (PASS/WARNING/FAIL) based on metric results
- Returns the response payload to the frontend

This is the file you run with `uvicorn`.

---

### `schemas_api.py`
**Purpose:** API contract definition between frontend and backend.

**Responsibilities:**
- Defines Pydantic request/response models used by FastAPI
- Keeps the payload shape aligned to UI concepts:
  - context panel fields
  - run controls
  - metric selection (metric_id + init_params + threshold)
  - test case payloads (loose shape; schema enforces per-metric requirements)

This file keeps API typing strict while letting the YAML schema drive deeper validation.

---

### `schema_registry.py`
**Purpose:** Load and index your YAML metric schema.

**Responsibilities:**
- Loads `deepEval_metrics.schema.yaml`
- Builds a `metric_id -> MetricDef` index so runtime lookups are fast and consistent
- Normalizes schema fields into a typed `MetricDef` structure (metric name, class, required fields, constraints, etc.)

This is the core piece that makes the service “schema-driven”.

---

### `deepeval_resolver.py`
**Purpose:** Resolve schema `metric_class` names into actual DeepEval Python classes.

**Responsibilities:**
- Converts a string like `"FaithfulnessMetric"` into a Python class reference
- In v1, tries `deepeval.metrics.<MetricClassName>` first
- Provides a single extension point for “import map fixes” if DeepEval version differences require explicit imports

If a metric class cannot be found, errors are surfaced cleanly to the frontend.

---

### `test_case_adapter.py`
**Purpose:** Convert frontend test case payloads into DeepEval test case objects.

**Responsibilities:**
- Implements `dict -> LLMTestCase` mapping for single-turn / batch evaluation (v1 supported)
- Defines placeholders for:
  - `ConversationalTestCase` mapping (not implemented in v1)
  - `ArenaTestCase` mapping (not implemented in v1)

This is where “Conversation Trace” becomes a DeepEval multi-turn object in a future iteration.

---

### `runner.py`
**Purpose:** Core evaluation orchestration engine.

**Responsibilities:**
- Runs schema-driven validation:
  - required test-case fields (per metric)
  - required init/constructor params (per metric)
  - constraints enforcement (e.g., image count rules)
- Instantiates selected DeepEval metrics via `deepeval_resolver.py`
- Converts test cases via `test_case_adapter.py`
- Executes `metric.measure(test_case)` and extracts:
  - score
  - reason (if exposed by metric)
  - pass/fail (using schema threshold semantics)
- Produces:
  - per-metric results list
  - evidence structure containing inputs, scores, reasons, and gap markers

This is the “evaluation core” that your UI calls.

---

### `evidence_store.py`
**Purpose:** Immutable evidence artifact persistence.

**Responsibilities:**
- Writes evidence JSON into `artifacts/<run_id>/evidence.json`
- Returns an `evidence_pointer` path that the UI can use to fetch/download evidence later

In a production setup, you can swap this implementation to write to S3/GCS, a database, or a document store.

---

### `config.py`
**Purpose:** Central configuration for paths and environment defaults.

**Responsibilities:**
- Defines where the schema file lives
- Defines where artifacts are written
- Ensures the artifact directory exists

This isolates filesystem/config concerns from business logic.

---

### `requirements.txt`
**Purpose:** Python dependency manifest.

**Responsibilities:**
- Defines required packages to run the backend service:
  - FastAPI, Uvicorn
  - Pydantic
  - PyYAML
  - DeepEval

---

### `deepEval_metrics.schema.yaml`
**Purpose:** Metric schema source of truth.

**Responsibilities:**
- Defines the metric catalog across evaluation types (RAG, Safety, Agents, Vision, etc.)
- Defines, per metric:
  - `metric_id` (stable identifier used by UI + API)
  - `metric_class` (DeepEval metric class to instantiate)
  - `test_case_type` (LLMTestCase / ConversationalTestCase / ArenaTestCase)
  - `required_test_case_fields` (schema-driven validation)
  - `required_metric_init_params` (constructor validation)
  - `threshold_semantics` (minimum/maximum passing rules)
  - constraints and notes for UI/backend behavior

The UI should use this schema to render metric selection and configuration forms, and the backend uses it to validate and execute.

---

## What Works in v1 vs What is Explicitly Deferred

### Works end-to-end in v1
- Metrics with `test_case_type: LLMTestCase`
- Schema-driven validation (fields, init params, threshold semantics)
- Artifact creation and evidence pointer return

### Deferred (explicit gaps)
- `ConversationalTestCase` execution (needs trace → test case adapter)
- `ArenaTestCase` execution
- Budget enforcement (max tokens / max cost)
- Baseline comparisons and regression deltas
- Full conditional field validation logic for complex MCP scenarios

The system surfaces these gaps as clear errors (not silent failures).

---

## Running Locally

From `backend/`:

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
