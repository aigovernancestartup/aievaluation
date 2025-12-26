# How It Works - AI Evaluation Engine Architecture

This document explains the internal architecture and data flow of the AI Evaluation Engine backend.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Breakdown](#component-breakdown)
4. [Request/Response Flow](#requestresponse-flow)
5. [Key Data Structures](#key-data-structures)
6. [Metric Execution Pipeline](#metric-execution-pipeline)
7. [Configuration](#configuration)
8. [Extension Points](#extension-points)

---

## High-Level Overview

The AI Evaluation Engine is a FastAPI-based backend that evaluates LLM outputs using the **DeepEval** library. It provides a unified REST API to run various evaluation metrics (relevancy, hallucination, faithfulness, toxicity, bias, etc.) against test cases.

**Key Features:**
- Schema-driven metric definitions (YAML)
- Automatic Ollama model injection for LLM-based metrics
- Evidence persistence for audit trails
- Support for RAG, safety, and custom evaluation metrics

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT REQUEST                                  │
│                    POST /v1/aievaluation/evaluate                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              app.py (FastAPI)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Validates request via Pydantic models (schemas_api.py)           │   │
│  │  • Loads schema & builds metric index on startup                    │   │
│  │  • Delegates to runner.py for execution                             │   │
│  │  • Writes evidence via evidence_store.py                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            runner.py (Orchestration)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  For each metric:                                                    │   │
│  │    1. Look up MetricDef from schema_registry                        │   │
│  │    2. Validate required fields & init params                        │   │
│  │    3. Auto-inject Ollama model if needed                            │   │
│  │    4. Resolve metric class via deepeval_resolver                    │   │
│  │    5. Instantiate metric with init_params                           │   │
│  │    6. Convert test cases via test_case_adapter                      │   │
│  │    7. Call metric.measure(test_case) for each                       │   │
│  │    8. Aggregate scores, determine pass/fail                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │schema_registry│  │deepeval_     │  │test_case_    │
         │              │  │resolver      │  │adapter       │
         │ • Load YAML  │  │ • Import     │  │ • dict →     │
         │ • Build      │  │   DeepEval   │  │   LLMTestCase│
         │   MetricDef  │  │   classes    │  │              │
         └──────────────┘  └──────────────┘  └──────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DeepEval Library                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • AnswerRelevancyMetric, HallucinationMetric, etc.                 │   │
│  │  • Each metric.measure(test_case) → score, reason                  │   │
│  │  • Uses Ollama (via GPTModel) for LLM-based evaluation             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Ollama (Local LLM Server)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Runs deepseek-r1:1.5b (or configured model)                      │   │
│  │  • Provides OpenAI-compatible API at localhost:11434                │   │
│  │  • Used as the "judge" for LLM-based metrics                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          evidence_store.py                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Persists full evaluation evidence to artifacts/<run_id>/         │   │
│  │  • JSON format with scores, reasons, metadata                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. `app.py` - FastAPI Application

**Purpose:** HTTP entry point and request/response handling.

**Responsibilities:**
- Defines the `/v1/aievaluation/evaluate` POST endpoint
- Validates incoming requests using Pydantic models
- Loads the metric schema on startup
- Calls `run_evaluation()` and returns structured responses
- Writes evidence files for audit trails

**Key Code Flow:**
```python
@router.post("/evaluate")  # Mounted at /v1/aievaluation
def evaluate(req: EvaluateRequest):
    # 1. Convert Pydantic models to dicts
    # 2. Call run_evaluation() 
    # 3. Write evidence to disk
    # 4. Determine overall pass/fail/warning
    # 5. Return EvaluateResponse
```

---

### 2. `schemas_api.py` - Pydantic Models

**Purpose:** Request/response validation and serialization.

**Key Models:**

| Model | Purpose |
|-------|---------|
| `EvaluateRequest` | Incoming API request body |
| `EvaluateResponse` | API response structure |
| `TestCaseModel` | Individual test case (input, actual_output, context, etc.) |
| `MetricSelectionModel` | Which metric to run + threshold + init params |
| `ContextModel` | Deployment context (stage, risk, impact) |
| `MetricResult` | Per-metric result (score, passed, reason, error) |

---

### 3. `schema_registry.py` - Metric Definition Loader

**Purpose:** Load and index metric definitions from YAML schema.

**Key Structures:**

```python
@dataclass
class MetricDef:
    metric_id: str                      # e.g., "rag.answer_relevancy"
    metric_name: str                    # e.g., "Answer Relevancy"
    metric_class: str                   # e.g., "AnswerRelevancyMetric"
    test_case_type: str                 # "LLMTestCase" | "ConversationalTestCase" | "ArenaTestCase"
    required_test_case_fields: List     # e.g., ["input", "actual_output"]
    required_metric_init_params: List   # Required constructor args
    optional_metric_init_params: List   # Optional constructor args
    threshold_semantics: str            # "minimum_is_passing" | "maximum_is_passing"
    constraints: List                   # e.g., ["input must contain exactly 1 image"]
    conditional_fields: List            # Fields required under certain conditions
    notes: List                         # Documentation notes
```

**Functions:**
- `load_schema(path)` → Parse YAML file
- `build_metric_index(schema)` → Dict[metric_id, MetricDef]

---

### 4. `deepeval_resolver.py` - Dynamic Class Loading

**Purpose:** Dynamically import DeepEval metric classes by name.

**How it works:**
```python
def resolve_metric_class(metric_class_name: str):
    import deepeval.metrics as metrics_mod
    cls = getattr(metrics_mod, metric_class_name, None)
    if cls:
        return cls
    raise ImportError(f"Cannot resolve: {metric_class_name}")
```

**Example:**
- Input: `"AnswerRelevancyMetric"`
- Output: `<class deepeval.metrics.AnswerRelevancyMetric>`

---

### 5. `runner.py` - Evaluation Orchestration

**Purpose:** Core execution engine that runs metrics against test cases.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `run_evaluation()` | Main entry point - orchestrates full evaluation |
| `_validate_required_fields()` | Check test case has required fields |
| `_validate_required_init_params()` | Check metric init params are provided |
| `_apply_constraints()` | Validate image count and other constraints |
| `_pass_fail()` | Determine pass/fail based on threshold semantics |
| `_configure_ollama_model()` | Auto-inject Ollama model if metric needs one |

**Execution Loop:**
```python
for each metric in request.metrics:
    1. Look up MetricDef from schema index
    2. Validate init params
    3. Auto-inject Ollama model if needed
    4. Validate all test cases have required fields
    5. Instantiate metric class with init_params
    6. For each test case:
       - Convert to LLMTestCase
       - Call metric.measure(test_case)
       - Collect score and reason
    7. Average scores, determine pass/fail
    8. Append to results
```

---

### 6. `evidence_store.py` - Persistence

**Purpose:** Save evaluation evidence for audit/debugging.

**Output Location:** `backend/artifacts/<run_id>/evidence.json`

**Evidence Structure:**
```json
{
  "run_id": "abc123...",
  "started_at": 1703123456.789,
  "evaluation_object": "my-chatbot",
  "use_case": "Customer support",
  "context": { "deployment_stage": "dev", ... },
  "metrics": [...],
  "test_cases": [...],
  "metric_evidence": [
    {
      "metric_id": "rag.answer_relevancy",
      "metric_class": "AnswerRelevancyMetric",
      "scores": [0.85, 0.92],
      "reasons": ["The answer is relevant...", ...]
    }
  ],
  "gaps": []
}
```

---

### 7. `config.py` - Configuration

**Purpose:** Centralized configuration with environment variable support.

| Setting | Default | Env Variable |
|---------|---------|--------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | `OLLAMA_BASE_URL` |
| `OLLAMA_MODEL` | `deepseek-r1:1.5b` | `OLLAMA_MODEL` |
| `OLLAMA_API_KEY` | `ollama` | `OLLAMA_API_KEY` |
| `OLLAMA_COST_PER_INPUT_TOKEN` | `0.0` | `OLLAMA_COST_PER_INPUT_TOKEN` |
| `OLLAMA_COST_PER_OUTPUT_TOKEN` | `0.0` | `OLLAMA_COST_PER_OUTPUT_TOKEN` |

---

## Request/Response Flow

### Request Example

```json
POST /v1/aievaluation/evaluate
{
  "evaluation_object": "customer-support-bot",
  "use_case": "Answer product questions",
  "context": {
    "deployment_stage": "dev",
    "risk_class": "low",
    "user_impact": "customer_facing",
    "domain": "general"
  },
  "run": {
    "mode": "one_off",
    "environment": "local"
  },
  "metrics": [
    {
      "metric_id": "rag.answer_relevancy",
      "threshold": 0.7,
      "init_params": {}
    }
  ],
  "test_cases": [
    {
      "input": "How do I return a product?",
      "actual_output": "Visit our returns portal at...",
      "retrieval_context": ["Returns policy allows 30 days..."]
    }
  ]
}
```

### Response Example

```json
{
  "run_id": "755d645db23443179bd38265ac8091aa",
  "overall_status": "PASS",
  "metric_results": [
    {
      "metric_id": "rag.answer_relevancy",
      "metric_name": "Answer Relevancy",
      "score": 0.85,
      "threshold": 0.7,
      "passed": true,
      "reason": "The answer directly addresses the question...",
      "error": null,
      "cost_signal": "llm_based"
    }
  ],
  "evidence_pointer": "backend/artifacts/755d645db.../evidence.json"
}
```

---

## Key Data Structures

### Test Case Types

| Type | Required Fields | Use Case |
|------|-----------------|----------|
| `LLMTestCase` | `input`, `actual_output` | Single-turn Q&A evaluation |
| `ConversationalTestCase` | `turns` | Multi-turn conversation evaluation |
| `ArenaTestCase` | `input`, `model_a_output`, `model_b_output` | A/B comparison |

### Threshold Semantics

| Semantic | Meaning | Example Metrics |
|----------|---------|-----------------|
| `minimum_is_passing` | Higher is better (score ≥ threshold = PASS) | Relevancy, Faithfulness |
| `maximum_is_passing` | Lower is better (score ≤ threshold = PASS) | Toxicity, Bias, Hallucination |

---

## Metric Execution Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Request    │───▶│   Validate   │───▶│  Instantiate │───▶│   Measure    │
│              │    │              │    │              │    │              │
│ metric_id    │    │ • init params│    │ MetricCls(   │    │ metric.      │
│ threshold    │    │ • test case  │    │   **params   │    │ measure(     │
│ init_params  │    │   fields     │    │ )            │    │   test_case  │
│ test_cases   │    │ • constraints│    │              │    │ )            │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────┐
                                                           │   Aggregate  │
                                                           │              │
                                                           │ • avg score  │
                                                           │ • pass/fail  │
                                                           │ • evidence   │
                                                           └──────────────┘
```

---

## Extension Points

### Adding a New Metric

1. **Add to Schema** (`deepEval_metrics.schema.yaml`):
   ```yaml
   - metric_name: My Custom Metric
     metric_id: custom.my_metric
     metric_class: MyCustomMetric
     test_case_type: LLMTestCase
     required_test_case_fields: [input, actual_output]
     ...
   ```

2. **Ensure DeepEval Support**: The metric class must exist in `deepeval.metrics`

3. **Test**: Call the API with `metric_id: "custom.my_metric"`

### Using a Different LLM

Override via environment variables:
```bash
export OLLAMA_MODEL="llama3:8b"
export OLLAMA_BASE_URL="http://my-server:11434/v1"
```

Or use OpenAI-compatible endpoints by changing the base URL and providing an API key.

### Adding New Test Case Types

1. Update `test_case_adapter.py` with new adapter function
2. Update `runner.py` to handle the new `test_case_type`
3. Add type definition to schema YAML

---

## File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~53 | FastAPI endpoints |
| `runner.py` | ~234 | Evaluation orchestration |
| `schema_registry.py` | ~45 | Schema loading & indexing |
| `deepeval_resolver.py` | ~21 | Dynamic class import |
| `evidence_store.py` | ~13 | Evidence persistence |
| `schemas_api.py` | ~65 | Pydantic models |
| `config.py` | ~15 | Configuration |
| `deepEval_metrics.schema.yaml` | ~760 | Metric definitions |

---

## See Also

- [HOW_TO_RUN.md](./HOW_TO_RUN.md) - Setup and running instructions
- [DeepEval Documentation](https://deepeval.com/docs) - Upstream library docs
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework docs

