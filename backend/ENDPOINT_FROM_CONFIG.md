# Evaluate from Config Endpoint

## Overview

The `/v1/evaluate-from-config` endpoint allows you to run evaluations using test cases from a configuration file. This endpoint is designed for frontend integration and supports:

- Using a default evaluation config file
- Uploading a custom evaluation config file
- Filtering by goals and metrics
- Providing evaluation metadata (ID, name, context)

## Endpoint

**POST** `/v1/evaluate-from-config`

## Request Format

This endpoint accepts `multipart/form-data` (for file uploads) or `application/x-www-form-urlencoded` (for default config).

### Required Parameters

- `evaluation_id` (string): Unique identifier for this evaluation
- `evaluation_name` (string): Human-readable name for the evaluation

### Optional Parameters

- `goals` (string): Comma-separated list of goal names to filter by (e.g., "Quality & Accuracy, Context Grounding & Hallucination Control")
  - If not provided, all goals are included
- `metrics` (string): Comma-separated list of metric names to filter by (e.g., "Task Completion, Answer Relevancy")
  - If not provided, all metrics are included
- `config_file` (file): JSON config file to upload
  - If not provided, uses default config at `Testing/testingconfig/evaluation_config.json`
- `deployment_stage` (string): One of "dev", "staging", "prod" (default: "dev")
- `risk_class` (string): One of "low", "medium", "high" (default: "low")
- `user_impact` (string): One of "internal", "customer_facing", "regulated" (default: "internal")
- `domain` (string): Optional - One of "healthcare", "finance", "education", "general"
- `mode` (string): One of "one_off", "batch", "regression" (default: "one_off")
- `environment` (string): One of "local", "ci", "production_sample" (default: "local")

## Response Format

Same as `/v1/evaluate` endpoint:

```json
{
  "run_id": "abc123...",
  "overall_status": "PASS",
  "metric_results": [
    {
      "metric_id": "agent.task_completion",
      "metric_name": "Task Completion",
      "score": 0.85,
      "threshold": 0.7,
      "passed": true,
      "reason": "...",
      "error": null
    }
  ],
  "evidence_pointer": "backend/artifacts/abc123.../evidence.json"
}
```

## Examples

### Example 1: Using Default Config with Filters

```bash
curl -X POST http://localhost:5008/v1/evaluate-from-config \
  -F "evaluation_id=eval-001" \
  -F "evaluation_name=Quality Test" \
  -F "goals=Quality & Accuracy" \
  -F "metrics=Task Completion, Answer Relevancy" \
  -F "deployment_stage=dev" \
  -F "risk_class=low" \
  -F "user_impact=internal"
```

### Example 2: Using Default Config (All Goals/Metrics)

```bash
curl -X POST http://localhost:5008/v1/evaluate-from-config \
  -F "evaluation_id=eval-002" \
  -F "evaluation_name=Full Evaluation"
```

### Example 3: Upload Custom Config File

```bash
curl -X POST http://localhost:5008/v1/evaluate-from-config \
  -F "evaluation_id=eval-003" \
  -F "evaluation_name=Custom Evaluation" \
  -F "goals=Quality & Accuracy" \
  -F "metrics=Task Completion" \
  -F "config_file=@/path/to/your/evaluation_config.json"
```

### Example 4: Using Python requests

```python
import requests

# Using default config
data = {
    "evaluation_id": "eval-001",
    "evaluation_name": "Test Evaluation",
    "goals": "Quality & Accuracy",
    "metrics": "Task Completion, Answer Relevancy",
    "deployment_stage": "dev",
    "risk_class": "low",
    "user_impact": "internal"
}

response = requests.post(
    "http://localhost:5008/v1/evaluate-from-config",
    data=data,
    timeout=300
)

result = response.json()
print(f"Run ID: {result['run_id']}")
print(f"Status: {result['overall_status']}")
```

### Example 5: Upload File with Python

```python
import requests

files = {
    "config_file": ("evaluation_config.json", open("evaluation_config.json", "rb"), "application/json")
}

data = {
    "evaluation_id": "eval-004",
    "evaluation_name": "Custom Config Test",
    "goals": "Quality & Accuracy",
    "metrics": "Task Completion"
}

response = requests.post(
    "http://localhost:5008/v1/evaluate-from-config",
    files=files,
    data=data,
    timeout=300
)
```

## Config File Format

The evaluation config file should follow this structure:

```json
{
  "goals": [
    {
      "goal": "Quality & Accuracy",
      "metrics": [
        {
          "metric": "Task Completion",
          "test_cases": [
            {
              "input": "Explain the difference between supervised and unsupervised learning.",
              "actual_output": "this is coming from the model interaction"
            }
          ]
        },
        {
          "metric": "Answer Relevancy",
          "test_cases": [
            {
              "input": "What are the main responsibilities of a product manager?",
              "actual_output": "this is coming from the model interaction"
            }
          ]
        }
      ]
    }
  ]
}
```

## Metric Name Mapping

The endpoint automatically maps metric names from the config file to metric IDs in the schema:

- "Task Completion" → `agent.task_completion`
- "Answer Relevancy" → `rag.answer_relevancy`
- "Faithfulness" → `rag.faithfulness`
- "Goal Accuracy" → `multiturn.goal_accuracy`
- etc.

See `config_parser.py` for the full mapping.

## Error Handling

- **404**: Default config file not found (if no file uploaded)
- **400**: Invalid JSON, no matching metrics, or no test cases found
- **500**: Internal server error during processing

## Notes

- The first evaluation may take 2-5 minutes (model loading)
- Subsequent evaluations are faster
- Test cases are extracted from the config based on the specified goals and metrics
- If goals/metrics are not specified, all test cases are included
- The endpoint uses the same evaluation engine as `/v1/evaluate`

