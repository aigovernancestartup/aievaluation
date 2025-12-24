# Where to See the Output

## 1. **API Response (Immediate)**

When you call the API, you get a JSON response with:
- `run_id` - Unique identifier for this evaluation run
- `overall_status` - "PASS", "FAIL", or "WARNING"
- `metric_results` - Array of results for each metric
- `evidence_pointer` - Path to the detailed evidence file

### View in Test Script:
```bash
python test_api_simple.py
```

### View in Browser:
Go to `http://localhost:5008/docs` and use the interactive API tester

### View with curl:
```bash
curl -X POST http://localhost:5008/v1/evaluate \
  -H "Content-Type: application/json" \
  -d @your_payload.json | jq
```

---

## 2. **Evidence JSON File (Detailed)**

Every evaluation creates a detailed evidence file at:
```
backend/artifacts/<run_id>/evidence.json
```

### View the file:
```bash
# List all runs
ls -la backend/artifacts/

# View a specific run (replace <run_id> with actual ID)
cat backend/artifacts/<run_id>/evidence.json

# Pretty print with jq
cat backend/artifacts/<run_id>/evidence.json | jq

# View in your editor
code backend/artifacts/<run_id>/evidence.json  # VS Code
# or
open backend/artifacts/<run_id>/evidence.json  # macOS default app
```

### What's in the evidence file:
- Full request details (test cases, metrics, context)
- Per-metric scores and reasons
- All test case inputs/outputs
- Timestamps
- Any gaps or errors encountered

---

## 3. **Quick View Commands**

### See all evaluation runs:
```bash
ls backend/artifacts/
```

### View latest run:
```bash
# Get the most recent run
LATEST=$(ls -t backend/artifacts/ | head -1)
cat backend/artifacts/$LATEST/evidence.json | jq
```

### Search for specific metric results:
```bash
cat backend/artifacts/<run_id>/evidence.json | jq '.metric_evidence'
```

### View only scores:
```bash
cat backend/artifacts/<run_id>/evidence.json | jq '.metric_evidence[].scores'
```

---

## 4. **Example Output Structure**

### API Response:
```json
{
  "run_id": "abc123...",
  "overall_status": "PASS",
  "metric_results": [
    {
      "metric_id": "rag.answer_relevancy",
      "metric_name": "Answer Relevancy",
      "score": 0.85,
      "threshold": 0.7,
      "passed": true,
      "reason": "The answer is highly relevant..."
    }
  ],
  "evidence_pointer": "/path/to/artifacts/abc123/evidence.json"
}
```

### Evidence File:
```json
{
  "run_id": "abc123...",
  "started_at": 1234567890.123,
  "evaluation_object": "test",
  "use_case": "test",
  "context": {...},
  "metrics": [...],
  "test_cases": [...],
  "metric_evidence": [
    {
      "metric_id": "rag.answer_relevancy",
      "metric_class": "AnswerRelevancyMetric",
      "scores": [0.85],
      "reasons": ["The answer is highly relevant..."]
    }
  ],
  "gaps": []
}
```

---

## 5. **Tips**

- **Run ID** is in the API response - use it to find the evidence file
- **Evidence files** are permanent - they don't get deleted automatically
- **Use jq** for pretty JSON viewing: `brew install jq` (macOS) or `apt install jq` (Linux)
- **Browser JSON viewer**: Open the file in Chrome/Firefox for formatted viewing


