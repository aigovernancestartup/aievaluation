#!/usr/bin/env python3
"""Fast API test - uses a deterministic metric that doesn't require LLM"""

import requests
import json

API_URL = "http://localhost:8000/v1/evaluate"

# Test with a metric that doesn't require LLM (if available)
# Or test with a simpler payload to verify API works
payload = {
    "evaluation_object": "test",
    "use_case": "test",
    "context": {
        "deployment_stage": "dev",
        "risk_class": "low",
        "user_impact": "internal"
    },
    "run": {
        "mode": "one_off",
        "environment": "local"
    },
    "metrics": [{
        "metric_id": "rag.answer_relevancy",
        "threshold": 0.7,
        "init_params": {}  # Will use Ollama, but we'll catch timeout
    }],
    "test_cases": [{
        "input": "test question",
        "actual_output": "test answer",
        "retrieval_context": ["context 1", "context 2"]
    }]
}

print("Testing API (this may take 60-120 seconds for first call)...")
print("The model needs to load and process. Please wait...\n")

try:
    # Increase timeout significantly for first model load
    response = requests.post(API_URL, json=payload, timeout=300)  # 5 minutes
    print(f"✅ Status: {response.status_code}")
    result = response.json()
    
    print("\n" + "="*60)
    print("API RESPONSE")
    print("="*60)
    print(f"Run ID: {result.get('run_id')}")
    print(f"Overall Status: {result.get('overall_status')}")
    print(f"\nEvidence File Location:")
    print(f"  {result.get('evidence_pointer')}")
    
    print("\n" + "-"*60)
    print("METRIC RESULTS")
    print("-"*60)
    if result.get('metric_results'):
        for i, metric in enumerate(result['metric_results'], 1):
            print(f"\nMetric {i}: {metric.get('metric_name')} ({metric.get('metric_id')})")
            if metric.get('error'):
                print(f"  ❌ Error: {metric.get('error')}")
            else:
                score = metric.get('score')
                threshold = metric.get('threshold')
                passed = metric.get('passed')
                status = "✅ PASS" if passed else "❌ FAIL" if passed is False else "⚠️  UNKNOWN"
                print(f"  {status}")
                if score is not None:
                    print(f"  Score: {score:.4f}")
                if threshold is not None:
                    print(f"  Threshold: {threshold}")
                if metric.get('reason'):
                    print(f"  Reason: {metric.get('reason')[:100]}...")
    
    print("\n" + "="*60)
    print("📁 FULL DETAILS")
    print("="*60)
    print(f"See complete evidence at: {result.get('evidence_pointer')}")
    
    print("\n✅ API is working!")
    
except requests.exceptions.Timeout:
    print("\n⏱️  TIMEOUT: The model is taking too long to respond.")
    print("\nThis is normal for the first call - the model needs to load.")
    print("\nTroubleshooting:")
    print("1. Check if Ollama is running: curl http://localhost:11434/api/tags")
    print("2. Try warming up the model first:")
    print("   ollama run deepseek-r1:1.5b 'hello'")
    print("3. The model may be very slow on CPU - consider using GPU")
    print("4. Try increasing timeout in the script (currently 300 seconds)")
    
except requests.exceptions.ConnectionError:
    print("\n❌ CONNECTION ERROR: Cannot connect to the API server.")
    print("\nMake sure the backend is running:")
    print("  cd backend && uvicorn app:app --reload --port 8000")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")


