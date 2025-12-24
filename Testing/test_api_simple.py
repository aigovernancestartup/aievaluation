#!/usr/bin/env python3
"""Bare-bones API test - just checks if it works"""

import requests
import json

# this is the API URL for the backend
API_URL = "http://localhost:5008/v1/evaluate"

# Minimal test payload
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
        "init_params": {}  # Ollama model will be auto-configured
    }],
    "test_cases": [{
        "input": "test question",
        "actual_output": "test answer",
        "retrieval_context": ["context 1", "context 2"]
    }]
}

print("Testing API...")
print("⚠️  Note: First call may take 2-5 minutes (model needs to load)")
print("   Subsequent calls will be faster.\n")

try:
    response = requests.post(API_URL, json=payload, timeout=300)  # 5 minutes for first load
    print(f"Status: {response.status_code}")
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
    print(f"\nTo view the full JSON file:")
    print(f"  cat {result.get('evidence_pointer')}")
    print(f"\nOr open it in your editor/JSON viewer")
    
    print("\n✅ API is working!")
except Exception as e:
    print(f"❌ Error: {e}")

