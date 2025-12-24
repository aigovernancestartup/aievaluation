#!/usr/bin/env python3
"""Test script for the new /v1/evaluate-from-config endpoint"""

import requests
import json

API_URL = "http://localhost:5008/v1/evaluate-from-config"

# Test 1: Use default config with specific goals and metrics
print("="*60)
print("Test 1: Using default config with specific goals/metrics")
print("="*60)

data = {
    "evaluation_id": "test-eval-001",
    "evaluation_name": "Test Evaluation",
    "goals": "Quality & Accuracy",
    "metrics": "Task Completion, Answer Relevancy",
    "deployment_stage": "dev",
    "risk_class": "low",
    "user_impact": "internal",
    "mode": "one_off",
    "environment": "local"
}

try:
    response = requests.post(API_URL, data=data, timeout=300)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Success!")
        print(f"Run ID: {result.get('run_id')}")
        print(f"Overall Status: {result.get('overall_status')}")
        print(f"Evidence: {result.get('evidence_pointer')}")
        print(f"\nMetric Results:")
        for metric in result.get('metric_results', []):
            print(f"  - {metric.get('metric_name')} ({metric.get('metric_id')}): {metric.get('score')}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("Test 2: Using default config with all goals/metrics")
print("="*60)

data2 = {
    "evaluation_id": "test-eval-002",
    "evaluation_name": "Full Evaluation",
    "deployment_stage": "dev",
    "risk_class": "low",
    "user_impact": "internal",
    "mode": "one_off",
    "environment": "local"
}

try:
    response = requests.post(API_URL, data=data2, timeout=300)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Success!")
        print(f"Run ID: {result.get('run_id')}")
        print(f"Overall Status: {result.get('overall_status')}")
        print(f"Number of metrics: {len(result.get('metric_results', []))}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("Test 3: Upload custom config file")
print("="*60)

# This would require a file upload - showing the structure
print("To test file upload, use:")
print("  curl -X POST http://localhost:5008/v1/evaluate-from-config \\")
print("    -F 'evaluation_id=test-003' \\")
print("    -F 'evaluation_name=Custom Config Test' \\")
print("    -F 'goals=Quality & Accuracy' \\")
print("    -F 'metrics=Task Completion' \\")
print("    -F 'config_file=@/path/to/your/config.json'")

