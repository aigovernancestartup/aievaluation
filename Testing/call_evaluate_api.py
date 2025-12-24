#!/usr/bin/env python3
"""
Script to call the /v1/evaluate endpoint of the AI Evaluation Engine.

Usage:
    python call_evaluate_api.py

Make sure the server is running:
    cd backend && python3 -m uvicorn app:app --reload --port 5008
"""

import requests
import json

API_URL = "http://localhost:5008/v1/evaluate"


def call_evaluate():
    """Send a sample evaluation request to the API."""
    
    payload = {
        "evaluation_object": "customer-support-chatbot",
        "use_case": "Answer customer questions about product returns",
        "context": {
            "deployment_stage": "dev",
            "risk_class": "low",
            "user_impact": "customer_facing",
            "domain": "general"
        },
        "run": {
            "mode": "one_off",
            "environment": "local",
            "baseline_run_id": None,
            "budget": {
                "max_tokens": 1000,
                "max_cost_usd": 0.10
            }
        },
        "metrics": [
            {
                "metric_id": "answer_relevancy",
                "threshold": 0.7,
                "init_params": {}
            }
        ],
        "test_cases": [
            {
                "input": "How do I return a product?",
                "actual_output": "To return a product, please visit our returns portal at example.com/returns, log in with your order number, and follow the instructions to print a shipping label.",
                "expected_output": None,
                "context": None,
                "retrieval_context": [
                    "Our return policy allows returns within 30 days of purchase.",
                    "Customers can initiate returns through the returns portal at example.com/returns."
                ]
            }
        ]
    }

    print("=" * 60)
    print("Calling POST /v1/evaluate")
    print("=" * 60)
    print("\nRequest payload:")
    print(json.dumps(payload, indent=2))
    print("\n" + "-" * 60)

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("\nResponse (Status: {})".format(response.status_code))
        print(json.dumps(result, indent=2))
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Run ID: {result.get('run_id')}")
        print(f"Overall Status: {result.get('overall_status')}")
        print(f"Evidence: {result.get('evidence_pointer')}")
        
        print("\nMetric Results:")
        for metric in result.get("metric_results", []):
            status = "✅ PASS" if metric.get("passed") else "❌ FAIL"
            if metric.get("error"):
                status = "⚠️ ERROR"
            print(f"  - {metric.get('metric_name')}: {status}")
            if metric.get("score") is not None:
                print(f"    Score: {metric.get('score'):.2f} (threshold: {metric.get('threshold')})")
            if metric.get("reason"):
                print(f"    Reason: {metric.get('reason')}")
            if metric.get("error"):
                print(f"    Error: {metric.get('error')}")
                
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server.")
        print("Make sure the API is running:")
        print("  cd backend && python3 -m uvicorn app:app --reload --port 5008")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    call_evaluate()


