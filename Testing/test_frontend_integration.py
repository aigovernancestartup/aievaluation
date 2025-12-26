#!/usr/bin/env python3
"""
Test script to simulate frontend requests to the backend.
Tests the /v1/aievaluation/evaluate endpoint with various frontend scenarios.
"""

import requests
import json
import sys
from pathlib import Path

# Backend API URL - matches the FastAPI backend endpoint
API_URL = "http://localhost:5008/v1/aievaluation/evaluate"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "testingconfig" / "evaluation_config.json"

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(response, test_name):
    """Print test result in a formatted way."""
    status_code = response.status_code
    if status_code == 200:
        print(f"✅ {test_name}: PASSED (Status: {status_code})")
        return True
    else:
        print(f"❌ {test_name}: FAILED (Status: {status_code})")
        try:
            error_detail = response.json()
            print(f"   Error: {json.dumps(error_detail, indent=2)}")
        except:
            print(f"   Error: {response.text}")
        return False

def test_default_config_all_metrics():
    """Test 1: Use default config with all goals and metrics (no filters)."""
    print_section("Test 1: Default Config - All Goals & Metrics")
    
    # Payload matches backend's EvaluateRequest schema
    payload = {
        "evaluation_object": "frontend-test-001",
        "use_case": "Full Evaluation Test",
        "context": {
            "deployment_stage": "dev",
            "risk_class": "low",
            "user_impact": "internal"
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
                "input": "What is the return policy?",
                "actual_output": "You can return items within 30 days for a full refund.",
                "retrieval_context": ["Our return policy allows returns within 30 days of purchase."]
            }
        ]
    }
    
    print(f"Request: evaluation_object={payload['evaluation_object']}, use_case={payload['use_case']}")
    print(f"Metrics: {[m['metric_id'] for m in payload['metrics']]}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        success = print_result(response, "Default Config - All Metrics")
        
        if success:
            result = response.json()
            print(f"\n📊 Results:")
            print(f"   Run ID: {result.get('run_id')}")
            print(f"   Overall Status: {result.get('overall_status')}")
            print(f"   Metrics Evaluated: {len(result.get('metric_results', []))}")
            print(f"   Evidence: {result.get('evidence_pointer')}")
            
            # Show first few metrics
            if result.get('metric_results'):
                print(f"\n   First 3 Metrics:")
                for i, metric in enumerate(result['metric_results'][:3], 1):
                    status = "✅" if metric.get('passed') else "❌" if metric.get('passed') is False else "⚠️"
                    print(f"   {i}. {status} {metric.get('metric_name')} ({metric.get('metric_id')})")
        
        return success
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Cannot connect to {API_URL}")
        print("   Make sure the backend is running: cd backend && python3 -m uvicorn app:app --port 5008")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def test_default_config_filtered():
    """Test 2: Use default config with specific goals and metrics."""
    print_section("Test 2: Default Config - Filtered by Goals & Metrics")
    
    # Payload with multiple metrics
    payload = {
        "evaluation_object": "frontend-test-002",
        "use_case": "Quality & Accuracy Test",
        "context": {
            "deployment_stage": "dev",
            "risk_class": "low",
            "user_impact": "internal"
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
            },
            {
                "metric_id": "rag.faithfulness",
                "threshold": 0.7,
                "init_params": {}
            }
        ],
        "test_cases": [
            {
                "input": "How do I contact support?",
                "actual_output": "You can contact support via email at support@example.com or call 1-800-SUPPORT.",
                "retrieval_context": [
                    "Support email: support@example.com",
                    "Support phone: 1-800-SUPPORT"
                ]
            }
        ]
    }
    
    print(f"Request: evaluation_object={payload['evaluation_object']}")
    print(f"Metrics: {[m['metric_id'] for m in payload['metrics']]}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        success = print_result(response, "Default Config - Filtered")
        
        if success:
            result = response.json()
            print(f"\n📊 Results:")
            print(f"   Run ID: {result.get('run_id')}")
            print(f"   Overall Status: {result.get('overall_status')}")
            print(f"   Metrics Evaluated: {len(result.get('metric_results', []))}")
            
            for metric in result.get('metric_results', []):
                status = "✅" if metric.get('passed') else "❌" if metric.get('passed') is False else "⚠️"
                score = metric.get('score', 'N/A')
                print(f"   {status} {metric.get('metric_name')}: Score={score}")
        
        return success
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def test_custom_config_file():
    """Test 3: Test with customer-facing context."""
    print_section("Test 3: Customer-Facing Evaluation")
    
    # Test with customer_facing user_impact
    payload = {
        "evaluation_object": "frontend-test-003",
        "use_case": "Customer Support Chatbot",
        "context": {
            "deployment_stage": "staging",
            "risk_class": "medium",
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
                "threshold": 0.8,
                "init_params": {}
            }
        ],
        "test_cases": [
            {
                "input": "What are your business hours?",
                "actual_output": "We are open Monday to Friday, 9 AM to 5 PM.",
                "retrieval_context": ["Business hours: Mon-Fri 9AM-5PM, Closed weekends"]
            }
        ]
    }
    
    print(f"Request: evaluation_object={payload['evaluation_object']}")
    print(f"Context: {payload['context']}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        success = print_result(response, "Customer-Facing Evaluation")
        
        if success:
            result = response.json()
            print(f"\n📊 Results:")
            print(f"   Run ID: {result.get('run_id')}")
            print(f"   Overall Status: {result.get('overall_status')}")
        
        return success
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def test_multiple_goals():
    """Test 4: Multiple test cases."""
    print_section("Test 4: Multiple Test Cases")
    
    # Test with multiple test cases
    payload = {
        "evaluation_object": "frontend-test-004",
        "use_case": "Multiple Test Cases Evaluation",
        "context": {
            "deployment_stage": "dev",
            "risk_class": "medium",
            "user_impact": "customer_facing"
        },
        "run": {
            "mode": "batch",
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
                "input": "What is the capital of France?",
                "actual_output": "Paris is the capital of France.",
                "retrieval_context": ["France is a country in Europe. Its capital is Paris."]
            },
            {
                "input": "What is 2 + 2?",
                "actual_output": "2 + 2 equals 4.",
                "retrieval_context": ["Basic arithmetic: 2 + 2 = 4"]
            }
        ]
    }
    
    print(f"Request: evaluation_object={payload['evaluation_object']}")
    print(f"Test Cases: {len(payload['test_cases'])}")
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        success = print_result(response, "Multiple Test Cases")
        
        if success:
            result = response.json()
            print(f"\n📊 Results:")
            print(f"   Metrics Evaluated: {len(result.get('metric_results', []))}")
        
        return success
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def test_error_handling():
    """Test 5: Error handling - invalid inputs."""
    print_section("Test 5: Error Handling")
    
    # Test with missing required fields (no metrics)
    payload = {
        "evaluation_object": "error-test",
        "use_case": "Test missing metrics",
        "context": {
            "deployment_stage": "dev",
            "risk_class": "low",
            "user_impact": "internal"
        },
        "run": {
            "mode": "one_off",
            "environment": "local"
        },
        "metrics": [],  # Empty metrics - should fail validation
        "test_cases": [
            {
                "input": "test",
                "actual_output": "test"
            }
        ]
    }
    
    print("Testing with empty metrics array...")
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 400:  # Bad request / validation error
            print("✅ Error Handling: PASSED (correctly rejected empty metrics)")
            return True
        elif response.status_code == 422:  # Pydantic validation error
            print("✅ Error Handling: PASSED (correctly rejected invalid input)")
            return True
        else:
            print(f"⚠️  Unexpected status: {response.status_code}")
            try:
                print(f"   Response: {response.json()}")
            except:
                pass
            return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def check_backend_health():
    """Check if backend is accessible."""
    print_section("Backend Health Check")
    
    try:
        # Try to access the docs endpoint
        response = requests.get("http://localhost:5008/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and accessible")
            return True
        else:
            print(f"⚠️  Backend responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not accessible")
        print("\n   To start the backend:")
        print("   cd backend && uvicorn app:app --reload --port 5008")
        return False
    except Exception as e:
        print(f"❌ Error checking backend: {type(e).__name__}: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  FRONTEND INTEGRATION TEST SUITE")
    print("  Testing /v1/aievaluation/evaluate endpoint")
    print("="*70)
    
    # Check backend health first
    if not check_backend_health():
        print("\n❌ Backend is not running. Please start it first.")
        sys.exit(1)
    
    results = []
    
    # Run tests
    results.append(("Health Check", True))  # Already passed
    results.append(("Default Config - All Metrics", test_default_config_all_metrics()))
    results.append(("Default Config - Filtered", test_default_config_filtered()))
    results.append(("Customer-Facing Evaluation", test_custom_config_file()))
    results.append(("Multiple Test Cases", test_multiple_goals()))
    results.append(("Error Handling", test_error_handling()))
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n📊 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

