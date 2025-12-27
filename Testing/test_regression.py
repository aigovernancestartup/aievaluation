#!/usr/bin/env python3
"""
Regression Test Suite for AI Evaluation API
Based on evaluation_config.json structure

Tests:
1. One goal, multiple metrics (across all goals)
2. Multiple goals, multiple metrics (randomized combinations)
"""

import requests
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Configuration
API_URL = "http://localhost:5008/v1/aievaluation/evaluate-from-config"
CONFIG_PATH = Path(__file__).parent / "testingconfig" / "evaluation_config.json"
TIMEOUT = 300  # 5 minutes per request

# Test configuration
RANDOM_SEED = 42  # For reproducible randomization
NUM_RANDOM_COMBINATIONS = 5  # Number of random multi-goal tests


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)


def print_test_header(test_num: int, test_name: str):
    """Print test header."""
    print(f"\n{'='*80}")
    print(f"TEST {test_num}: {test_name}")
    print(f"{'='*80}")


def load_config() -> Dict[str, Any]:
    """Load evaluation config JSON."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_goals(config: Dict[str, Any]) -> List[str]:
    """Extract all goal names from config."""
    return [goal_obj.get("goal", "") for goal_obj in config.get("goals", [])]


def get_metrics_for_goal(config: Dict[str, Any], goal_name: str) -> List[str]:
    """Extract all metric names for a specific goal."""
    for goal_obj in config.get("goals", []):
        if goal_obj.get("goal", "").lower() == goal_name.lower():
            return [m.get("metric", "") for m in goal_obj.get("metrics", [])]
    return []


def get_all_metrics(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract all (goal, metric) pairs from config."""
    pairs = []
    for goal_obj in config.get("goals", []):
        goal_name = goal_obj.get("goal", "")
        for metric_obj in goal_obj.get("metrics", []):
            metric_name = metric_obj.get("metric", "")
            pairs.append((goal_name, metric_name))
    return pairs


def call_api(
    evaluation_id: str,
    evaluation_name: str,
    goals: str = None,
    metrics: str = None,
    **kwargs
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Call the evaluate-from-config API endpoint.
    
    Returns:
        (success: bool, response_data: dict, error_message: str)
    """
    data = {
        "evaluation_id": evaluation_id,
        "evaluation_name": evaluation_name,
        "deployment_stage": kwargs.get("deployment_stage", "dev"),
        "risk_class": kwargs.get("risk_class", "low"),
        "user_impact": kwargs.get("user_impact", "internal"),
        "mode": kwargs.get("mode", "one_off"),
        "environment": kwargs.get("environment", "local"),
    }
    
    if goals:
        data["goals"] = goals
    if metrics:
        data["metrics"] = metrics
    
    try:
        response = requests.post(API_URL, data=data, timeout=TIMEOUT)
        
        if response.status_code == 200:
            return True, response.json(), ""
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            return False, {}, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, {}, f"Connection error: Cannot connect to {API_URL}"
    except requests.exceptions.Timeout:
        return False, {}, f"Request timed out after {TIMEOUT} seconds"
    except Exception as e:
        return False, {}, f"Error: {type(e).__name__}: {str(e)}"


def print_result(success: bool, response_data: Dict[str, Any], error_msg: str):
    """Print test result in a formatted way."""
    if success:
        print(f"✅ Status: PASSED")
        print(f"   Run ID: {response_data.get('run_id', 'N/A')}")
        print(f"   Overall Status: {response_data.get('overall_status', 'N/A')}")
        
        metric_results = response_data.get('metric_results', [])
        print(f"   Metrics Evaluated: {len(metric_results)}")
        
        if metric_results:
            print(f"\n   Metric Results:")
            for i, metric in enumerate(metric_results[:5], 1):  # Show first 5
                status_icon = "✅" if metric.get('passed') else "❌" if metric.get('passed') is False else "⚠️"
                score = metric.get('score', 'N/A')
                print(f"   {i}. {status_icon} {metric.get('metric_name', 'N/A')}: {score}")
            
            if len(metric_results) > 5:
                print(f"   ... and {len(metric_results) - 5} more metrics")
        
        print(f"   Evidence: {response_data.get('evidence_pointer', 'N/A')}")
    else:
        print(f"❌ Status: FAILED")
        print(f"   Error: {error_msg}")


def test_one_goal_multiple_metrics(config: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """
    Test: One goal, multiple metrics (across all goals).
    For each goal, test with all its metrics.
    """
    print_section("TEST SUITE 1: One Goal, Multiple Metrics")
    
    results = []
    goals = get_all_goals(config)
    
    for idx, goal_name in enumerate(goals, 1):
        metrics = get_metrics_for_goal(config, goal_name)
        
        if not metrics:
            print(f"\n⚠️  Skipping goal '{goal_name}': No metrics found")
            continue
        
        print_test_header(
            idx,
            f"Goal: '{goal_name}' with {len(metrics)} metrics"
        )
        print(f"Goal: {goal_name}")
        print(f"Metrics: {', '.join(metrics)}")
        
        metrics_str = ", ".join(metrics)
        evaluation_id = f"regression-1goal-{idx:02d}"
        evaluation_name = f"Regression Test: {goal_name}"
        
        success, response_data, error_msg = call_api(
            evaluation_id=evaluation_id,
            evaluation_name=evaluation_name,
            goals=goal_name,
            metrics=metrics_str
        )
        
        print_result(success, response_data, error_msg)
        results.append((f"Goal: {goal_name}", success))
    
    return results


def test_multiple_goals_multiple_metrics(config: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """
    Test: Multiple goals, multiple metrics (randomized combinations).
    Generates random combinations of 2-4 goals with their metrics.
    """
    print_section("TEST SUITE 2: Multiple Goals, Multiple Metrics (Randomized)")
    
    results = []
    goals = get_all_goals(config)
    all_metrics = get_all_metrics(config)
    
    if len(goals) < 2:
        print("⚠️  Not enough goals for multi-goal testing (need at least 2)")
        return results
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    for test_num in range(1, NUM_RANDOM_COMBINATIONS + 1):
        # Randomly select 2-4 goals
        num_goals = random.randint(2, min(4, len(goals)))
        selected_goals = random.sample(goals, num_goals)
        
        # Collect all metrics from selected goals
        selected_metrics = []
        for goal_name in selected_goals:
            metrics = get_metrics_for_goal(config, goal_name)
            selected_metrics.extend(metrics)
        
        # Optionally, randomly sample some metrics (to avoid too many)
        if len(selected_metrics) > 10:
            selected_metrics = random.sample(selected_metrics, 10)
        
        print_test_header(
            test_num,
            f"Multiple Goals ({len(selected_goals)} goals, {len(selected_metrics)} metrics)"
        )
        print(f"Goals: {', '.join(selected_goals)}")
        print(f"Metrics: {', '.join(selected_metrics[:5])}...")  # Show first 5
        
        goals_str = ", ".join(selected_goals)
        metrics_str = ", ".join(selected_metrics)
        evaluation_id = f"regression-multigoal-{test_num:02d}"
        evaluation_name = f"Regression Test: {len(selected_goals)} Goals"
        
        success, response_data, error_msg = call_api(
            evaluation_id=evaluation_id,
            evaluation_name=evaluation_name,
            goals=goals_str,
            metrics=metrics_str
        )
        
        print_result(success, response_data, error_msg)
        results.append((f"Multi-Goal Test {test_num}", success))
    
    return results


def test_all_goals_all_metrics(config: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Test: All goals, all metrics (comprehensive test).
    """
    print_section("TEST SUITE 3: All Goals, All Metrics (Comprehensive)")
    
    print_test_header(1, "All Goals with All Metrics")
    
    goals = get_all_goals(config)
    all_metrics = get_all_metrics(config)
    metric_names = [metric for _, metric in all_metrics]
    
    print(f"Goals: {len(goals)} goals")
    print(f"Metrics: {len(metric_names)} metrics")
    print(f"Goals: {', '.join(goals)}")
    print(f"Metrics: {', '.join(metric_names[:10])}...")  # Show first 10
    
    goals_str = ", ".join(goals)
    metrics_str = ", ".join(metric_names)
    evaluation_id = "regression-comprehensive"
    evaluation_name = "Regression Test: Comprehensive (All Goals & Metrics)"
    
    success, response_data, error_msg = call_api(
        evaluation_id=evaluation_id,
        evaluation_name=evaluation_name,
        goals=goals_str,
        metrics=metrics_str
    )
    
    print_result(success, response_data, error_msg)
    return ("Comprehensive Test", success)


def check_backend_health() -> bool:
    """Check if backend is accessible."""
    try:
        response = requests.get("http://localhost:5008/docs", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Run all regression tests."""
    print_section("REGRESSION TEST SUITE", "=")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config file: {CONFIG_PATH}")
    print(f"API URL: {API_URL}")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Check backend health
    print_section("Backend Health Check")
    if not check_backend_health():
        print("❌ Backend is not accessible")
        print("\n   To start the backend:")
        print("   cd backend && uvicorn app:app --reload --port 5008")
        sys.exit(1)
    print("✅ Backend is running and accessible")
    
    # Load config
    try:
        config = load_config()
        print(f"\n✅ Loaded config: {len(get_all_goals(config))} goals")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    all_results = []
    
    # Test Suite 1: One goal, multiple metrics
    print("\n" + "="*80)
    print("="*80)
    results_1 = test_one_goal_multiple_metrics(config)
    all_results.extend(results_1)
    
    # Test Suite 2: Multiple goals, multiple metrics (randomized)
    print("\n" + "="*80)
    print("="*80)
    results_2 = test_multiple_goals_multiple_metrics(config)
    all_results.extend(results_2)
    
    # Test Suite 3: All goals, all metrics
    print("\n" + "="*80)
    print("="*80)
    result_3 = test_all_goals_all_metrics(config)
    all_results.append(result_3)
    
    # Summary
    print_section("TEST SUMMARY", "=")
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    print(f"\n{'='*80}")
    print("Detailed Results:")
    print(f"{'='*80}")
    for test_name, result in all_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    if passed == total:
        print("\n🎉 All regression tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

