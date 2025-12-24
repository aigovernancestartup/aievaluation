#!/usr/bin/env python3
"""
Flask-based AI Evaluation API

This API provides an endpoint to trigger AI model evaluations with configurable
parameters including model selection, evaluation goals, metrics, and test configurations.

Usage:
    python flask_api.py

API Endpoint:
    POST /aievaluation
    
Parameters (JSON body):
    - model: Name of the model to evaluate (e.g., "deepseek-r1:1.5b", "gpt-4")
    - evaluation_goals: List of evaluation goals or a string describing goals
    - metrics: Metrics to run ("all" or list of specific metrics)
    - testingconfig_file: Path to the testing configuration file

Example:
    curl -X POST http://localhost:5009/aievaluation \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek-r1:1.5b",
            "evaluation_goals": ["accuracy", "relevancy", "safety"],
            "metrics": "all",
            "testingconfig_file": "config/test_config.yaml"
        }'
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import os
import sys
import uuid
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from functools import wraps

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
LOG_DIR = BASE_DIR / "logs"

# Create directories
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file path
LOG_FILE = LOG_DIR / "flask_api.log"

# =============================================================================
# Logging Configuration
# =============================================================================
def setup_logging():
    """Configure logging with both file and console handlers."""
    
    # Create logger
    logger = logging.getLogger("aievaluation")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Log format with detailed information
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Rotating file handler (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Initialize logger
logger = setup_logging()

# =============================================================================
# Flask App
# =============================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests


# =============================================================================
# Request Logging Decorator
# =============================================================================
def log_request(func):
    """Decorator to log incoming requests and responses."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_id = uuid.uuid4().hex[:8]
        
        # Log request details
        logger.info(f"[{request_id}] === INCOMING REQUEST ===")
        logger.info(f"[{request_id}] Endpoint: {request.method} {request.path}")
        logger.info(f"[{request_id}] Remote IP: {request.remote_addr}")
        logger.info(f"[{request_id}] User-Agent: {request.headers.get('User-Agent', 'N/A')}")
        
        if request.is_json:
            try:
                body = request.get_json()
                logger.info(f"[{request_id}] Request Body: {json.dumps(body, indent=2)}")
            except Exception as e:
                logger.warning(f"[{request_id}] Could not parse JSON body: {e}")
        
        # Execute the function
        try:
            response = func(*args, **kwargs)
            
            # Log response
            if isinstance(response, tuple):
                resp_data, status_code = response[0], response[1]
            else:
                resp_data, status_code = response, 200
            
            logger.info(f"[{request_id}] Response Status: {status_code}")
            logger.debug(f"[{request_id}] Response Data: {resp_data.get_json() if hasattr(resp_data, 'get_json') else resp_data}")
            logger.info(f"[{request_id}] === REQUEST COMPLETE ===")
            
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Exception: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
    
    return wrapper


# =============================================================================
# Endpoints
# =============================================================================
@app.route("/", methods=["GET"])
@log_request
def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return jsonify({
        "status": "healthy",
        "service": "AI Evaluation API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "log_file": str(LOG_FILE)
    })


@app.route("/aievaluation", methods=["POST"])
@log_request
def run_evaluation():
    """
    Run AI model evaluation with specified parameters.
    
    Request Body (JSON):
        {
            "model": "string - model name (required)",
            "evaluation_goals": "string or list - evaluation goals (required)",
            "metrics": "string or list - metrics to evaluate (default: 'all')",
            "testingconfig_file": "string - path to config file (optional)"
        }
    
    Returns:
        JSON response with evaluation run details
    """
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({
                "error": "No JSON data provided",
                "status": "failed"
            }), 400
        
        # Extract parameters with detailed logging
        logger.info("=" * 50)
        logger.info("PARAMETER EXTRACTION")
        logger.info("=" * 50)
        
        model = data.get("model")
        evaluation_goals = data.get("evaluation_goals")
        metrics = data.get("metrics", "all")
        testingconfig_file = data.get("testingconfig_file")
        
        # Log each parameter individually
        logger.info(f"  model: {model}")
        logger.info(f"  model type: {type(model).__name__}")
        logger.info(f"  evaluation_goals: {evaluation_goals}")
        logger.info(f"  evaluation_goals type: {type(evaluation_goals).__name__}")
        logger.info(f"  metrics: {metrics}")
        logger.info(f"  metrics type: {type(metrics).__name__}")
        logger.info(f"  testingconfig_file: {testingconfig_file}")
        logger.info(f"  testingconfig_file type: {type(testingconfig_file).__name__ if testingconfig_file else 'None'}")
        
        # Validation
        errors = []
        if not model:
            errors.append("'model' is required")
            logger.error("Validation failed: 'model' is required")
        if not evaluation_goals:
            errors.append("'evaluation_goals' is required")
            logger.error("Validation failed: 'evaluation_goals' is required")
            
        if errors:
            logger.warning(f"Request validation failed with {len(errors)} error(s)")
            return jsonify({
                "error": "Validation failed",
                "details": errors,
                "status": "failed"
            }), 400
        
        # Generate run ID
        run_id = uuid.uuid4().hex
        started_at = datetime.utcnow().isoformat()
        
        logger.info(f"Generated run_id: {run_id}")
        logger.info(f"Started at: {started_at}")
        
        # Build command arguments
        cmd_args = build_command_args(
            model=model,
            evaluation_goals=evaluation_goals,
            metrics=metrics,
            testingconfig_file=testingconfig_file
        )
        
        logger.info("=" * 50)
        logger.info("COMMAND ARGUMENTS BUILT")
        logger.info("=" * 50)
        for i, arg in enumerate(cmd_args):
            logger.info(f"  arg[{i}]: {arg}")
        logger.info(f"  Full command: python evaluate.py {' '.join(cmd_args)}")
        
        # Create evaluation log entry
        evaluation_log = {
            "run_id": run_id,
            "started_at": started_at,
            "client_ip": request.remote_addr,
            "user_agent": request.headers.get("User-Agent", "N/A"),
            "parameters": {
                "model": model,
                "model_type": type(model).__name__,
                "evaluation_goals": evaluation_goals,
                "evaluation_goals_type": type(evaluation_goals).__name__,
                "metrics": metrics,
                "metrics_type": type(metrics).__name__,
                "testingconfig_file": testingconfig_file,
                "testingconfig_file_type": type(testingconfig_file).__name__ if testingconfig_file else "None"
            },
            "command_args": cmd_args,
            "command_string": f"python evaluate.py {' '.join(cmd_args)}",
            "status": "initiated"
        }
        
        # Save evaluation request to artifacts
        run_dir = ARTIFACT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "evaluation_request.json"
        
        with open(log_path, "w") as f:
            json.dump(evaluation_log, f, indent=2)
        
        logger.info(f"Evaluation request saved to: {log_path}")
        
        # Execute the evaluation
        logger.info("=" * 50)
        logger.info("EXECUTING EVALUATION")
        logger.info("=" * 50)
        
        result = execute_evaluation(
            run_id=run_id,
            model=model,
            evaluation_goals=evaluation_goals,
            metrics=metrics,
            testingconfig_file=testingconfig_file,
            cmd_args=cmd_args
        )
        
        logger.info(f"Evaluation completed with status: {result.get('status', 'unknown')}")
        
        response_data = {
            "run_id": run_id,
            "status": result.get("status", "completed"),
            "message": f"Evaluation initiated for model '{model}'",
            "parameters": {
                "model": model,
                "evaluation_goals": evaluation_goals,
                "metrics": metrics,
                "testingconfig_file": testingconfig_file
            },
            "command_args": cmd_args,
            "result": result,
            "artifacts_path": str(run_dir),
            "log_file": str(LOG_FILE)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Exception in run_evaluation: {type(e).__name__}: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


@app.route("/aievaluation/<run_id>", methods=["GET"])
@log_request
def get_evaluation_status(run_id):
    """Get the status of a specific evaluation run."""
    logger.info(f"Status check for run_id: {run_id}")
    
    run_dir = ARTIFACT_DIR / run_id
    
    if not run_dir.exists():
        logger.warning(f"Run not found: {run_id}")
        return jsonify({
            "error": f"Evaluation run '{run_id}' not found",
            "status": "not_found"
        }), 404
    
    # Read evaluation request
    request_file = run_dir / "evaluation_request.json"
    result_file = run_dir / "evaluation_result.json"
    
    response = {"run_id": run_id}
    
    if request_file.exists():
        with open(request_file) as f:
            response["request"] = json.load(f)
            logger.debug(f"Loaded request data for {run_id}")
    
    if result_file.exists():
        with open(result_file) as f:
            response["result"] = json.load(f)
            response["status"] = "completed"
            logger.info(f"Run {run_id} status: completed")
    else:
        response["status"] = "in_progress"
        logger.info(f"Run {run_id} status: in_progress")
    
    return jsonify(response)


@app.route("/aievaluation/list", methods=["GET"])
@log_request
def list_evaluations():
    """List all evaluation runs."""
    logger.info("Listing all evaluation runs")
    runs = []
    
    if ARTIFACT_DIR.exists():
        for run_dir in ARTIFACT_DIR.iterdir():
            if run_dir.is_dir():
                request_file = run_dir / "evaluation_request.json"
                if request_file.exists():
                    with open(request_file) as f:
                        run_data = json.load(f)
                        runs.append({
                            "run_id": run_data.get("run_id"),
                            "started_at": run_data.get("started_at"),
                            "model": run_data.get("parameters", {}).get("model"),
                            "status": run_data.get("status")
                        })
    
    # Sort by started_at descending
    runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    logger.info(f"Found {len(runs)} evaluation runs")
    
    return jsonify({
        "total": len(runs),
        "runs": runs
    })


@app.route("/logs", methods=["GET"])
@log_request
def get_logs():
    """Get recent log entries."""
    lines = request.args.get("lines", 100, type=int)
    
    logger.info(f"Retrieving last {lines} log lines")
    
    if not LOG_FILE.exists():
        return jsonify({"error": "Log file not found", "log_file": str(LOG_FILE)}), 404
    
    try:
        with open(LOG_FILE, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return jsonify({
            "log_file": str(LOG_FILE),
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
            "logs": [line.strip() for line in recent_lines]
        })
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Helper Functions
# =============================================================================
def build_command_args(model, evaluation_goals, metrics, testingconfig_file):
    """
    Build command-line arguments for the evaluation script.
    
    Returns:
        list: Command-line arguments
    """
    logger.debug("Building command arguments...")
    args = []
    
    # Add model parameter
    args.extend(["--model", str(model)])
    logger.debug(f"  Added --model: {model}")
    
    # Add evaluation goals
    if isinstance(evaluation_goals, list):
        goals_str = ",".join(str(g) for g in evaluation_goals)
    else:
        goals_str = str(evaluation_goals)
    args.extend(["--evaluation-goals", goals_str])
    logger.debug(f"  Added --evaluation-goals: {goals_str}")
    
    # Add metrics
    if isinstance(metrics, list):
        metrics_str = ",".join(str(m) for m in metrics)
    else:
        metrics_str = str(metrics)
    args.extend(["--metrics", metrics_str])
    logger.debug(f"  Added --metrics: {metrics_str}")
    
    # Add testing config file if provided
    if testingconfig_file:
        args.extend(["--testingconfig-file", str(testingconfig_file)])
        logger.debug(f"  Added --testingconfig-file: {testingconfig_file}")
    
    logger.debug(f"Final args list: {args}")
    return args


def execute_evaluation(run_id, model, evaluation_goals, metrics, testingconfig_file, cmd_args):
    """
    Execute the evaluation process.
    
    This function can be extended to:
    - Run evaluation synchronously
    - Queue evaluation for async processing
    - Call external evaluation service
    
    Returns:
        dict: Evaluation result
    """
    logger.info(f"[{run_id}] Starting evaluation execution")
    logger.info(f"[{run_id}] Model: {model}")
    logger.info(f"[{run_id}] Evaluation Goals: {evaluation_goals}")
    logger.info(f"[{run_id}] Metrics: {metrics}")
    logger.info(f"[{run_id}] Config File: {testingconfig_file}")
    
    result = {
        "run_id": run_id,
        "status": "completed",
        "executed_at": datetime.utcnow().isoformat(),
        "model": model,
        "evaluation_goals": evaluation_goals,
        "metrics": metrics,
        "testingconfig_file": testingconfig_file,
        "command_preview": f"python evaluate.py {' '.join(cmd_args)}",
        "message": "Evaluation parameters validated and logged. Ready for execution."
    }
    
    # Option 1: Execute as subprocess (uncomment to enable)
    # try:
    #     logger.info(f"[{run_id}] Executing subprocess...")
    #     process = subprocess.run(
    #         ["python", "evaluate.py"] + cmd_args,
    #         capture_output=True,
    #         text=True,
    #         timeout=300  # 5 minute timeout
    #     )
    #     result["stdout"] = process.stdout
    #     result["stderr"] = process.stderr
    #     result["returncode"] = process.returncode
    #     result["status"] = "completed" if process.returncode == 0 else "failed"
    #     logger.info(f"[{run_id}] Subprocess completed with return code: {process.returncode}")
    # except subprocess.TimeoutExpired:
    #     result["status"] = "timeout"
    #     result["error"] = "Evaluation timed out after 5 minutes"
    #     logger.error(f"[{run_id}] Subprocess timed out")
    # except Exception as e:
    #     result["status"] = "error"
    #     result["error"] = str(e)
    #     logger.error(f"[{run_id}] Subprocess error: {e}")
    
    # Option 2: Call the FastAPI backend on port 8000
    try:
        import requests as req_lib
        
        logger.info(f"[{run_id}] Building API payload for FastAPI backend")
        
        # Map parameters to the existing API format
        api_payload = {
            "evaluation_object": model,
            "use_case": evaluation_goals if isinstance(evaluation_goals, str) else ", ".join(str(g) for g in evaluation_goals),
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
            ] if metrics == "all" else [{"metric_id": m, "threshold": 0.7, "init_params": {}} for m in (metrics if isinstance(metrics, list) else [metrics])],
            "test_cases": [
                {
                    "input": "Test input",
                    "actual_output": "Test output",
                    "retrieval_context": ["Test context"]
                }
            ]
        }
        
        logger.debug(f"[{run_id}] API Payload: {json.dumps(api_payload, indent=2)}")
        
        # Note: Uncomment to actually call the FastAPI backend
        # logger.info(f"[{run_id}] Calling FastAPI backend at http://localhost:8000/v1/evaluate")
        # response = req_lib.post("http://localhost:8000/v1/evaluate", json=api_payload, timeout=300)
        # result["api_response"] = response.json()
        # logger.info(f"[{run_id}] FastAPI response status: {response.status_code}")
        
        result["api_payload_preview"] = api_payload
        result["integration_note"] = "Ready to integrate with FastAPI backend on port 8000"
        
    except ImportError:
        result["note"] = "requests library not available for API integration"
        logger.warning(f"[{run_id}] requests library not available")
    except Exception as e:
        result["integration_error"] = str(e)
        logger.error(f"[{run_id}] Integration error: {e}")
    
    # Save result to artifacts
    run_dir = ARTIFACT_DIR / run_id
    result_path = run_dir / "evaluation_result.json"
    
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"[{run_id}] Evaluation result saved to: {result_path}")
    logger.info(f"[{run_id}] Evaluation execution complete")
    
    return result


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AI Evaluation API Starting")
    logger.info("=" * 60)
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Artifacts directory: {ARTIFACT_DIR}")
    
    print("=" * 60)
    print("AI Evaluation API")
    print("=" * 60)
    print(f"Server starting on http://localhost:5009")
    print(f"Artifacts directory: {ARTIFACT_DIR}")
    print(f"Log file: {LOG_FILE}")
    print()
    print("Endpoints:")
    print("  GET  /                    - Health check")
    print("  POST /aievaluation        - Run evaluation")
    print("  GET  /aievaluation/<id>   - Get evaluation status")
    print("  GET  /aievaluation/list   - List all evaluations")
    print("  GET  /logs                - View recent logs")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5009, debug=True)
