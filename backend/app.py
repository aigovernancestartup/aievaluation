from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import tempfile
import os
import json

from config import SCHEMA_PATH, ARTIFACT_DIR
from schema_registry import load_schema, build_metric_index
from schemas_api import EvaluateRequest, EvaluateResponse, MetricResult, EvaluateFromConfigRequest
from runner import run_evaluation
from evidence_store import write_evidence
from config_parser import (
    load_evaluation_config,
    extract_test_cases_from_config,
    build_metric_name_to_id_map
)

app = FastAPI(
    title="AI Evaluation Engine Backend",
    version="1.0.0",
    description="API for running AI model evaluations with configurable metrics and test cases"
)

# =============================================================================
# CORS Configuration - Handle preflight OPTIONS requests
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify: ["http://localhost:8080", "http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Type", "X-Request-Id"],
    max_age=86400,  # Cache preflight response for 24 hours
)

# Create router with /v1/aievaluation prefix
router = APIRouter(prefix="/v1/aievaluation", tags=["AI Evaluation"])

_schema = load_schema(str(SCHEMA_PATH))
_metric_index = build_metric_index(_schema)
_metric_name_to_id_map = build_metric_name_to_id_map(_metric_index)

# Default config path (relative to backend directory)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "Testing" / "testingconfig" / "evaluation_config.json"


# =============================================================================
# Health Check Endpoints
# =============================================================================
@app.get("/", tags=["Health"])
@app.get("/v1", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Evaluation Engine",
        "version": "1.0.0",
        "api_base_path": "/v1/aievaluation",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "evaluate": "POST /v1/aievaluation/evaluate",
            "evaluate_from_config": "POST /v1/aievaluation/evaluate-from-config"
        }
    }


# =============================================================================
# Evaluation Endpoints
# =============================================================================
@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    if not req.metrics:
        raise HTTPException(status_code=400, detail="No metrics selected.")

    # Convert Pydantic models to dict payloads for runner
    metrics = [m.model_dump() for m in req.metrics]
    test_cases = [tc.model_dump(exclude_none=True) for tc in req.test_cases]

    run_id, evidence, raw_results = run_evaluation(
        metric_index=_metric_index,
        metrics=metrics,
        test_cases=test_cases,
        context=req.context.model_dump(),
        evaluation_object=req.evaluation_object,
        use_case=req.use_case,
    )

    evidence_path = write_evidence(ARTIFACT_DIR, run_id, evidence)

    # Decision-first status (v1 heuristic)
    any_fail = any(r.get("passed") is False for r in raw_results if r.get("passed") is not None)
    any_error = any(r.get("error") for r in raw_results)

    overall_status = "PASS"
    if any_fail:
        overall_status = "FAIL"
    elif any_error:
        overall_status = "WARNING"

    metric_results = [MetricResult(**r) for r in raw_results]

    return EvaluateResponse(
        run_id=run_id,
        overall_status=overall_status,
        metric_results=metric_results,
        evidence_pointer=evidence_path,
    )


@router.post("/evaluate-from-config", response_model=EvaluateResponse)
async def evaluate_from_config(
    evaluation_id: str = Form(...),
    evaluation_name: str = Form(...),
    goals: str = Form(None),  # Comma-separated list
    metrics: str = Form(None),  # Comma-separated list
    deployment_stage: str = Form("dev"),
    risk_class: str = Form("low"),
    user_impact: str = Form("internal"),
    domain: str = Form(None),
    mode: str = Form("one_off"),
    environment: str = Form("local"),
    config_file: UploadFile = File(None)
):
    """
    Evaluate using test cases from a config file.
    
    - If config_file is provided, use that file
    - If config_file is not provided, use the default evaluation_config.json
    - Filter by goals and metrics if provided (comma-separated lists)
    """
    # Parse goals and metrics from comma-separated strings
    goals_list = []
    if goals:
        goals_list = [g.strip() for g in goals.split(",") if g.strip()]
    
    metrics_list = []
    if metrics:
        metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    
    # Load config file
    config_path = None
    is_temp_file = False
    if config_file and config_file.filename:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".json") as tmp_file:
            content = await config_file.read()
            tmp_file.write(content)
            config_path = tmp_file.name
            is_temp_file = True
    else:
        # Use default config
        if not DEFAULT_CONFIG_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Default config file not found at {DEFAULT_CONFIG_PATH}. Please upload a config file."
            )
        config_path = str(DEFAULT_CONFIG_PATH)
    
    try:
        # Load and parse config
        config = load_evaluation_config(config_path)
        
        # Extract test cases and metric configs
        test_cases, metric_configs = extract_test_cases_from_config(
            config=config,
            goals=goals_list,
            metrics=metrics_list,
            name_to_id_map=_metric_name_to_id_map
        )
        
        if not metric_configs:
            raise HTTPException(
                status_code=400,
                detail="No matching metrics found in config. Check metric names and goals."
            )
        
        if not test_cases:
            raise HTTPException(
                status_code=400,
                detail="No test cases found for the specified goals and metrics."
            )
        
        # Build context
        from schemas_api import ContextModel, RunConfigModel
        context = ContextModel(
            deployment_stage=deployment_stage,
            risk_class=risk_class,
            user_impact=user_impact,
            domain=domain
        )
        
        run_config = RunConfigModel(
            mode=mode,
            environment=environment
        )
        
        # Run evaluation
        run_id, evidence, raw_results = run_evaluation(
            metric_index=_metric_index,
            metrics=metric_configs,
            test_cases=test_cases,
            context=context.model_dump(),
            evaluation_object=evaluation_name,
            use_case=evaluation_id,
        )
        
        evidence_path = write_evidence(ARTIFACT_DIR, run_id, evidence)
        
        # Decision-first status (v1 heuristic)
        any_fail = any(r.get("passed") is False for r in raw_results if r.get("passed") is not None)
        any_error = any(r.get("error") for r in raw_results)
        
        overall_status = "PASS"
        if any_fail:
            overall_status = "FAIL"
        elif any_error:
            overall_status = "WARNING"
        
        metric_results = [MetricResult(**r) for r in raw_results]
        
        return EvaluateResponse(
            run_id=run_id,
            overall_status=overall_status,
            metric_results=metric_results,
            evidence_pointer=evidence_path,
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing config: {str(e)}")
    
    finally:
        # Clean up temp file if we created one
        if is_temp_file and config_path and os.path.exists(config_path):
            try:
                os.unlink(config_path)
            except:
                pass


# =============================================================================
# Register Router
# =============================================================================
app.include_router(router)
