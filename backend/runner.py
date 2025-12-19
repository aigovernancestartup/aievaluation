from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Tuple
import uuid
import time

from schema_registry import MetricDef
from deepeval_resolver import resolve_metric_class
from test_case_adapter import to_llm_test_case, to_conversational_test_case, to_arena_test_case

def _apply_constraints(metric_def: MetricDef, test_case: Dict[str, Any]) -> List[str]:
    """
    v1: implement only simple image-count constraint strings from schema.
    Your schema includes constraints like:
      - "input must contain exactly 1 image" :contentReference[oaicite:8]{index=8}
    """
    errors: List[str] = []
    for c in metric_def.constraints:
        c_lower = c.lower().strip()
        if "input must contain exactly" in c_lower and "image" in c_lower:
            # naive parser
            required = int(c_lower.split("exactly")[1].split("image")[0].strip())
            images = test_case.get("image_inputs") or []
            if len(images) != required:
                errors.append(f"{metric_def.metric_id}: constraint failed: {c} (got {len(images)})")
        if "actual_output must contain exactly" in c_lower and "image" in c_lower:
            required = int(c_lower.split("exactly")[1].split("image")[0].strip())
            images = test_case.get("image_outputs") or []
            if len(images) != required:
                errors.append(f"{metric_def.metric_id}: constraint failed: {c} (got {len(images)})")
    return errors

def _validate_required_fields(metric_def: MetricDef, test_case: Dict[str, Any]) -> List[str]:
    missing = []
    for f in metric_def.required_test_case_fields:
        if test_case.get(f) is None:
            missing.append(f)
    if missing:
        return [f"{metric_def.metric_id}: missing required test-case fields: {missing}"]
    return []

def _validate_required_init_params(metric_def: MetricDef, init_params: Dict[str, Any]) -> List[str]:
    missing = [p for p in metric_def.required_metric_init_params if init_params.get(p) is None]
    if missing:
        return [f"{metric_def.metric_id}: missing required metric init params: {missing}"]
    return []

def _pass_fail(metric_def: MetricDef, score: float, threshold: float) -> bool:
    if metric_def.threshold_semantics == "maximum_is_passing":
        return score <= threshold
    return score >= threshold  # minimum_is_passing default

def run_evaluation(
    metric_index: Dict[str, MetricDef],
    metrics: List[Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
    context: Dict[str, Any],
    evaluation_object: str,
    use_case: str,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:

    run_id = uuid.uuid4().hex
    started = time.time()

    results: List[Dict[str, Any]] = []
    evidence: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started,
        "evaluation_object": evaluation_object,
        "use_case": use_case,
        "context": context,
        "metrics": metrics,
        "test_cases": test_cases,
        "metric_evidence": [],
        "gaps": [],
    }

    # Execute metric-by-metric so you can return partial results deterministically
    for msel in metrics:
        metric_id = msel["metric_id"]
        metric_def = metric_index.get(metric_id)
        if metric_def is None:
            results.append({
                "metric_id": metric_id,
                "metric_name": metric_id,
                "error": "Unknown metric_id (not found in schema).",
            })
            continue

        init_params = dict(msel.get("init_params") or {})
        threshold_override = msel.get("threshold", None)
        if threshold_override is not None:
            init_params.setdefault("threshold", threshold_override)

        init_errs = _validate_required_init_params(metric_def, init_params)
        if init_errs:
            results.append({
                "metric_id": metric_def.metric_id,
                "metric_name": metric_def.metric_name,
                "error": "; ".join(init_errs),
            })
            continue

        # Test-case level validations
        tc_errors: List[str] = []
        for tc in test_cases:
            tc_errors += _validate_required_fields(metric_def, tc)
            tc_errors += _apply_constraints(metric_def, tc)

        if tc_errors:
            results.append({
                "metric_id": metric_def.metric_id,
                "metric_name": metric_def.metric_name,
                "error": "; ".join(tc_errors[:5]) + (" ..." if len(tc_errors) > 5 else ""),
            })
            continue

        # Instantiate metric
        try:
            MetricCls = resolve_metric_class(metric_def.metric_class)
            metric_obj = MetricCls(**init_params)
        except Exception as e:
            results.append({
                "metric_id": metric_def.metric_id,
                "metric_name": metric_def.metric_name,
                "error": f"Metric init failed: {type(e).__name__}: {e}",
            })
            continue

        # Build test case objects and measure
        metric_scores: List[float] = []
        metric_reasons: List[str] = []

        try:
            for tc in test_cases:
                if metric_def.test_case_type == "LLMTestCase":
                    tco = to_llm_test_case(tc)
                elif metric_def.test_case_type == "ConversationalTestCase":
                    evidence["gaps"].append({
                        "metric_id": metric_def.metric_id,
                        "gap": "ConversationalTestCase execution not implemented in v1.",
                    })
                    raise NotImplementedError("Conversational metrics not runnable in v1.")
                elif metric_def.test_case_type == "ArenaTestCase":
                    evidence["gaps"].append({
                        "metric_id": metric_def.metric_id,
                        "gap": "ArenaTestCase execution not implemented in v1.",
                    })
                    raise NotImplementedError("Arena metrics not runnable in v1.")
                else:
                    raise ValueError(f"Unknown test_case_type: {metric_def.test_case_type}")

                metric_obj.measure(tco)
                score = getattr(metric_obj, "score", None)
                if score is None:
                    raise RuntimeError("Metric returned no score (metric_obj.score is None).")

                metric_scores.append(float(score))
                reason = getattr(metric_obj, "reason", None)
                if isinstance(reason, str) and reason.strip():
                    metric_reasons.append(reason.strip())

            avg_score = sum(metric_scores) / len(metric_scores)
            threshold = getattr(metric_obj, "threshold", None)
            passed = None
            if threshold is not None:
                passed = _pass_fail(metric_def, avg_score, float(threshold))

            results.append({
                "metric_id": metric_def.metric_id,
                "metric_name": metric_def.metric_name,
                "score": avg_score,
                "threshold": threshold,
                "passed": passed,
                "reason": metric_reasons[0] if metric_reasons else None,
                "cost_signal": "llm_based" if "model" in init_params else "deterministic_or_unknown",
            })

            evidence["metric_evidence"].append({
                "metric_id": metric_def.metric_id,
                "metric_class": metric_def.metric_class,
                "scores": metric_scores,
                "reasons": metric_reasons[:3],
            })

        except NotImplementedError as e:
            results.append({
                "metric_id": metric_def.metric_id,
                "metric_name": metric_def.metric_name,
                "error": str(e),
            })
        except Exception as e:
            results.append({
                "metric_id": metric_def.metric_id,
                "metric_name": metric_def.metric_name,
                "error": f"Execution failed: {type(e).__name__}: {e}",
            })

    return run_id, evidence, results
