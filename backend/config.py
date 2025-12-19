from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "deepEval_metrics.schema.yaml"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
