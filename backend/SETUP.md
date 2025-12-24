# Backend Setup Guide

## Prerequisites

1. **Install Ollama** (if not already installed):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull DeepSeek R1 1.5B model**:
   ```bash
   ollama pull deepseek-r1:1.5b
   ```

3. **Start Ollama server** (in a separate terminal):
   ```bash
   ollama serve
   ```
   The server will run on `http://localhost:11434` by default.

## Installation

1. **Install Python dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

## Running the Backend

1. **Start the FastAPI server**:
   ```bash
   cd backend
   uvicorn app:app --reload --port 5008
   ```

2. **Test the API**:
   - Open browser: `http://localhost:5008/docs` (FastAPI auto-docs)
   - Or run: `python test_api_simple.py`

## Configuration

The backend is configured to use Ollama with DeepSeek R1 1.5B by default. You can override these via environment variables:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
- `OLLAMA_MODEL` (default: `deepseek-r1:1.5b`)
- `OLLAMA_API_KEY` (default: `ollama`)
- `OLLAMA_COST_PER_INPUT_TOKEN` (default: `0.0`)
- `OLLAMA_COST_PER_OUTPUT_TOKEN` (default: `0.0`)

## Available Metrics

Use metric IDs from the schema, for example:
- `rag.answer_relevancy`
- `rag.faithfulness`
- `rag.contextual_precision`
- `rag.contextual_recall`
- `rag.contextual_relevancy`
- `agent.task_completion`
- `safety.bias`
- `safety.toxicity`
- And more...

See `deepEval_metrics.schema.yaml` for the complete list.


