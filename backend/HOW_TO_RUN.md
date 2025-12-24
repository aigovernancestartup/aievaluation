# How to Run the Backend - Step by Step

## Prerequisites Check

First, make sure you have:
- Python 3.8+ installed
- Ollama installed (see Step 1)

---

## Step 1: Install and Setup Ollama

### Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Download the DeepSeek R1 1.5B model:
```bash
ollama pull deepseek-r1:1.5b
```

This will download ~1.1GB. Wait for it to complete.

---

## Step 2: Start Ollama Server

**Open Terminal 1** and run:
```bash
ollama serve
```

Keep this terminal open. You should see Ollama start on `http://localhost:11434`

**Verify it's working:**
```bash
# In a new terminal, test:
curl http://localhost:11434/api/tags
```

You should see JSON with your models listed.

---

## Step 3: Install Python Dependencies

**Open Terminal 2** and run:
```bash
cd /Users/anmolkumar/Personal/Zenjin/aievaluation-main/backend
pip install -r requirements.txt
```

Wait for all packages to install.

---

## Step 4: Start the FastAPI Backend Server

**Still in Terminal 2**, run:
```bash
cd /Users/anmolkumar/Personal/Zenjin/aievaluation-main/backend
uvicorn app:app --reload --port 5008
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:5008
INFO:     Application startup complete.
```

**Keep this terminal open!**

---

## Step 5: Test the API

### Option A: Use the Test Script

**Open Terminal 3** and run:
```bash
cd /Users/anmolkumar/Personal/Zenjin/aievaluation-main/backend
python test_api_simple.py
```

### Option B: Use the Browser

Open your browser and go to:
```
http://localhost:5008/docs
```

This opens FastAPI's interactive API documentation where you can test endpoints directly.

### Option C: Use curl

```bash
curl -X POST http://localhost:5008/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
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
      "init_params": {}
    }],
    "test_cases": [{
      "input": "test question",
      "actual_output": "test answer",
      "retrieval_context": ["context 1", "context 2"]
    }]
  }'
```

---

## Expected Behavior

### ✅ Success:
- API returns status 200
- Response includes `run_id`, `overall_status`, and `metric_results`
- Evidence file is created in `backend/artifacts/<run_id>/evidence.json`

### ⚠️ Slow Responses:
- First evaluation may take 30-60+ seconds (model is loading)
- Subsequent evaluations may be faster
- This is normal for local LLM inference

### ❌ Common Issues:

**"Connection refused" to Ollama:**
- Make sure `ollama serve` is running in Terminal 1
- Check: `curl http://localhost:11434/api/tags`

**"Model not found":**
- Run: `ollama pull deepseek-r1:1.5b`
- Verify: `ollama list` should show the model

**"Module not found" errors:**
- Make sure you ran: `pip install -r requirements.txt`
- Check you're in the `backend/` directory

**Timeout errors:**
- The model is slow - increase timeout in test script
- Or wait longer (first call can take 60+ seconds)

---

## Quick Reference

**Terminal 1:** `ollama serve` (keep running)  
**Terminal 2:** `uvicorn app:app --reload --port 5008` (keep running)  
**Terminal 3:** Run tests or make API calls

**API URL:** `http://localhost:5008`  
**API Docs:** `http://localhost:5008/docs`  
**Ollama:** `http://localhost:11434`

---

## Stopping Everything

1. Press `Ctrl+C` in Terminal 2 (stops FastAPI)
2. Press `Ctrl+C` in Terminal 1 (stops Ollama)
3. Or close the terminals

---

## Next Steps

- Check `backend/artifacts/` for evaluation results
- Explore available metrics in `deepEval_metrics.schema.yaml`
- Customize model settings in `config.py` or via environment variables

