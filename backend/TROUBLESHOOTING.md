# Troubleshooting Timeout Issues

## Problem: API Request Times Out

If you're seeing `Read timed out` errors, here's how to fix it:

---

## Solution 1: Warm Up the Model First

The first call is always slowest because the model needs to load into memory.

**Before testing the API, warm up Ollama:**
```bash
# This loads the model into memory
ollama run deepseek-r1:1.5b "hello"
```

Wait for it to complete, then try your API test again.

---

## Solution 2: Increase Timeout

The model can take 2-5 minutes for the first evaluation. Update the timeout:

**In `test_api_simple.py`:**
```python
response = requests.post(API_URL, json=payload, timeout=300)  # 5 minutes
```

Or use `test_api_fast.py` which has a longer timeout.

---

## Solution 3: Check Ollama Performance

Test Ollama directly to see response time:

```bash
# Time a simple request
time curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1:1.5b",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 10
  }'
```

If this takes >60 seconds, the model is just slow on your hardware.

---

## Solution 4: Use a Faster Model

DeepSeek R1 1.5B can be slow. Try a faster model:

**Option A: Use a smaller/faster model**
```bash
ollama pull llama3.2:1b  # Much faster
```

Then update `config.py` or set environment variable:
```bash
export OLLAMA_MODEL=llama3.2:1b
```

**Option B: Use GPU acceleration**
If you have a GPU, Ollama will use it automatically and be much faster.

---

## Solution 5: Check Server Logs

See what's happening on the backend:

**Check uvicorn logs:**
Look at the terminal where you ran `uvicorn app:app --port 5008`

**Check Ollama logs:**
Look at the terminal where you ran `ollama serve`

---

## Solution 6: Test API Without LLM

To verify the API works without waiting for the model:

**Use a deterministic metric** (if available in your schema) that doesn't require LLM calls.

Or **test the API endpoint directly**:
```bash
# This should respond immediately (even if metric fails)
curl -X POST http://localhost:5008/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_object": "test",
    "use_case": "test",
    "context": {"deployment_stage": "dev", "risk_class": "low", "user_impact": "internal"},
    "run": {"mode": "one_off", "environment": "local"},
    "metrics": [{"metric_id": "rag.answer_relevancy", "threshold": 0.7, "init_params": {}}],
    "test_cases": [{"input": "test", "actual_output": "test", "retrieval_context": ["test"]}]
  }' \
  --max-time 300
```

---

## Expected Behavior

### First Call:
- **Time**: 2-5 minutes (model loading + processing)
- **Status**: This is normal!

### Subsequent Calls:
- **Time**: 30-120 seconds (model already loaded)
- **Status**: Much faster

### If Still Timing Out:
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Check backend is running: `curl http://localhost:5008/docs`
- Try warming up the model first (Solution 1)
- Consider using a faster model (Solution 4)

---

## Quick Diagnostic Commands

```bash
# 1. Check Ollama is running
curl http://localhost:11434/api/tags

# 2. Check backend is running  
curl http://localhost:5008/docs

# 3. Warm up model
ollama run deepseek-r1:1.5b "test"

# 4. Test Ollama directly (should respond in <60s after warmup)
time curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1:1.5b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```


