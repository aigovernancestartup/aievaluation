# Quick Start Guide

## 1. Start Ollama Server

**IMPORTANT**: You must have Ollama running before testing the API!

```bash
# In Terminal 1: Start Ollama server
ollama serve
```

## 2. Verify Ollama is Running

```bash
# Check if Ollama is accessible
curl http://localhost:11434/api/tags
```

## 3. Start the Backend API

```bash
# In Terminal 2: Start the FastAPI server
cd backend
uvicorn app:app --reload --port 5008
```

## 4. Test the API

```bash
# In Terminal 3: Run the test
cd backend
python test_api_simple.py
```

Or open in browser: `http://localhost:5008/docs`

## Troubleshooting

- **"Connection refused" or timeout errors**: Make sure Ollama is running (`ollama serve`)
- **"Model not found"**: Run `ollama pull deepseek-r1:1.5b`
- **API errors**: Check that both Ollama (port 11434) and FastAPI (port 5008) are running


