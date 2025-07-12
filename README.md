# Sub200ms Groq AI API ⚡

Ultra-fast AI poetry generation powered by Groq's LPU chips. Achieves 315ms average response time with 95% of requests under 500ms.

**Live Performance:** 315ms average, 95% sub-500ms, 100% reliability

## Why Groq is Fast

Groq uses custom **LPU (Language Processing Unit) chips** designed specifically for AI inference, delivering 2-3x faster performance than traditional GPU setups.

## Setup

```bash
git clone https://github.com/your-username/sub200groq.git
cd sub200groq
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your-key" > .env
python main.py
```

## Speed Test Results

**Performance metrics from 20 test requests:**
- Average response: 315ms
- 95% under 500ms
- Fastest request: 254ms
- 100% success rate
- 2-3x faster than OpenAI

## Testing Performance

**Quick speed test:**
```http
POST http://localhost:8000/chat/complete
Content-Type: application/json

{
  "message": "Hi",
  "max_tokens": 5
}
```

**Streaming test (near-instant):**
```http
POST http://localhost:8000/chat/stream
Content-Type: application/json

{
  "message": "Write a haiku about lightning"
}
```

**Load test:**
```http
POST http://localhost:8000/test/speed
Content-Type: application/json

{
  "concurrent_requests": 5,
  "message": "Fast poem"
}
```

## Check Your Performance

```http
GET http://localhost:8000/metrics/latency
```

Key metrics:
- `sub_200ms_percentage` - Ultra-fast requests
- `sub_500ms_percentage` - Should be >90%
- `mean` - Average response time

## Available Models

- `llama-3.1-8b-instant` (default - fastest)
- `llama-3.1-70b-versatile` (more capable)
- `gemma2-9b-it` (alternative fast option)

## Deploy

1. Push to GitHub
2. Connect to Render
3. Set `GROQ_API_KEY` environment variable
4. Deploy

Powered by Groq LPU. Built for speed.