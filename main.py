import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from datetime import datetime, timedelta
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Global metrics storage
class MetricsStore:
    def __init__(self):
        self.latencies = deque(maxlen=1000)  # Keep last 1000 requests
        self.provider_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        self.start_time = time.time()
    
    def add_latency(self, total_time: float, llm_time: float, provider: str, success: bool = True):
        timestamp = time.time()
        self.latencies.append({
            "timestamp": timestamp,
            "total_time": total_time,
            "llm_time": llm_time,
            "provider": provider,
            "success": success
        })
        
        if success:
            self.provider_stats[provider]["count"] += 1
            self.provider_stats[provider]["total_time"] += total_time
        else:
            self.provider_stats[provider]["errors"] += 1
    
    def get_stats(self):
        if not self.latencies:
            return {"message": "No requests yet"}
        
        recent_latencies = [r["total_time"] for r in self.latencies if r["success"]]
        if not recent_latencies:
            return {"message": "No successful requests yet"}
        
        recent_latencies.sort()
        n = len(recent_latencies)
        
        stats = {
            "total_requests": len(self.latencies),
            "successful_requests": len(recent_latencies),
            "uptime_seconds": time.time() - self.start_time,
            "latency_ms": {
                "min": min(recent_latencies) * 1000,
                "max": max(recent_latencies) * 1000,
                "mean": sum(recent_latencies) / n * 1000,
                "p50": recent_latencies[int(n * 0.5)] * 1000,
                "p95": recent_latencies[int(n * 0.95)] * 1000,
                "p99": recent_latencies[int(n * 0.99)] * 1000,
            },
            "sub_200ms_percentage": len([x for x in recent_latencies if x < 0.2]) / n * 100,
            "sub_500ms_percentage": len([x for x in recent_latencies if x < 0.5]) / n * 100,
            "provider_stats": dict(self.provider_stats)
        }
        
        return stats


# Global instances
metrics = MetricsStore()
http_client = None
groq_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global http_client, groq_client
    
    # Initialize HTTP client for other requests
    timeout = httpx.Timeout(10.0, connect=2.0)  # Faster timeouts for speed
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    
    http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True
    )
    
    # Initialize Groq client
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        groq_client = AsyncGroq(api_key=api_key)
        
        # Test connection
        try:
            await groq_client.models.list()
        except:
            pass  # Continue if test fails
    
    yield
    
    # Shutdown
    await http_client.aclose()
    if groq_client:
        await groq_client.close()


# FastAPI app
app = FastAPI(
    title="Sub-200ms Groq AI API",
    description="Ultra-fast AI poetry generation powered by Groq's LPU chips",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    model: Optional[str] = "llama-3.1-8b-instant"  # Fast current Groq model
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7


class LoadTestRequest(BaseModel):
    concurrent_requests: int = 5
    message: str = "Write a short poem about speed"


# Groq Provider functions
async def call_groq_streaming(message: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 200, temperature: float = 0.7):
    """Call Groq API for streaming responses - optimized for speed"""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized - check API key")
    
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are a fast, creative poet. Write concise, beautiful poetry quickly."
            },
            {
                "role": "user", 
                "content": message
            }
        ]
        
        response = await groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


async def call_groq_complete(message: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 200, temperature: float = 0.7):
    """Call Groq API for non-streaming responses - optimized for speed"""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized - check API key")
    
    start_time = time.time()
    
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are a fast, creative poet. Write concise, beautiful poetry quickly."
            },
            {
                "role": "user", 
                "content": message
            }
        ]
        
        response = await groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        llm_time = time.time() - start_time
        content = response.choices[0].message.content
        return content, llm_time
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")


# API Routes
@app.get("/")
async def root():
    return {
        "message": "Sub-200ms Groq AI Poetry API",
        "description": "Ultra-fast poetry generation powered by Groq's LPU chips",
        "version": "1.0.0",
        "performance_targets": {
            "total_response": "< 200ms",
            "time_to_first_token": "< 100ms",
            "powered_by": "Groq LPU"
        },
        "docs_url": "/docs",
        "metrics_url": "/metrics/latency",
        "example_requests": {
            "ultra_fast": "Write one word",
            "quick_haiku": "Write a haiku about lightning",
            "speed_test": "Write a short poem about Groq's speed"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - metrics.start_time,
        "provider": "groq",
        "target_latency": "< 200ms"
    }


@app.post("/chat/complete")
async def chat_complete(request: ChatMessage):
    """Ultra-fast non-streaming chat completion with timing"""
    print(f"DEBUG: Received request: {request}")
    print(f"DEBUG: Message: {request.message}")
    print(f"DEBUG: Model: {request.model}")
    
    start_time = time.time()
    
    try:
        print("DEBUG: About to call call_groq_complete...")
        
        # Call Groq
        response_text, llm_time = await call_groq_complete(
            request.message,
            request.model,
            request.max_tokens,
            request.temperature
        )
        
        print(f"DEBUG: Got response, length: {len(response_text) if response_text else 0}")
        
        total_time = time.time() - start_time
        processing_time = total_time - llm_time
        
        # Record metrics
        metrics.add_latency(total_time, llm_time, "groq", True)
        
        # Return response with timing
        response_data = {
            "response": response_text,
            "timing": {
                "total_time_ms": round(total_time * 1000, 2),
                "llm_time_ms": round(llm_time * 1000, 2),
                "processing_time_ms": round(processing_time * 1000, 2)
            },
            "provider": "groq",
            "model": request.model,
            "performance": "ultra-fast" if total_time < 0.2 else "fast" if total_time < 0.5 else "normal"
        }
        
        return response_data
        
    except Exception as e:
        print(f"DEBUG: Exception occurred: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        total_time = time.time() - start_time
        metrics.add_latency(total_time, 0, "groq", False)
        raise e


@app.post("/chat/stream")
async def chat_stream(request: ChatMessage):
    """Ultra-fast streaming with Server-Sent Events"""
    
    async def generate():
        start_time = time.time()
        first_token_time = None
        
        try:
            async for chunk in call_groq_streaming(
                request.message,
                request.model,
                request.max_tokens,
                request.temperature
            ):
                if first_token_time is None:
                    first_token_time = time.time()
                    time_to_first_token = first_token_time - start_time
                    
                    # Send timing info as first event
                    yield f"data: {json.dumps({'type': 'timing', 'time_to_first_token_ms': round(time_to_first_token * 1000, 2), 'provider': 'groq'})}\n\n"
                
                # Send content chunk
                if chunk:
                    chunk_data = {
                        "type": "content",
                        "content": chunk
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send completion timing
            total_time = time.time() - start_time
            llm_time = total_time
            
            metrics.add_latency(total_time, llm_time, "groq", True)
            
            completion_data = {
                "type": "complete",
                "timing": {
                    "total_time_ms": round(total_time * 1000, 2),
                    "time_to_first_token_ms": round((first_token_time - start_time) * 1000, 2) if first_token_time else 0
                },
                "provider": "groq",
                "performance": "ultra-fast" if total_time < 0.2 else "fast" if total_time < 0.5 else "normal"
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            total_time = time.time() - start_time
            metrics.add_latency(total_time, 0, "groq", False)
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@app.get("/metrics/latency")
async def get_latency_metrics():
    """Get detailed latency statistics"""
    return metrics.get_stats()


@app.get("/metrics/live")
async def live_metrics():
    """Live metrics stream via Server-Sent Events"""
    
    async def generate():
        while True:
            stats = metrics.get_stats()
            yield f"data: {json.dumps(stats)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post("/test/speed")
async def speed_test(request: LoadTestRequest):
    """Built-in speed testing endpoint optimized for Groq"""
    
    async def single_request():
        try:
            chat_request = ChatMessage(message=request.message)
            start_time = time.time()
            
            response_text, llm_time = await call_groq_complete(
                chat_request.message,
                chat_request.model,
                chat_request.max_tokens,
                chat_request.temperature
            )
            
            total_time = time.time() - start_time
            return {"success": True, "time": total_time, "llm_time": llm_time}
            
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time}
    
    # Run concurrent requests
    start_time = time.time()
    tasks = [single_request() for _ in range(request.concurrent_requests)]
    results = await asyncio.gather(*tasks)
    total_test_time = time.time() - start_time
    
    # Analyze results
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    if successful_requests:
        response_times = [r["time"] for r in successful_requests]
        response_times.sort()
        n = len(response_times)
        
        stats = {
            "test_summary": {
                "total_requests": request.concurrent_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "test_duration_seconds": round(total_test_time, 3),
                "requests_per_second": round(request.concurrent_requests / total_test_time, 2)
            },
            "speed_stats": {
                "min_ms": round(min(response_times) * 1000, 2),
                "max_ms": round(max(response_times) * 1000, 2),
                "mean_ms": round(sum(response_times) / n * 1000, 2),
                "p50_ms": round(response_times[int(n * 0.5)] * 1000, 2),
                "p95_ms": round(response_times[int(n * 0.95)] * 1000, 2),
                "sub_200ms_count": len([x for x in response_times if x < 0.2]),
                "sub_200ms_percentage": round(len([x for x in response_times if x < 0.2]) / n * 100, 2),
                "sub_500ms_count": len([x for x in response_times if x < 0.5]),
                "sub_500ms_percentage": round(len([x for x in response_times if x < 0.5]) / n * 100, 2)
            },
            "groq_performance": "excellent" if sum(response_times) / n < 0.2 else "good" if sum(response_times) / n < 0.5 else "normal"
        }
        
        if failed_requests:
            stats["errors"] = [r["error"] for r in failed_requests]
        
        return stats
    else:
        return {
            "test_summary": {
                "total_requests": request.concurrent_requests,
                "successful_requests": 0,
                "failed_requests": len(failed_requests),
                "test_duration_seconds": round(total_test_time, 3)
            },
            "errors": [r["error"] for r in failed_requests]
        }


@app.get("/models")
async def list_models():
    """List available Groq models"""
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq client not initialized")
    
    try:
        models = await groq_client.models.list()
        return {
            "available_models": [model.id for model in models.data],
            "recommended_for_speed": ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "gemma2-9b-it"],
            "current_default": "llama-3.1-8b-instant"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)