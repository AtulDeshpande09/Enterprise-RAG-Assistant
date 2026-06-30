import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import ollama

app = FastAPI(title="Phi-4 Mini Core Node")
templates = Jinja2Templates(directory="templates")

class ChatPayload(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Notice we explicitly add context=
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/chat")
async def chat_stream(payload: ChatPayload):
    async def event_generator():
        try:
            # Connect directly to the local C++ model runner
            response = await ollama.AsyncClient().chat(
                model='phi4-mini',
                messages=[{'role': 'user', 'content': payload.prompt}],
                stream=True
            )
            async for chunk in response:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    yield f"data: {content}\n\n"
        except Exception as e:
            yield f"data: [Error: {str(e)}]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 listens to all local home network connections on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

