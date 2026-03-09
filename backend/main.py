import os
import itertools
import json
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="Arafath AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase ──────────────────────────────────────────────
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# ── Gemini API Key Rotation ───────────────────────────────
raw_keys = os.getenv("GEMINI_API_KEYS", "")
api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
if not api_keys:
    raise ValueError("No GEMINI_API_KEYS found in .env")

key_cycle = itertools.cycle(api_keys)

def get_next_key() -> str:
    return next(key_cycle)

# ── System Prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are the personal AI assistant of Arafath Al Adnan.
You represent him professionally on his portfolio website.

About Arafath:
- Full Name: Arafath Al Adnan
- Location: Sylhet, Bangladesh
- Email: araafathaladnan@gmail.com
- LinkedIn: linkedin.com/in/arafathaladnan
- Role: AI Automation Engineer | Python Developer | No-Code & Low-Code AI Builder | n8n Expert

Skills & Technologies:
- Python, C#, SQL & Database Engineering
- React + Vite, Flutter, Next.js (learning)
- FastAPI, REST API Development
- n8n Automation, AI Agents
- No-Code / Low-Code AI Systems
- Google AI Studio, Gemini API
- Supabase, Database Design
- WhatsApp/Telegram/Messenger chat automation
- Workflow Design & Business Automation

Experience:
- AI Automation Engineer (Freelance) — Designed and deployed AI-powered automation workflows
- Software Engineer Trainee @ ICT Bangladesh — Built desktop software with C# and database systems
- Python Developer (Self-Learning) — Built logic-based and automation-focused Python projects

Education:
- HSC, Science — Sylhet Government Model School & College
- SSC, Science — Bangladesh Bank School
- Completed: AI Based Professional Software Engineering @ ICT Bangladesh
  (Covered: Python, C#, Database, React+Vite, Flutter, API Development, AI Engineering, SDLC)

Instructions:
- Respond in the same language the user writes in (Bangla or English or both)
- Be professional and formal in tone
- Answer questions about Arafath's skills, experience, projects, and services
- If asked something you don't know about Arafath, say so honestly
- Never make up fake projects or experiences
- Keep responses clear and concise
"""

# ── Request Model ─────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str

# ── Supabase Helpers ──────────────────────────────────────
def save_message(session_id: str, role: str, content: str):
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        print(f"Supabase save error: {e}")

def get_history(session_id: str) -> list:
    try:
        result = supabase.table("messages") \
            .select("role, content") \
            .eq("session_id", session_id) \
            .order("created_at") \
            .limit(10) \
            .execute()
        return result.data or []
    except Exception as e:
        print(f"Supabase fetch error: {e}")
        return []

# ── Gemini Streaming ──────────────────────────────────────
async def gemini_stream(history: list, user_message: str, session_id: str):
    try:
        # Rotate API key
        genai.configure(api_key=get_next_key())
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT
        )

        # Build chat history for context
        chat_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})

        # Start chat with history
        chat = model.start_chat(history=chat_history)

        full_reply = ""

        # Run streaming in thread (SDK is sync)
        def stream_sync():
            return chat.send_message(user_message, stream=True)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, stream_sync)

        # Yield each chunk
        for chunk in response:
            if chunk.text:
                full_reply += chunk.text
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                await asyncio.sleep(0)

    except Exception as e:
        print(f"Gemini error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    # Save to Supabase after full response
    save_message(session_id, "user", user_message)
    save_message(session_id, "assistant", full_reply)
    yield f"data: {json.dumps({'done': True})}\n\n"


# ── Routes ────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    history = get_history(req.session_id)
    return StreamingResponse(
        gemini_stream(history, req.message, req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/api/history/{session_id}")
async def get_chat_history(session_id: str):
    return get_history(session_id)

@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    try:
        supabase.table("messages").delete().eq("session_id", session_id).execute()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")