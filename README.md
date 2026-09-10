<div align="center">

# Arafath AI

### Personal AI Assistant for Arafath Al Adnan

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3" />
</p>

<p>
  <a href="https://www.arafath-al-adnan.work.gd/"><strong>Live Application</strong></a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://arafath-ai.onrender.com/"><strong>Render Deployment</strong></a>
</p>

</div>

---

## Overview

**Arafath AI** is a full-stack personal AI chatbot created to answer questions about **Arafath Al Adnan's skills, projects, experience, services, and professional background**.

The application combines a **Python + FastAPI backend**, a lightweight **HTML/CSS/JavaScript frontend**, **Google Gemini** for AI responses, and **Supabase's PostgreSQL-backed database** for session-based conversation history.

> <img src="https://raw.githubusercontent.com/apache/netbeans/master/ide/bugtracking.bridge/src/org/netbeans/modules/bugtracking/bridge/resources/warning.gif" width="18" height="18" alt="Warning" /> **Current status:** This project is still under active development. Conversation history depends on the hosted Supabase database being available; if the database is sleeping or temporarily unavailable, previous messages may not be retrieved reliably.

---

## Architecture

<div align="center">

```text
┌─────────────────────┐
│      User / UI      │
│    HTML + CSS + JS  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     FastAPI API     │
│     Python Backend  │
└──────┬────────┬─────┘
       │        │
       │        ▼
       │   ┌─────────────────┐
       │   │  Google Gemini  │
       │   │  AI Generation  │
       │   └─────────────────┘
       │
       ▼
┌─────────────────────┐
│ Supabase / PostgreSQL│
│ Session Chat History │
└─────────────────────┘
```

</div>

### How it works

1. The user sends a message through the web interface.
2. The FastAPI backend receives the request through `/api/chat`.
3. Recent session messages are retrieved from Supabase when available.
4. Google Gemini generates the response using the personal-assistant system prompt.
5. The response is streamed back to the frontend using **Server-Sent Events (SSE)**.
6. User and assistant messages are stored in the database for the current session.

---

## Features

<table>
<tr>
<td width="50%">

### AI Conversation

- Personal AI assistant behavior
- Google Gemini 2.5 Flash
- Bangla, English, and mixed-language conversations
- Streaming responses

</td>
<td width="50%">

### Conversation History

- Session-based memory
- Recent message retrieval
- Supabase database storage
- History retrieval and clearing endpoints

</td>
</tr>
<tr>
<td width="50%">

### Backend

- Python
- FastAPI
- Uvicorn
- REST-style API endpoints
- Server-Sent Events

</td>
<td width="50%">

### Deployment

- Render web service
- Environment-based credentials
- Static frontend served by FastAPI
- Windows local launcher

</td>
</tr>
</table>

---

## Current Memory System

Arafath AI currently uses **session-based conversation history**, not vector RAG.

The application stores messages in Supabase and retrieves recent messages for the active session before generating a response. This provides conversational continuity, but it is **not semantic long-term memory** and does not currently use embeddings or a vector database.

### Planned Evolution

The longer-term direction is to evolve the project into a **RAG + AI Agent architecture**, potentially adding:

- Vector-based knowledge retrieval
- Personal knowledge-base ingestion
- Embeddings and semantic search
- Long-term memory
- Project and document retrieval
- Tool/API integrations
- Agentic task execution

These are planned capabilities, not current features.

---

## Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Language** | Python |
| **Backend** | FastAPI + Uvicorn |
| **AI** | Google Gemini 2.5 Flash |
| **Database** | Supabase / PostgreSQL |
| **Frontend** | HTML5 + CSS3 + JavaScript |
| **Streaming** | Server-Sent Events (SSE) |
| **Hosting** | Render |

</div>

### Technology Icons

<p align="center">
  <img src="https://cdn.simpleicons.org/python" width="42" height="42" alt="Python" />
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/fastapi" width="42" height="42" alt="FastAPI" />
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/postgresql" width="42" height="42" alt="PostgreSQL" />
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/html5" width="42" height="42" alt="HTML5" />
  &nbsp;&nbsp;
  <img src="https://cdn.simpleicons.org/css3" width="42" height="42" alt="CSS3" />
</p>

---

## Project Structure

```text
arafath_ai/
├── backend/
│   └── main.py          # FastAPI backend, Gemini integration and chat history
├── frontend/
│   └── index.html       # HTML/CSS/JavaScript chat interface
├── .gitignore           # Local environment and Python exclusions
├── render.yaml          # Render deployment configuration
├── requirements.txt     # Python dependencies
└── run.bat              # Windows local development launcher
```

---

## Environment Variables

Create a local `.env` file or configure these variables in your hosting provider:

```env
GEMINI_API_KEYS=your_key_1,your_key_2
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_key
```

> **Security:** Never commit real API keys, Supabase credentials, or `.env` files to GitHub.

---

## Local Development

### 1. Clone

```bash
git clone https://github.com/adnan1299otm/arafath_ai.git
cd arafath_ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create `.env` in the project root and add your own Gemini and Supabase credentials.

### 4. Run locally

```bash
uvicorn backend.main:app --reload --port 8000
```

For Windows, the included `run.bat` launcher can also be used.

---

## Deployment

The project includes a `render.yaml` configuration for deployment on **Render**.

Runtime credentials are supplied through environment variables rather than stored in the repository.

---

## Database

The application uses **Supabase** for storing session messages. Supabase provides the PostgreSQL-backed database layer used by the project.

The current memory implementation is intentionally simple: recent messages are fetched by `session_id` and supplied as conversational context to Gemini.

Because the hosted database can become temporarily unavailable, database-backed conversation memory is **availability-dependent** in the current deployment.

---

## Security Considerations

- Keep `.env` files outside version control.
- Store Gemini and Supabase credentials as environment variables.
- Do not expose server-side credentials in frontend code.
- Apply appropriate Supabase security policies before treating the deployment as a production-grade service.
- Restrict CORS origins for a hardened production deployment instead of allowing every origin.

---

## Project Status

**Active development** — Arafath AI is currently a working personal AI chatbot and the foundation for a future **RAG-powered personal AI agent**.

---

## Author

<div align="center">

### Arafath Al Adnan

Software Engineer · AI Builder · Python Developer · AI Automation Engineer

<p>
  <a href="https://www.arafath-al-adnan.work.gd/">Portfolio</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/adnan1299otm">GitHub</a>
  &nbsp;•&nbsp;
  <a href="https://www.linkedin.com/in/arafathaladnan/">LinkedIn</a>
</p>

</div>

---

<div align="center">

**Built as a personal AI project by Arafath Al Adnan.**

</div>
