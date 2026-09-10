# Arafath AI

**A personal AI assistant for Arafath Al Adnan's portfolio and professional profile.**

Arafath AI is a full-stack personal chatbot built to answer questions about Arafath's skills, projects, experience, services, and professional background. It uses **Google Gemini** for generation, **FastAPI** as the backend API, and **Supabase** for conversation storage.

## Live Application

- **Primary:** https://www.arafath-al-adnan.work.gd/
- **Render deployment:** https://arafath-ai.onrender.com/

<img src="https://raw.githubusercontent.com/apache/netbeans/master/ide/bugtracking.bridge/src/org/netbeans/modules/bugtracking/bridge/resources/warning.gif" width="18" height="18" alt="Warning" /> **Current status:** This project is still under active development. Conversation history depends on the Supabase database being available; when the hosted Supabase project is sleeping or unavailable, the assistant may not be able to retrieve previous messages reliably.

## What It Does

Arafath AI provides a conversational interface for learning about Arafath Al Adnan. The backend can:

- Answer questions using a dedicated personal-assistant system prompt
- Respond in Bangla, English, or a combination based on the user's language
- Maintain conversation context through stored session messages
- Stream Gemini responses to the frontend in real time
- Retrieve and clear chat history by session
- Run as a single FastAPI application serving both the API and frontend

## Current AI Architecture

```text
User
  |
  v
Frontend Chat UI
  |
  v
FastAPI /api/chat
  |
  +----> Supabase ----> Session Conversation History
  |
  v
Google Gemini
  |
  v
Streaming Response
  |
  v
Frontend
```

### Conversation Memory

The current implementation uses **session-based chat history stored in Supabase**. It is **not a vector RAG system** and does not currently use embeddings or a vector database for semantic retrieval.

The backend retrieves recent messages for a session and sends them to Gemini as conversation context. This provides short conversational continuity rather than long-term semantic memory.

## Planned Direction

A future version is planned to evolve Arafath AI into a more capable **RAG + AI Agent system**, with components such as:

- Vector-based knowledge retrieval
- Personal knowledge-base ingestion
- Semantic search over projects, skills, and documents
- More persistent long-term memory
- Tool/API integrations
- Agentic workflows and task execution

These are planned improvements and are **not represented as current features**.

## Core Technologies

| Layer | Technology |
|---|---|
| AI | Google Gemini 2.5 Flash |
| Backend | Python, FastAPI |
| Database | Supabase |
| Frontend | HTML, CSS, JavaScript |
| Streaming | Server-Sent Events (SSE) |
| Hosting | Render |
| Environment | Python + Uvicorn |

## Project Structure

```text
arafath_ai/
├── backend/
│   └── main.py          # FastAPI backend, Gemini integration and Supabase history
├── frontend/
│   └── index.html       # Chat interface
├── .gitignore           # Local environment and Python exclusions
├── render.yaml          # Render deployment configuration
├── requirements.txt     # Python dependencies
└── run.bat              # Windows local development launcher
```

## Environment Variables

Create a local `.env` file or configure the variables in your hosting provider:

```env
GEMINI_API_KEYS=your_key_1,your_key_2
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_key
```

**Never commit real API keys, Supabase credentials, or `.env` files to GitHub.**

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/adnan1299otm/arafath_ai.git
cd arafath_ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `.env` in the project root with the required Gemini and Supabase credentials.

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The Windows `run.bat` launcher can also be used for local development.

## Deployment

The repository includes `render.yaml` for deploying the FastAPI application on Render. Runtime credentials are configured through environment variables rather than stored in the repository.

## Database

Supabase currently stores messages using a session-based structure. The application reads recent messages for a session and can also clear the stored history through the backend API.

Because the hosted database may sleep or become temporarily unavailable, database-backed memory should be considered **availability-dependent** in the current deployment.

## Security Notes

- Keep `.env` files out of version control.
- Store Gemini and Supabase credentials in environment variables.
- Do not expose server-side credentials in frontend code.
- Use production-specific Supabase security policies before treating the deployment as a production-grade public service.

## Project Status

**Active development** — the current version is a working personal AI chatbot prototype and foundation for a future RAG-based personal AI agent.

## Author

**Arafath Al Adnan**

- Portfolio: https://www.arafath-al-adnan.work.gd/
- GitHub: https://github.com/adnan1299otm
- LinkedIn: https://www.linkedin.com/in/arafathaladnan/

## License

This project is currently presented as a personal portfolio project. See the repository for the applicable source and usage terms.
