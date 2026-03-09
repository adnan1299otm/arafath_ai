@echo off
echo Starting Arafath AI...
cd backend
uvicorn main:app --reload --port 8000
