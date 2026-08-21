@echo off
echo ====================================================
echo   Starting FinDocs-AI Production FastAPI Server   
echo   Swagger Docs: http://localhost:8000/docs        
echo ====================================================
.\venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
pause
