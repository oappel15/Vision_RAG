@echo off
REM Windows launcher for Vision RAG Migration Wizard (Web UI)
REM Double-click this file to start the wizard and open Chrome.

echo ============================================================
echo  Vision RAG Migration Wizard v4.0 (Web UI)
echo ============================================================
echo.
echo  Starting server... Chrome will open in a moment.
echo  If it doesn't open, go to: http://localhost:5555
echo  Close this window to stop the server.
echo.

start "" wsl.exe -d Ubuntu -e bash -c "cd /mnt/f/VisionRAG_Update/gui/v4 && python3 vision_rag_web.py"

timeout /t 4 /nobreak >nul

start chrome http://localhost:5555
