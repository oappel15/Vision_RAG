@echo off
REM Windows launcher for Vision RAG Migration Wizard (Web UI)
REM Double-click this file to start the wizard and open Chrome.
REM
REM FIRST-TIME SETUP (run once in a WSL terminal):
REM   bash /mnt/c/Users/oappe/projects/Vision_RAG_Git/migration_wizard/v4/setup-wsl-mounts.sh

echo ============================================================
echo  Vision RAG Migration Wizard v4.0 (Web UI)
echo ============================================================
echo.
echo  Starting server... Chrome will open in a moment.
echo  If it doesn't open, go to: http://localhost:5555
echo  Close this window to stop the server.
echo.

start "" wsl.exe -d Ubuntu -e bash -c "cd /mnt/c/Users/oappe/projects/Vision_RAG_Git/migration_wizard/v4 && python3 vision_rag_web.py"

echo  Waiting 8 seconds for server to start...
timeout /t 8 /nobreak >nul

start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" "http://localhost:5555"
REM explorer "http://localhost:5555"
