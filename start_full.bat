@echo off
cd /d D:\AI\OmniVoice-API
set HF_ENDPOINT=https://hf-mirror.com
echo Starting OmniVoice Full Server...
echo.
echo Web UI: http://localhost:8001
echo API: http://localhost:5050/v1
echo.
uv run python omnivoice_full_server.py
pause
