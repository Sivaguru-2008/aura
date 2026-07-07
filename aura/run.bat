@echo off
REM AURA — one-shot launcher (Windows).
cd /d "%~dp0"
echo [AURA] installing dependencies (first run may take a minute) ...
py -m pip install -q -r requirements.txt
echo [AURA] training models + running quantum-vs-classical benchmark ...
py -m aura_cli train
py -m aura_cli bench
echo [AURA] starting gateway + dashboard on http://127.0.0.1:8000
py -m aura_cli serve
