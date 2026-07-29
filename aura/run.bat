@echo off
REM AURA — one-shot launcher (Windows).
REM The repository root, not aura\, is the import root: every module is addressed as
REM aura.<...>, so -m has to be invoked from one level up.
cd /d "%~dp0.."
echo [AURA] installing dependencies (first run may take a minute) ...
py -m pip install -q -r aura\requirements.txt
echo [AURA] training models + running quantum-vs-classical benchmark ...
py -m aura.aura_cli train
py -m aura.aura_cli bench
echo [AURA] starting gateway + dashboard on http://127.0.0.1:8000
py -m aura.aura_cli serve
