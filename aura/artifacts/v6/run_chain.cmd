@echo off
set PYTHONPATH=E:\AURA\aura-main
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
cd /d E:\AURA\aura-main
set PY=E:\AURA\venv\Scripts\python.exe
set M=aura.ml.evaluation.v6_confirmatory
%PY% -u -m %M% c1 y_u0 >> aura\artifacts\v6\chain.log 2>&1
%PY% -u -m %M% c1 y_u1 >> aura\artifacts\v6\chain.log 2>&1
%PY% -u -m %M% c1b >> aura\artifacts\v6\chain.log 2>&1
%PY% -u -m %M% c2 classical >> aura\artifacts\v6\chain.log 2>&1
%PY% -u -m %M% c2 learnable >> aura\artifacts\v6\chain.log 2>&1
%PY% -u -m %M% c2 quantum >> aura\artifacts\v6\chain.log 2>&1
echo CHAIN DONE >> aura\artifacts\v6\chain.log
