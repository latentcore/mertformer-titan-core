@echo off
REM Double-click to run the Nutrition5k vision side-experiment from inside
REM the real mertformer-titan-core repository (see scripts/train_nutrition5k.py
REM for the full write-up: purpose, data provenance, reuse boundary).
REM
REM Requires: Python 3.10+ with a CUDA-enabled torch already installed and
REM matching your GPU driver. This script never installs/upgrades torch.
cd /d "%~dp0"
python train_nutrition5k.py
pause
