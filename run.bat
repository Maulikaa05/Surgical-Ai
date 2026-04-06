@echo off
title SurgicalAI v3 — Arthroscopic Performance Evaluator
color 0A
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║  SurgicalAI v3 — AI Arthroscopic Performance Evaluator  ║
echo  ║  Ceaser's Medathon 2026  ^|  Problem Statement #1        ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  [1/2] Installing dependencies...
pip install -r requirements.txt --quiet
echo  [2/2] Launching Streamlit dashboard...
echo.
echo  Open your browser at: http://localhost:8501
echo.
streamlit run app.py --server.port 8501 --server.headless false
pause
