#!/bin/bash
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SurgicalAI v3 — AI Arthroscopic Performance Evaluator  ║"
echo "║  Ceaser's Medathon 2026  |  Problem Statement #1        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "[1/2] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "[2/2] Launching Streamlit dashboard..."
echo ""
echo "Open browser at: http://localhost:8501"
echo ""
streamlit run app.py --server.port 8501
