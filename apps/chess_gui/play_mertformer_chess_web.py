#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
LOCAL_ONEFILE_PATH = BASE_DIR / "chess_5080_onefile.py"
CANONICAL_ONEFILE_PATH = REPO_ROOT / "scripts" / "chess_5080_onefile.py"
ONEFILE_PATH = LOCAL_ONEFILE_PATH if LOCAL_ONEFILE_PATH.exists() else CANONICAL_ONEFILE_PATH
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_by_val_loss.pt"
SUMMARY_PATH = BASE_DIR / "assets" / "run_summary.json"
LOGS_DIR = BASE_DIR / "logs"
SESSION_LOG_PATH = LOGS_DIR / "arena_session.jsonl"
BENCHMARK_HISTORY_PATH = LOGS_DIR / "benchmark_history.json"
DEFAULT_PORT = 8765
DEFAULT_STOCKFISH_ELO = 1100
DEFAULT_STOCKFISH_TIME_SEC = 0.18
DEFAULT_BENCHMARK_MAX_PLIES = 220
DEFAULT_BENCHMARK_VIS_DELAY_SEC = 0.95

PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}

HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MertFormer Chess Arena</title>
  <style>
    :root {
      --paper: #f4ecd7;
      --ink: #172018;
      --muted: #5a655a;
      --card: rgba(255, 252, 244, 0.78);
      --forest: #18392f;
      --brass: #b68137;
      --sand: #f1ddae;
      --sage: #5d745a;
      --target: rgba(24, 57, 47, 0.26);
      --select: #d85835;
      --check: rgba(178, 48, 48, 0.34);
      --last: rgba(182, 129, 55, 0.34);
      --shadow: 0 26px 80px rgba(35, 25, 8, 0.16);
      --radius: 26px;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 14% 16%, rgba(182,129,55,0.20), transparent 24%),
        radial-gradient(circle at 86% 0%, rgba(24,57,47,0.14), transparent 32%),
        linear-gradient(180deg, #f8f1df 0%, #efe4c8 48%, #e4d4af 100%);
    }
    .shell {
      width: min(1480px, calc(100vw - 28px));
      margin: 14px auto 20px;
      display: grid;
      grid-template-columns: minmax(400px, 780px) minmax(340px, 1fr);
      gap: 18px;
      align-items: start;
    }
    .card {
      background: var(--card);
      backdrop-filter: blur(18px);
      border: 1px solid rgba(23, 32, 24, 0.08);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .board-card {
      padding: 20px;
      position: sticky;
      top: 16px;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 8px;
      font-weight: 700;
    }
    .board-head {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }
    h1 {
      margin: 0;
      font-size: clamp(34px, 5vw, 56px);
      line-height: 0.92;
      letter-spacing: -0.03em;
    }
    .sub {
      margin-top: 10px;
      color: var(--muted);
      max-width: 42ch;
      font-size: 14px;
      line-height: 1.5;
    }
    .badge-stack {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .badge {
      padding: 10px 12px;
      border-radius: 999px;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.06em;
      background: rgba(24, 57, 47, 0.10);
      color: var(--forest);
      border: 1px solid rgba(24, 57, 47, 0.12);
      white-space: nowrap;
    }
    .board-wrap {
      padding: 18px;
      border-radius: 30px;
      background:
        linear-gradient(145deg, rgba(255,255,255,0.62), rgba(255,255,255,0.12)),
        radial-gradient(circle at 50% 0%, rgba(255,255,255,0.35), transparent 55%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.65), inset 0 -18px 50px rgba(17,38,31,0.07);
    }
    .board-grid {
      width: min(100%, 760px);
      margin: 0 auto;
      aspect-ratio: 1 / 1;
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      grid-template-rows: repeat(8, minmax(0, 1fr));
      grid-auto-columns: 1fr;
      grid-auto-rows: 1fr;
      border-radius: 22px;
      overflow: hidden;
      border: 1px solid rgba(23, 32, 24, 0.18);
      background: #d7c291;
    }
    .square {
      width: 100%;
      height: 100%;
      aspect-ratio: 1 / 1;
      min-width: 0;
      min-height: 0;
      border: 0;
      padding: 0;
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      cursor: pointer;
      transition: transform 0.08s ease, filter 0.12s ease;
      background: transparent;
      overflow: hidden;
      user-select: none;
    }
    .square:hover { filter: brightness(1.045); }
    .square:active { transform: scale(0.985); }
    .square.light { background: var(--sand); }
    .square.dark { background: var(--sage); }
    .piece {
      font-size: clamp(38px, 5vw, 68px);
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      transform: translateY(-1px);
      font-family: "Apple Symbols", "Segoe UI Symbol", "Noto Sans Symbols 2", "Arial Unicode MS", sans-serif;
      font-weight: 700;
      text-shadow: 0 2px 10px rgba(0,0,0,0.12);
      pointer-events: none;
    }
    .piece.white-piece {
      color: #fff7e8;
      text-shadow:
        -1px 0 rgba(22,32,25,0.92),
        0 1px rgba(22,32,25,0.92),
        1px 0 rgba(22,32,25,0.92),
        0 -1px rgba(22,32,25,0.92),
        0 2px 10px rgba(71,46,10,0.30);
    }
    .piece.black-piece {
      color: #162019;
      text-shadow:
        0 0 1px rgba(255,255,255,0.10),
        0 3px 10px rgba(0,0,0,0.12);
    }
    .square.selected::after,
    .square.last-move::after,
    .square.check::after,
    .square.legal-target::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    .square.selected::after {
      border: 4px solid var(--select);
    }
    .square.last-move::after {
      background: var(--last);
    }
    .square.check::after {
      background: var(--check);
    }
    .square.legal-target::before {
      content: "";
      position: absolute;
      width: 22%;
      aspect-ratio: 1;
      border-radius: 999px;
      background: var(--target);
      pointer-events: none;
    }
    .coord {
      position: absolute;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 11px;
      font-weight: 800;
      opacity: 0.88;
      pointer-events: none;
    }
    .coord.file { right: 7px; bottom: 5px; }
    .coord.rank { left: 7px; top: 5px; }
    .light .coord { color: rgba(24, 57, 47, 0.80); }
    .dark .coord { color: rgba(255, 247, 232, 0.86); }
    .controls {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .side-map {
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .side-pill {
      padding: 12px 14px;
      border-radius: 18px;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border: 1px solid rgba(23, 32, 24, 0.08);
      background: rgba(255,255,255,0.64);
    }
    .side-pill.white {
      background: linear-gradient(180deg, #fffaf1 0%, #f2e6c9 100%);
      color: #4b3713;
    }
    .side-pill.black {
      background: linear-gradient(180deg, #25372f 0%, #142019 100%);
      color: #f7f0df;
    }
    .control {
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      cursor: pointer;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      transition: transform 0.08s ease, opacity 0.15s ease;
    }
    .control:hover { transform: translateY(-1px); }
    .control.primary {
      background: var(--forest);
      color: #fdf9ef;
      box-shadow: 0 12px 28px rgba(24, 57, 47, 0.22);
    }
    .control.secondary {
      background: rgba(24, 57, 47, 0.08);
      color: var(--forest);
      border: 1px solid rgba(24, 57, 47, 0.12);
    }
    .control.warning {
      background: rgba(182, 129, 55, 0.16);
      color: #7d4d11;
      border: 1px solid rgba(182, 129, 55, 0.18);
    }
    .control.active {
      background: var(--forest);
      color: #fdf9ef;
      box-shadow: 0 12px 28px rgba(24, 57, 47, 0.18);
    }
    .control:disabled {
      opacity: 0.55;
      cursor: progress;
      transform: none;
    }
    .status-bar {
      margin-top: 14px;
      padding: 14px 16px;
      border-radius: 18px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      background: rgba(24,57,47,0.07);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }
    .status-main { font-weight: 800; }
    .status-sub { color: var(--muted); font-size: 13px; }
    .side {
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    .side section {
      padding: 16px;
      border-radius: 20px;
      background: rgba(255,255,255,0.44);
      border: 1px solid rgba(23,32,24,0.08);
    }
    .section-title {
      margin: 0 0 10px;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      padding: 12px;
      border-radius: 15px;
      background: rgba(255,255,255,0.74);
      min-height: 78px;
    }
    .metric-label {
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .metric-value {
      font-size: 28px;
      line-height: 1;
      font-weight: 800;
    }
    .metric-note {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      line-height: 1.35;
    }
    .stack, .list {
      display: grid;
      gap: 8px;
      max-height: 320px;
      overflow: auto;
      padding-right: 4px;
    }
    .row {
      display: grid;
      grid-template-columns: 44px 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-radius: 13px;
      background: rgba(255,255,255,0.74);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 13px;
    }
    .row .idx {
      color: var(--muted);
      font-weight: 800;
    }
    .row .meta {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-top: 3px;
    }
    .candidate-row {
      grid-template-columns: 32px 1fr auto;
    }
    .benchmark-panel {
      display: grid;
      gap: 12px;
    }
    .benchmark-summary {
      padding: 12px;
      border-radius: 15px;
      background: rgba(24,57,47,0.08);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }
    .mono {
      font-family: "SF Mono", "Menlo", monospace;
      font-size: 12px;
      word-break: break-word;
      color: #2d352f;
    }
    .mode-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .tag-cloud {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .tag-pill {
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(24, 57, 47, 0.08);
      border: 1px solid rgba(24, 57, 47, 0.10);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--forest);
    }
    .explanation-box {
      padding: 12px;
      border-radius: 15px;
      background: rgba(255,255,255,0.74);
      display: grid;
      gap: 8px;
    }
    .explanation-copy {
      font-size: 14px;
      line-height: 1.5;
      color: #263128;
    }
    .empty {
      color: var(--muted);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    .promo-modal {
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      background: rgba(18, 20, 15, 0.40);
      backdrop-filter: blur(10px);
      z-index: 50;
    }
    .promo-modal.open { display: grid; }
    .promo-card {
      width: min(360px, calc(100vw - 28px));
      padding: 18px;
      border-radius: 22px;
      background: rgba(255,252,244,0.96);
      box-shadow: var(--shadow);
    }
    .promo-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    .promo-option {
      border: 0;
      border-radius: 16px;
      padding: 16px 8px;
      background: rgba(24,57,47,0.08);
      cursor: pointer;
    }
    .promo-option .piece { font-size: 38px; }
    @media (max-width: 1120px) {
      .shell { grid-template-columns: 1fr; }
      .board-card { position: relative; top: 0; }
    }
    @media (max-width: 680px) {
      .shell { width: calc(100vw - 16px); gap: 12px; }
      .board-card, .side { padding: 14px; }
      .board-wrap { padding: 12px; }
      .controls { gap: 8px; }
      .control { flex: 1 1 calc(50% - 8px); text-align: center; }
      .metric-grid { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="card board-card">
      <div class="board-head">
        <div>
          <div class="eyebrow">Local Arena</div>
          <h1>MertFormer Chess</h1>
          <div class="sub">Your trained checkpoint, rendered as a clean local arena. The board is brighter, piece ownership is explicit, and the benchmark lane is wired straight into Stockfish 1100 with a slowed spectator feed.</div>
        </div>
        <div class="badge-stack">
          <div class="badge" id="deviceBadge">Device</div>
          <div class="badge" id="runBadge">Run</div>
          <div class="badge" id="viewBadge">Arena View</div>
          <div class="badge" id="claimBadge">No Claim</div>
        </div>
      </div>
      <div class="board-wrap">
        <div class="board-grid" id="board"></div>
      </div>
      <div class="controls">
        <button class="control primary" onclick="newGame('white')">Play White</button>
        <button class="control primary" onclick="newGame('black')">Play Black</button>
        <button class="control secondary" onclick="undoTurn()">Undo Turn</button>
        <button class="control secondary" onclick="flipBoard()">Flip Board</button>
        <button class="control secondary" onclick="copyFen()">Copy FEN</button>
        <button class="control secondary" onclick="switchView('arena')">Arena View</button>
        <button class="control secondary" onclick="switchView('benchmark')">Benchmark Theater</button>
      </div>
      <div class="side-map">
        <div class="side-pill white" id="whiteSide">White · Model</div>
        <div class="side-pill black" id="blackSide">Black · You</div>
      </div>
      <div class="status-bar">
        <div>
          <div class="status-main" id="statusMain">Loading arena…</div>
          <div class="status-sub" id="statusSub">Waiting for model bootstrap.</div>
        </div>
        <div class="status-sub" id="turnBadge">Turn</div>
      </div>
    </div>
    <div class="card side">
      <section>
        <h2 class="section-title">Checkpoint Snapshot</h2>
        <div class="metric-grid">
          <div class="metric"><div class="metric-label">Steps</div><div class="metric-value" id="metricSteps">-</div><div class="metric-note">Completed training steps</div></div>
          <div class="metric"><div class="metric-label">Holdout Acc</div><div class="metric-value" id="metricHoldout">-</div><div class="metric-note">Masked policy accuracy</div></div>
          <div class="metric"><div class="metric-label">Top-5</div><div class="metric-value" id="metricTop5">-</div><div class="metric-note">Holdout masked top-5</div></div>
          <div class="metric"><div class="metric-label">Positions</div><div class="metric-value" id="metricPositions">-</div><div class="metric-note">Training positions</div></div>
        </div>
      </section>
      <section>
        <h2 class="section-title">Model Trace</h2>
        <div class="metric-grid">
          <div class="metric"><div class="metric-label">Value</div><div class="metric-value" id="traceValue">-</div><div class="metric-note">Model evaluation head</div></div>
          <div class="metric"><div class="metric-label">Latency</div><div class="metric-value" id="traceLatency">-</div><div class="metric-note">Last inference</div></div>
        </div>
        <div style="margin-top:12px" class="stack" id="topkList"></div>
      </section>
      <section>
        <h2 class="section-title">Teaching Contract</h2>
        <div class="status-sub">Arena mode and Turkish explanation level both feed the same structured response contract.</div>
        <div class="mode-strip" id="modeStrip">
          <button class="control secondary" id="modePlay" onclick="setArenaMode('play')">Play</button>
          <button class="control secondary" id="modeTeach" onclick="setArenaMode('teach')">Teach</button>
          <button class="control secondary" id="modeAnalyze" onclick="setArenaMode('analyze')">Analyze</button>
          <button class="control secondary" id="modeTurkishTeach" onclick="setArenaMode('turkish_teach')">Turkish Teach</button>
        </div>
        <div class="mode-strip" id="levelStrip">
          <button class="control secondary" id="levelBasic" onclick="setTeachingLevel('basic')">Basic</button>
          <button class="control secondary" id="levelClub" onclick="setTeachingLevel('club')">Club</button>
          <button class="control secondary" id="levelAdvanced" onclick="setTeachingLevel('advanced')">Advanced</button>
        </div>
        <div class="metric-grid" style="margin-top:12px;">
          <div class="metric"><div class="metric-label">Best Move</div><div class="metric-value" id="contractBestMove">-</div><div class="metric-note" id="contractBestMoveSan">Structured best move</div></div>
          <div class="metric"><div class="metric-label">Confidence</div><div class="metric-value" id="contractConfidence">-</div><div class="metric-note" id="contractConfidenceNote">Confidence tier and top-gap</div></div>
        </div>
        <div class="explanation-box" style="margin-top:12px;">
          <div class="metric-label">Principal Variation</div>
          <div class="mono" id="contractPv">-</div>
          <div class="tag-cloud" id="contractTags"></div>
        </div>
        <div class="explanation-box" style="margin-top:12px;">
          <div class="metric-label">Short Turkish Explanation</div>
          <div class="explanation-copy" id="contractShort">No response contract yet.</div>
        </div>
        <div class="explanation-box" style="margin-top:12px;">
          <div class="metric-label">Long Turkish Explanation</div>
          <div class="explanation-copy" id="contractLong">Make a move, let the model answer, or switch to benchmark theater.</div>
        </div>
      </section>
      <section>
        <h2 class="section-title">Stockfish 1100 Probe</h2>
        <div class="benchmark-panel">
          <div class="benchmark-summary">
            <div style="font-weight:800; margin-bottom:4px;" id="benchStatusMain">Benchmark idle</div>
            <div style="color:var(--muted); font-size:13px; line-height:1.45;" id="benchStatusSub">Model will play White. Stockfish will run with local 1100-strength settings.</div>
          </div>
          <div class="controls" style="margin-top:0;">
            <button class="control warning" id="benchBtn" onclick="startBenchmark()">Run White vs Stockfish 1100</button>
          </div>
          <div class="metric-grid">
            <div class="metric"><div class="metric-label">Result</div><div class="metric-value" id="benchResult">-</div><div class="metric-note">Latest benchmark score</div></div>
            <div class="metric"><div class="metric-label">Plies</div><div class="metric-value" id="benchPlies">-</div><div class="metric-note">Moves logged</div></div>
          </div>
          <div class="mono" id="benchLogPath">No benchmark log yet.</div>
          <div class="stack" id="benchHistory"></div>
        </div>
      </section>
      <section>
        <h2 class="section-title">Move Ledger</h2>
        <div class="list" id="moveList"></div>
      </section>
    </div>
  </div>
  <div class="promo-modal" id="promoModal">
    <div class="promo-card">
      <div class="eyebrow">Promotion</div>
      <div style="font-size:24px; font-weight:800;">Choose the promotion piece</div>
      <div class="promo-grid" id="promoGrid"></div>
      <div class="status-sub" style="margin-top:10px;">Close this modal if you want to cancel the move.</div>
    </div>
  </div>
  <script>
    const pieceMap = {P:'♙',N:'♘',B:'♗',R:'♖',Q:'♕',K:'♔',p:'♟',n:'♞',b:'♝',r:'♜',q:'♛',k:'♚'};
    let appState = null;
    let selectedSquare = null;
    let flipped = false;
    let refreshTimer = null;
    let viewMode = 'arena';

    function refreshDelayMs() {
      return viewMode === 'benchmark' ? 900 : 2400;
    }

    function scheduleRefresh() {
      if (refreshTimer) {
        clearTimeout(refreshTimer);
      }
      refreshTimer = setTimeout(async () => {
        await refreshState().catch(() => {});
        scheduleRefresh();
      }, refreshDelayMs());
    }

    function pct(v) {
      return `${(v * 100).toFixed(1)}%`;
    }

    function preserveScroll(fn) {
      const y = window.scrollY;
      fn();
      requestAnimationFrame(() => window.scrollTo(0, y));
    }

    function activeViewState() {
      if (viewMode === 'benchmark') {
        return appState?.benchmark_live || null;
      }
      return appState;
    }

    function activeViewHumanColor() {
      if (viewMode === 'benchmark') return 'white';
      return appState?.human_color || 'white';
    }

    function orientSquares(humanColor) {
      const files = ['a','b','c','d','e','f','g','h'];
      const ranks = ['1','2','3','4','5','6','7','8'];
      const whiteBottom = flipped ? humanColor !== 'white' : humanColor === 'white';
      const fileOrder = whiteBottom ? files : [...files].reverse();
      const rankOrder = whiteBottom ? [...ranks].reverse() : ranks;
      const squares = [];
      for (const rank of rankOrder) {
        for (const file of fileOrder) {
          squares.push(`${file}${rank}`);
        }
      }
      return { squares, whiteBottom };
    }

    function boardMapFromFen(fen) {
      const placement = fen.split(' ')[0];
      const rows = placement.split('/');
      const files = ['a','b','c','d','e','f','g','h'];
      const board = {};
      rows.forEach((row, rowIdx) => {
        let fileIdx = 0;
        for (const ch of row) {
          if (/\d/.test(ch)) {
            fileIdx += Number(ch);
          } else {
            board[`${files[fileIdx]}${8 - rowIdx}`] = ch;
            fileIdx += 1;
          }
        }
      });
      return board;
    }

    function legalTargets(square) {
      if (!appState) return [];
      return appState.legal_targets[square] || [];
    }

    function promotionChoices(prefix) {
      if (!appState) return [];
      return appState.legal_groups[prefix] || [];
    }

    function isPlayersTurn() {
      return viewMode === 'arena' && appState && !appState.game_over && appState.turn === appState.human_color;
    }

    function createPieceNode(pieceChar) {
      const span = document.createElement('span');
      span.className = `piece ${pieceChar === pieceChar.toUpperCase() ? 'white-piece' : 'black-piece'}`;
      span.textContent = pieceMap[pieceChar] || '';
      span.setAttribute('aria-hidden', 'true');
      return span;
    }

    function renderBoard() {
      const state = activeViewState();
      if (!state) return;
      const boardEl = document.getElementById('board');
      boardEl.innerHTML = '';
      const boardMap = boardMapFromFen(state.fen);
      const { squares, whiteBottom } = orientSquares(activeViewHumanColor());
      const lastMove = state.last_move || {};
      const checkSquare = state.check_square;
      const targets = new Set(viewMode === 'arena' && selectedSquare ? legalTargets(selectedSquare) : []);
      for (const square of squares) {
        const file = square[0];
        const rank = square[1];
        const isLight = (file.charCodeAt(0) - 97 + Number(rank)) % 2 === 1;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `square ${isLight ? 'light' : 'dark'}`;
        button.dataset.square = square;
        if (viewMode === 'arena' && selectedSquare === square) button.classList.add('selected');
        if (square === lastMove.from || square === lastMove.to) button.classList.add('last-move');
        if (square === checkSquare) button.classList.add('check');
        if (targets.has(square)) button.classList.add('legal-target');
        if (boardMap[square]) {
          button.appendChild(createPieceNode(boardMap[square]));
        }
        button.addEventListener('click', () => onSquareClick(square, boardMap[square] || null));

        if (whiteBottom ? rank === '1' : rank === '8') {
          const coord = document.createElement('span');
          coord.className = 'coord file';
          coord.textContent = file;
          button.appendChild(coord);
        }
        if (whiteBottom ? file === 'a' : file === 'h') {
          const coord = document.createElement('span');
          coord.className = 'coord rank';
          coord.textContent = rank;
          button.appendChild(coord);
        }
        boardEl.appendChild(button);
      }
    }

    function renderLists() {
      const topkList = document.getElementById('topkList');
      topkList.innerHTML = '';
      const traceSource = viewMode === 'benchmark' ? appState?.benchmark_live?.last_trace : appState?.last_trace;
      const candidates = traceSource?.masked_topk || [];
      const candidateScores = traceSource?.masked_topk_scores || [];
      if (!candidates.length) {
        topkList.innerHTML = viewMode === 'benchmark'
          ? '<div class="empty">Benchmark theater is waiting for the next model move.</div>'
          : '<div class="empty">No model trace yet. Make a move or let the model open as Black.</div>';
      } else {
        candidates.forEach((cand, idx) => {
          const row = document.createElement('div');
          row.className = 'row candidate-row';
          const score = Number.isFinite(candidateScores[idx]) ? candidateScores[idx].toFixed(3) : '';
          row.innerHTML = `<div class="idx">#${idx + 1}</div><div>${cand}</div><div>${idx === 0 ? 'played' : score}</div>`;
          topkList.appendChild(row);
        });
      }

      const moveList = document.getElementById('moveList');
      moveList.innerHTML = '';
      const transcript = viewMode === 'benchmark' ? (appState?.benchmark_live?.transcript || []) : (appState?.transcript || []);
      if (!transcript.length) {
        moveList.innerHTML = viewMode === 'benchmark'
          ? '<div class="empty">Benchmark theater is armed. Press the Stockfish 1100 button to watch the game unfold here.</div>'
          : '<div class="empty">Fresh board. Start a game as White or Black.</div>';
      } else {
        transcript.forEach((item) => {
          const row = document.createElement('div');
          row.className = 'row';
          const shortExplain = item.response_contract?.explanation_tr_short || '';
          row.innerHTML = `<div class="idx">${item.ply}</div><div><div>${item.san} <span style="color:var(--muted)">(${item.move})</span></div><div class="meta">${item.actor}${shortExplain ? ` · ${shortExplain}` : ''}</div></div><div>${item.actor === 'model' && item.value !== null ? item.value.toFixed(2) : ''}</div>`;
          moveList.appendChild(row);
        });
      }
    }

    function renderTeachingContract() {
      const traceSource = viewMode === 'benchmark' ? appState?.benchmark_live?.last_trace : appState?.last_trace;
      const contract = traceSource?.response_contract || null;
      const bestMove = document.getElementById('contractBestMove');
      const bestMoveSan = document.getElementById('contractBestMoveSan');
      const confidence = document.getElementById('contractConfidence');
      const confidenceNote = document.getElementById('contractConfidenceNote');
      const pv = document.getElementById('contractPv');
      const tags = document.getElementById('contractTags');
      const shortBox = document.getElementById('contractShort');
      const longBox = document.getElementById('contractLong');
      tags.innerHTML = '';

      const modeButtons = {
        play: document.getElementById('modePlay'),
        teach: document.getElementById('modeTeach'),
        analyze: document.getElementById('modeAnalyze'),
        turkish_teach: document.getElementById('modeTurkishTeach'),
      };
      Object.entries(modeButtons).forEach(([name, button]) => {
        button.classList.toggle('active', appState?.ui_mode === name);
        button.classList.toggle('secondary', appState?.ui_mode !== name);
      });
      const levelButtons = {
        basic: document.getElementById('levelBasic'),
        club: document.getElementById('levelClub'),
        advanced: document.getElementById('levelAdvanced'),
      };
      Object.entries(levelButtons).forEach(([name, button]) => {
        button.classList.toggle('active', appState?.teaching_level === name);
        button.classList.toggle('secondary', appState?.teaching_level !== name);
      });

      if (!contract) {
        bestMove.textContent = '-';
        bestMoveSan.textContent = 'Structured best move';
        confidence.textContent = '-';
        confidenceNote.textContent = 'Confidence tier and top-gap';
        pv.textContent = '-';
        shortBox.textContent = 'No response contract yet.';
        longBox.textContent = 'Make a move, let the model answer, or switch to benchmark theater.';
        return;
      }

      bestMove.textContent = contract.best_move || '-';
      bestMoveSan.textContent = contract.best_move_san || 'Structured best move';
      confidence.textContent = contract.confidence?.tier || '-';
      const confScore = Number(contract.confidence?.score ?? NaN);
      const confGap = Number(contract.confidence?.gap ?? NaN);
      confidenceNote.textContent = contract.confidence ? `score ${Number.isFinite(confScore) ? confScore.toFixed(3) : '-'} · gap ${Number.isFinite(confGap) ? confGap.toFixed(3) : '-'}` : 'Confidence tier and top-gap';
      pv.textContent = (contract.principal_variation || []).join(' → ') || '-';
      (contract.teaching_tags || []).forEach((tag) => {
        const pill = document.createElement('span');
        pill.className = 'tag-pill';
        pill.textContent = String(tag).replaceAll('_', ' ');
        tags.appendChild(pill);
      });
      shortBox.textContent = contract.explanation_tr_short || 'No short Turkish explanation.';
      longBox.textContent = contract.explanation_tr_long || 'No long Turkish explanation.';
    }

    function renderBenchmark() {
      const bench = appState?.benchmark || {};
      document.getElementById('benchStatusMain').textContent = bench.status_main || 'Benchmark idle';
      document.getElementById('benchStatusSub').textContent = bench.status_sub || 'Stockfish 1100 benchmark ready.';
      document.getElementById('benchResult').textContent = bench.latest_result || '-';
      document.getElementById('benchPlies').textContent = bench.latest_plies || '-';
      document.getElementById('benchLogPath').textContent = bench.log_path || 'No benchmark log yet.';
      document.getElementById('benchBtn').disabled = !!bench.running;
      const benchHistory = document.getElementById('benchHistory');
      benchHistory.innerHTML = '';
      const history = appState?.benchmark_history || [];
      if (!history.length) {
        benchHistory.innerHTML = '<div class="empty">No benchmark history yet. Run the 1100 probe and the JSON, Markdown, PGN, and text logs will appear here.</div>';
      } else {
        history.forEach((item) => {
          const row = document.createElement('div');
          row.className = 'row';
          row.innerHTML = `<div class="idx">${item.result}</div><div><div>${item.benchmark_id}</div><div class="meta">${item.winner} · ${item.plies} plies</div></div><div>${item.duration_sec.toFixed(1)}s</div>`;
          benchHistory.appendChild(row);
        });
      }
    }

    function renderMetrics() {
      if (!appState) return;
      const bench = appState?.benchmark || {};
      document.getElementById('deviceBadge').textContent = `Device ${String(appState.device).toUpperCase()}`;
      document.getElementById('runBadge').textContent = `Run ${appState.run_id}`;
      const activeArenaMode = appState?.ui_mode || 'play';
      document.getElementById('viewBadge').textContent = viewMode === 'benchmark' ? 'Benchmark Theater' : `Arena · ${activeArenaMode.replaceAll('_', ' ')}`;
      document.getElementById('claimBadge').textContent = (appState.rating_claim_status || 'no_claim').replaceAll('_', ' ');
      document.getElementById('metricSteps').textContent = `${appState.summary.steps_completed}`;
      document.getElementById('metricHoldout').textContent = pct(appState.summary.holdout_masked_accuracy);
      document.getElementById('metricTop5').textContent = pct(appState.summary.holdout_masked_top5);
      document.getElementById('metricPositions').textContent = `${Math.round(appState.summary.positions_total / 1000)}k`;
      const activeTrace = viewMode === 'benchmark' ? appState?.benchmark_live?.last_trace : appState.last_trace;
      const activeGameOver = viewMode === 'benchmark' ? appState?.benchmark_live?.game_over : appState.game_over;
      const activeTurn = viewMode === 'benchmark' ? appState?.benchmark_live?.turn : appState.turn;
      document.getElementById('traceValue').textContent = activeTrace ? activeTrace.value.toFixed(3) : '-';
      document.getElementById('traceLatency').textContent = activeTrace ? `${activeTrace.latency_ms.toFixed(1)}ms` : '-';
      document.getElementById('turnBadge').textContent = activeGameOver ? 'Game Finished' : `${activeTurn === 'white' ? 'White' : 'Black'} to move`;
      document.getElementById('statusMain').textContent = viewMode === 'benchmark' ? (bench.status_main || 'Benchmark theater') : appState.status_main;
      document.getElementById('statusSub').textContent = viewMode === 'benchmark' ? (bench.status_sub || 'Watching benchmark.') : appState.status_sub;
      const whiteRole = viewMode === 'benchmark' ? 'Model' : (appState.human_color === 'white' ? 'You' : 'Model');
      const blackRole = viewMode === 'benchmark' ? 'Stockfish 1100' : (appState.human_color === 'black' ? 'You' : 'Model');
      document.getElementById('whiteSide').textContent = `White · ${whiteRole}`;
      document.getElementById('blackSide').textContent = `Black · ${blackRole}`;
      renderLists();
      renderTeachingContract();
      renderBenchmark();
    }

    function switchView(mode) {
      viewMode = mode === 'benchmark' ? 'benchmark' : 'arena';
      scheduleRefresh();
      preserveScroll(() => {
        renderBoard();
        renderMetrics();
      });
    }

    async function api(path, payload) {
      const res = await fetch(path, {
        method: payload ? 'POST' : 'GET',
        headers: { 'Content-Type': 'application/json' },
        body: payload ? JSON.stringify(payload) : undefined,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      return res.json();
    }

    async function refreshState() {
      appState = await api('/api/state');
      if ((appState?.benchmark?.running || appState?.benchmark_live?.transcript?.length) && viewMode !== 'benchmark' && !appState.transcript?.length) {
        viewMode = 'benchmark';
      }
      preserveScroll(() => {
        renderBoard();
        renderMetrics();
      });
    }

    async function setArenaMode(mode) {
      appState = await api('/api/mode', { mode });
      preserveScroll(() => renderMetrics());
    }

    async function setTeachingLevel(teachingLevel) {
      appState = await api('/api/mode', { teaching_level: teachingLevel });
      preserveScroll(() => renderMetrics());
    }

    function clearSelection() {
      selectedSquare = null;
      preserveScroll(() => renderBoard());
    }

    async function onSquareClick(square, piece) {
      if (!isPlayersTurn()) return;
      const isOwnPiece = piece && ((appState.human_color === 'white' && piece === piece.toUpperCase()) || (appState.human_color === 'black' && piece === piece.toLowerCase()));
      if (!selectedSquare) {
        if (!isOwnPiece) return;
        selectedSquare = square;
        renderBoard();
        return;
      }
      if (selectedSquare === square) {
        clearSelection();
        return;
      }
      if (isOwnPiece) {
        selectedSquare = square;
        renderBoard();
        return;
      }
      const prefix = `${selectedSquare}${square}`;
      const candidates = promotionChoices(prefix);
      if (!candidates.length) return;
      if (candidates.length === 1) {
        await sendMove(candidates[0]);
        return;
      }
      openPromotionModal(candidates);
    }

    async function sendMove(move) {
      try {
        appState = await api('/api/move', { move });
        selectedSquare = null;
        closePromotionModal();
        preserveScroll(() => {
          renderBoard();
          renderMetrics();
        });
      } catch (err) {
        alert(`Move rejected: ${err.message}`);
      }
    }

    async function newGame(color) {
      appState = await api('/api/new_game', { human_color: color });
      selectedSquare = null;
      viewMode = 'arena';
      scheduleRefresh();
      preserveScroll(() => {
        renderBoard();
        renderMetrics();
      });
    }

    async function undoTurn() {
      appState = await api('/api/undo', {});
      selectedSquare = null;
      preserveScroll(() => {
        renderBoard();
        renderMetrics();
      });
    }

    function flipBoard() {
      flipped = !flipped;
      preserveScroll(() => renderBoard());
    }

    async function copyFen() {
      if (!appState) return;
      try {
        await navigator.clipboard.writeText(appState.fen);
        document.getElementById('statusSub').textContent = 'FEN copied to clipboard.';
      } catch {
        alert(appState.fen);
      }
    }

    async function startBenchmark() {
      try {
        appState = await api('/api/benchmark/start', {});
        viewMode = 'benchmark';
        scheduleRefresh();
        preserveScroll(() => {
          renderBoard();
          renderMetrics();
        });
      } catch (err) {
        alert(`Benchmark failed to start: ${err.message}`);
      }
    }

    function openPromotionModal(candidates) {
      const modal = document.getElementById('promoModal');
      const grid = document.getElementById('promoGrid');
      grid.innerHTML = '';
      candidates.forEach((move) => {
        const btn = document.createElement('button');
        btn.className = 'promo-option';
        const promo = move.slice(-1);
        const piece = appState.human_color === 'white' ? promo.toUpperCase() : promo;
        btn.appendChild(createPieceNode(piece));
        btn.onclick = () => sendMove(move);
        grid.appendChild(btn);
      });
      modal.classList.add('open');
    }

    function closePromotionModal() {
      document.getElementById('promoModal').classList.remove('open');
    }

    document.getElementById('promoModal').addEventListener('click', (event) => {
      if (event.target.id === 'promoModal') closePromotionModal();
    });

    refreshState().catch((err) => {
      document.getElementById('statusMain').textContent = 'Arena bootstrap failed';
      document.getElementById('statusSub').textContent = err.message;
    });
    scheduleRefresh();
  </script>
</body>
</html>
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("mertformer_chess_onefile_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_device(onefile: Any, preferred: Optional[str]) -> str:
    if preferred:
        return preferred
    torch = onefile.torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_stockfish_path() -> Optional[str]:
    for candidate in (
        shutil.which("stockfish"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


GUI_RESPONSE_MODES = {"play", "teach", "analyze", "turkish_teach", "benchmark"}
GUI_TEACHING_LEVELS = {"basic", "club", "advanced"}


def normalize_gui_mode(onefile: Any, mode: str) -> str:
    if hasattr(onefile, "normalize_chess_response_mode"):
        return str(onefile.normalize_chess_response_mode(mode))
    mode_norm = str(mode or "play").strip().lower()
    return mode_norm if mode_norm in GUI_RESPONSE_MODES else "play"


def normalize_gui_teaching_level(onefile: Any, level: str) -> str:
    if hasattr(onefile, "normalize_teaching_level"):
        return str(onefile.normalize_teaching_level(level))
    level_norm = str(level or "club").strip().lower()
    return level_norm if level_norm in GUI_TEACHING_LEVELS else "club"


def _fallback_teaching_tags(onefile: Any, board: Any, move: Any) -> List[str]:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return ["positional_choice"]
    tags: List[str] = []
    if board.is_capture(move):
        tags.append("capture")
    if board.is_castling(move):
        tags.append("castle")
    if getattr(move, "promotion", None):
        tags.append("promotion")
    if hasattr(board, "gives_check") and board.gives_check(move):
        tags.append("check")
    target = onefile.chess.square_name(move.to_square)
    source = onefile.chess.square_name(move.from_square)
    if piece.piece_type == onefile.chess.PAWN and target in {"c4", "d4", "e4", "f4", "c5", "d5", "e5", "f5"}:
        tags.append("center_control")
    if piece.piece_type in {onefile.chess.KNIGHT, onefile.chess.BISHOP} and source in {"b1", "g1", "c1", "f1", "b8", "g8", "c8", "f8"}:
        tags.append("development")
    if not tags:
        tags.append("positional_choice")
    return list(dict.fromkeys(tags))


def build_gui_fallback_response_contract(
    onefile: Any,
    board: Any,
    trace: Dict[str, Any],
    *,
    mode: str,
    teaching_level: str,
) -> Dict[str, Any]:
    move = onefile.chess.Move.from_uci(trace["move"])
    san = board.san(move)
    value = float(trace.get("value", 0.0))
    tags = _fallback_teaching_tags(onefile, board, move)
    if value >= 0.25:
        eval_phrase = "hafif artı bölgede"
    elif value <= -0.25:
        eval_phrase = "hafif eksi bölgede"
    else:
        eval_phrase = "dengeye yakın"
    confidence = dict(trace.get("confidence") or {"score": 0.0, "gap": 0.0, "tier": "low"})
    short = f"{mode.replace('_', ' ')} modu: {san} hamlesi {', '.join(tags)} fikrini taşıyor. Konum {eval_phrase}."
    return {
        "contract_version": "gui-fallback-1.0",
        "best_move": trace["move"],
        "best_move_san": san,
        "evaluation": {"value": round(value, 4), "label": eval_phrase, "perspective": "side_to_move"},
        "principal_variation": [trace["move"]],
        "confidence": confidence,
        "teaching_tags": tags,
        "explanation_tr_short": short,
        "explanation_tr_long": short + " Bu varyant search derinliği değil, tek hamlelik policy özetidir.",
        "mode": mode,
        "teaching_level": teaching_level,
    }


@dataclass
class ArenaSummary:
    run_id: str
    rating_claim_status: str
    steps_completed: int
    holdout_masked_accuracy: float
    holdout_masked_top5: float
    positions_total: int
    best_val_loss: float


@dataclass
class BenchmarkSnapshot:
    running: bool = False
    status_main: str = "Benchmark idle"
    status_sub: str = "Model will play White. Stockfish will run with local 1100-strength settings."
    log_path: str = ""
    pgn_path: str = ""
    latest_result: str = "-"
    latest_plies: str = "-"
    latest_report: Optional[Dict[str, Any]] = None
    live_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    live_turn: str = "white"
    live_last_move: Optional[Dict[str, str]] = None
    live_check_square: Optional[str] = None
    live_transcript: List[Dict[str, Any]] = field(default_factory=list)
    live_last_trace: Optional[Dict[str, Any]] = None
    live_game_over: bool = False
    live_result: str = "*"
    live_winner: Optional[str] = None


@dataclass
class ArenaState:
    onefile: Any
    model: Any
    device: Any
    summary: ArenaSummary
    human_color: bool
    ui_mode: str = "play"
    teaching_level: str = "club"
    board: Any = field(default=None)
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    last_trace: Optional[Dict[str, Any]] = None
    last_move: Optional[Dict[str, str]] = None
    status_main: str = "Ready"
    status_sub: str = "Choose a side and make the first move."
    lock: threading.Lock = field(default_factory=threading.Lock)
    model_lock: threading.Lock = field(default_factory=threading.Lock)
    benchmark: BenchmarkSnapshot = field(default_factory=BenchmarkSnapshot)
    benchmark_history: List[Dict[str, Any]] = field(default_factory=list)
    stockfish_path: Optional[str] = None
    fast_arena_cfg: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.board = self.onefile.chess.Board()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def _log_event(self, event: str, **payload: Any) -> None:
        append_jsonl(
            SESSION_LOG_PATH,
            {
                "ts_utc": utc_now(),
                "event": event,
                "run_id": self.summary.run_id,
                **payload,
            },
        )

    def _actor_color_name(self, actor_color: bool) -> str:
        return "white" if actor_color == self.onefile.chess.WHITE else "black"

    def _push_with_record(self, move: Any, actor: str, trace: Optional[Dict[str, Any]] = None) -> None:
        san = self.board.san(move)
        from_sq = move.uci()[:2]
        to_sq = move.uci()[2:4]
        self.board.push(move)
        self.last_move = {"from": from_sq, "to": to_sq}
        self.transcript.append(
            {
                "ply": len(self.transcript) + 1,
                "actor": actor,
                "move": move.uci(),
                "san": san,
                "fen": self.board.fen(),
                "value": None if trace is None else float(trace.get("value", 0.0)),
                "latency_ms": None if trace is None else float(trace.get("latency_ms", 0.0)),
                "mode": None if trace is None else trace.get("response_contract", {}).get("mode"),
                "response_contract": None if trace is None else trace.get("response_contract"),
            }
        )

    def _check_square(self) -> Optional[str]:
        if not self.board.is_check():
            return None
        king_square = self.board.king(self.board.turn)
        if king_square is None:
            return None
        return self.onefile.chess.square_name(king_square)

    def _check_square_for_board(self, board: Any) -> Optional[str]:
        if not board.is_check():
            return None
        king_square = board.king(board.turn)
        if king_square is None:
            return None
        return self.onefile.chess.square_name(king_square)

    def _legal_maps(self) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        targets: Dict[str, List[str]] = {}
        groups: Dict[str, List[str]] = {}
        for move in self.board.legal_moves:
            uci = move.uci()
            src, dst = uci[:2], uci[2:4]
            targets.setdefault(src, []).append(dst)
            groups.setdefault(src + dst, []).append(uci)
        return targets, groups

    def _result_payload(self) -> Dict[str, Any]:
        outcome = self.board.outcome()
        if outcome is None:
            return {"game_over": False, "result": "*", "winner": None, "termination": None}
        winner = (
            "white"
            if outcome.winner is self.onefile.chess.WHITE
            else "black"
            if outcome.winner is self.onefile.chess.BLACK
            else "draw"
        )
        return {
            "game_over": True,
            "result": outcome.result(),
            "winner": winner,
            "termination": str(outcome.termination),
        }

    def _result_payload_for_board(self, board: Any) -> Dict[str, Any]:
        outcome = board.outcome()
        if outcome is None:
            return {"game_over": False, "result": "*", "winner": None, "termination": None}
        winner = (
            "white"
            if outcome.winner is self.onefile.chess.WHITE
            else "black"
            if outcome.winner is self.onefile.chess.BLACK
            else "draw"
        )
        return {
            "game_over": True,
            "result": outcome.result(),
            "winner": winner,
            "termination": str(outcome.termination),
        }

    def _set_benchmark_live_state(
        self,
        board: Any,
        transcript: List[Dict[str, Any]],
        last_move: Optional[Dict[str, str]],
        last_trace: Optional[Dict[str, Any]],
    ) -> None:
        result = self._result_payload_for_board(board)
        self.benchmark.live_fen = board.fen()
        self.benchmark.live_turn = self._actor_color_name(board.turn)
        self.benchmark.live_last_move = last_move
        self.benchmark.live_check_square = self._check_square_for_board(board)
        self.benchmark.live_transcript = list(transcript)
        self.benchmark.live_last_trace = last_trace
        self.benchmark.live_game_over = result["game_over"]
        self.benchmark.live_result = result["result"]
        self.benchmark.live_winner = result["winner"]

    def _model_trace(self, board: Any, *, mode: Optional[str] = None) -> Dict[str, Any]:
        active_mode = normalize_gui_mode(self.onefile, mode or self.ui_mode)
        teaching_level = normalize_gui_teaching_level(self.onefile, self.teaching_level)
        trace_cfg: Dict[str, Any] = {}
        if active_mode != "benchmark":
            trace_cfg.update(self.fast_arena_cfg)
        with self.model_lock:
            try:
                trace = self.onefile.choose_move_trace(
                    self.model,
                    board,
                    self.device,
                    cfg=trace_cfg or None,
                    mode=active_mode,
                    teaching_level=teaching_level,
                )
            except TypeError:
                # Backward-compatible retry for onefile variants that do not accept `cfg`
                # but still support explicit mode and teaching-level control.
                try:
                    trace = self.onefile.choose_move_trace(
                        self.model,
                        board,
                        self.device,
                        mode=active_mode,
                        teaching_level=teaching_level,
                    )
                except TypeError:
                    trace = self.onefile.choose_move_trace(self.model, board, self.device)
            trace.setdefault("raw_topk_scores", [])
            trace.setdefault("masked_topk_scores", [])
            trace.setdefault("confidence", {"score": 0.0, "gap": 0.0, "tier": "low"})
            if "response_contract" not in trace:
                if hasattr(self.onefile, "build_chess_response_contract"):
                    trace["response_contract"] = self.onefile.build_chess_response_contract(
                        board,
                        trace,
                        mode=active_mode,
                        teaching_level=teaching_level,
                    )
                else:
                    trace["response_contract"] = build_gui_fallback_response_contract(
                        self.onefile,
                        board,
                        trace,
                        mode=active_mode,
                        teaching_level=teaching_level,
                    )
            return trace

    def _apply_model_move_if_needed(self) -> None:
        if self.board.is_game_over():
            result = self._result_payload()
            self.status_main = f"Game over: {result['result']}"
            self.status_sub = result["termination"] or "Finished"
            self._log_event("game_over", result=result["result"], winner=result["winner"], termination=result["termination"])
            return
        if self.board.turn == self.human_color:
            self.status_main = "Your move"
            self.status_sub = "Select a piece, then click a legal destination."
            return
        trace = self._model_trace(self.board)
        move = self.onefile.chess.Move.from_uci(trace["move"])
        self._push_with_record(move, "model", trace)
        self.last_trace = trace
        self._log_event(
            "model_move",
            move=move.uci(),
            value=float(trace["value"]),
            latency_ms=float(trace["latency_ms"]),
            fen=self.board.fen(),
        )
        if self.board.is_game_over():
            result = self._result_payload()
            self.status_main = f"Game over: {result['result']}"
            self.status_sub = result["termination"] or "Finished"
            self._log_event("game_over", result=result["result"], winner=result["winner"], termination=result["termination"])
        else:
            self.status_main = f"Model played {move.uci()}"
            contract = trace.get("response_contract", {})
            self.status_sub = str(contract.get("explanation_tr_short") or f"Value {trace['value']:.3f} · Latency {trace['latency_ms']:.1f} ms")

    def new_game(self, human_color_name: str) -> Dict[str, Any]:
        with self.lock:
            self.human_color = self.onefile.chess.WHITE if human_color_name == "white" else self.onefile.chess.BLACK
            self.board = self.onefile.chess.Board()
            self.transcript = []
            self.last_trace = None
            self.last_move = None
            self.status_main = "New game"
            self.status_sub = "Arena reset."
            self._log_event("new_game", human_color=human_color_name)
            self._apply_model_move_if_needed()
            return self.state_payload()

    def play_human_move(self, move_uci: str) -> Dict[str, Any]:
        with self.lock:
            if self.board.is_game_over():
                raise ValueError("Game already finished. Start a new game.")
            if self.board.turn != self.human_color:
                raise ValueError("It is not the human side's turn.")
            move = self.onefile.chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                raise ValueError(f"Illegal move: {move_uci}")
            self._push_with_record(move, "human")
            self._log_event("human_move", move=move_uci, human_color=self._actor_color_name(self.human_color), fen=self.board.fen())
            self._apply_model_move_if_needed()
            return self.state_payload()

    def undo_full_turn(self) -> Dict[str, Any]:
        with self.lock:
            if not self.transcript:
                return self.state_payload()
            if self.transcript and self.transcript[-1]["actor"] == "model":
                self.board.pop()
                self.transcript.pop()
            if self.transcript and self.transcript[-1]["actor"] == "human":
                self.board.pop()
                self.transcript.pop()
            self.last_trace = None
            if self.transcript:
                last = self.transcript[-1]
                self.last_move = {"from": last["move"][:2], "to": last["move"][2:4]}
            else:
                self.last_move = None
            self.status_main = "Move undone"
            self.status_sub = "Last full turn removed."
            self._log_event("undo_turn", transcript_len=len(self.transcript), fen=self.board.fen())
            return self.state_payload()

    def _serialize_engine_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, value in info.items():
            name = str(key)
            if name == "score":
                try:
                    payload["score"] = str(value)
                except Exception as e:
                    print(f"[warn] engine info 'score' serialize failed: {type(e).__name__}: {e}", file=sys.stderr)
                    payload["score"] = repr(value)
            elif name == "pv":
                try:
                    payload["pv"] = [move.uci() for move in value]
                except (AttributeError, TypeError) as e:
                    print(f"[warn] engine info 'pv' serialize failed: {type(e).__name__}: {e}", file=sys.stderr)
                    payload["pv"] = repr(value)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                payload[name] = value
            else:
                payload[name] = repr(value)
        return payload

    def _configure_engine(self, engine: Any) -> Dict[str, Any]:
        options = getattr(engine, "options", {})
        applied: Dict[str, Any] = {}
        if "Threads" in options:
            applied["Threads"] = 1
        if "Hash" in options:
            applied["Hash"] = 64
        elo_min = None
        if "UCI_Elo" in options:
            elo_min = getattr(options["UCI_Elo"], "min", None)
        if elo_min is not None and DEFAULT_STOCKFISH_ELO < elo_min:
            if "Skill Level" in options:
                applied["Skill Level"] = 1
        else:
            if "UCI_LimitStrength" in options:
                applied["UCI_LimitStrength"] = True
            if "UCI_Elo" in options:
                applied["UCI_Elo"] = DEFAULT_STOCKFISH_ELO
        if applied:
            engine.configure(applied)
        return applied

    def _write_benchmark_outputs(self, benchmark_id: str, report: Dict[str, Any]) -> Dict[str, str]:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = LOGS_DIR / f"stockfish_1100_{benchmark_id}.json"
        md_path = LOGS_DIR / f"stockfish_1100_{benchmark_id}.md"
        pgn_path = LOGS_DIR / f"stockfish_1100_{benchmark_id}.pgn"
        txt_path = LOGS_DIR / f"stockfish_1100_{benchmark_id}.txt"
        atomic_json(json_path, report)
        atomic_write_text(
            md_path,
            "\n".join(
                [
                    "# MertFormer vs Stockfish 1100",
                    "",
                    f"- Benchmark ID: `{benchmark_id}`",
                    f"- Result: `{report['result']}`",
                    f"- Winner: `{report['winner']}`",
                    f"- Termination: `{report['termination']}`",
                    f"- Total plies: `{report['plies']}`",
                    f"- Device: `{report['device']}`",
                    f"- Stockfish path: `{report['stockfish']['path']}`",
                    f"- Stockfish settings: `{report['stockfish']['applied_options']}`",
                    f"- JSON log: `{json_path}`",
                    f"- PGN: `{pgn_path}`",
                    f"- Text log: `{txt_path}`",
                ]
            ) + "\n",
        )
        atomic_write_text(pgn_path, report["pgn"] + "\n")
        atomic_write_text(
            txt_path,
            "\n".join(
                [
                    f"benchmark_id={benchmark_id}",
                    f"result={report['result']}",
                    f"winner={report['winner']}",
                    f"termination={report['termination']}",
                    f"plies={report['plies']}",
                    f"duration_sec={report['duration_sec']}",
                    f"stockfish_path={report['stockfish']['path']}",
                    "",
                    "moves:",
                    *[
                        f"{item['ply']:03d} {item['actor']} {item['move']} {item['san']}"
                        for item in report["transcript"]
                    ],
                ]
            )
            + "\n",
        )
        return {"json": str(json_path), "md": str(md_path), "pgn": str(pgn_path), "txt": str(txt_path)}

    def _benchmark_worker(self) -> None:
        benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        started = time.time()
        board = self.onefile.chess.Board()
        game = self.onefile.chess.pgn.Game()
        game.headers["Event"] = "MertFormer vs Stockfish 1100"
        game.headers["Site"] = "Local GUI"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "MertFormer"
        game.headers["Black"] = "Stockfish 1100"
        node = game
        transcript: List[Dict[str, Any]] = []
        self.benchmark.running = True
        self.benchmark.status_main = "Benchmark running"
        self.benchmark.status_sub = "Model is White. Stockfish is Black at 1100 strength."
        try:
            if not self.stockfish_path:
                raise RuntimeError("Stockfish binary not found on this Mac")
            engine = self.onefile.chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            try:
                applied_options = self._configure_engine(engine)
                engine_id = dict(getattr(engine, "id", {}))
                last_move: Optional[Dict[str, str]] = None
                self._set_benchmark_live_state(board, transcript, last_move, None)
                for _ in range(DEFAULT_BENCHMARK_MAX_PLIES):
                    if board.is_game_over():
                        break
                    if board.turn == self.onefile.chess.WHITE:
                        trace = self._model_trace(board, mode="benchmark")
                        move = self.onefile.chess.Move.from_uci(trace["move"])
                        san = board.san(move)
                        board.push(move)
                        node = node.add_variation(move)
                        transcript.append(
                            {
                                "ply": len(transcript) + 1,
                                "actor": "model",
                                "color": "white",
                                "move": move.uci(),
                                "san": san,
                                "fen_after": board.fen(),
                                "value": float(trace.get("value", 0.0)),
                                "latency_ms": float(trace.get("latency_ms", 0.0)),
                                "response_contract": trace.get("response_contract"),
                                "trace": trace,
                            }
                        )
                        last_move = {"from": move.uci()[:2], "to": move.uci()[2:4]}
                        self._set_benchmark_live_state(board, transcript, last_move, trace)
                        time.sleep(DEFAULT_BENCHMARK_VIS_DELAY_SEC)
                    else:
                        result = engine.play(
                            board,
                            self.onefile.chess.engine.Limit(time=DEFAULT_STOCKFISH_TIME_SEC),
                            info=self.onefile.chess.engine.INFO_ALL,
                        )
                        move = result.move
                        if move is None:
                            raise RuntimeError("Stockfish did not return a move")
                        san = board.san(move)
                        board.push(move)
                        node = node.add_variation(move)
                        transcript.append(
                            {
                                "ply": len(transcript) + 1,
                                "actor": "stockfish",
                                "color": "black",
                                "move": move.uci(),
                                "san": san,
                                "fen_after": board.fen(),
                                "info": self._serialize_engine_info(getattr(result, "info", {})),
                            }
                        )
                        last_move = {"from": move.uci()[:2], "to": move.uci()[2:4]}
                        self._set_benchmark_live_state(board, transcript, last_move, None)
                        time.sleep(DEFAULT_BENCHMARK_VIS_DELAY_SEC)
                outcome = board.outcome()
                result_str = outcome.result() if outcome is not None else "*"
                winner = (
                    "white"
                    if outcome is not None and outcome.winner is self.onefile.chess.WHITE
                    else "black"
                    if outcome is not None and outcome.winner is self.onefile.chess.BLACK
                    else "draw"
                )
                termination = str(outcome.termination) if outcome is not None else "unfinished"
                game.headers["Result"] = result_str
                report = {
                    "benchmark_id": benchmark_id,
                    "started_at_utc": utc_now(),
                    "duration_sec": round(time.time() - started, 4),
                    "run_id": self.summary.run_id,
                    "device": str(self.device),
                    "checkpoint_path": str(CHECKPOINT_PATH),
                    "checkpoint_sha256": path_sha256(CHECKPOINT_PATH),
                    "result": result_str,
                    "winner": winner,
                    "termination": termination,
                    "plies": len(transcript),
                    "stockfish": {
                        "path": self.stockfish_path,
                        "engine_id": engine_id,
                        "applied_options": applied_options,
                        "elo_target": DEFAULT_STOCKFISH_ELO,
                        "time_per_move_sec": DEFAULT_STOCKFISH_TIME_SEC,
                    },
                    "model": {
                        "color": "white",
                        "run_id": self.summary.run_id,
                        "steps_completed": self.summary.steps_completed,
                    },
                    "transcript": transcript,
                    "final_fen": board.fen(),
                    "pgn": str(game),
                }
                paths = self._write_benchmark_outputs(benchmark_id, report)
                history_entry = {
                    "benchmark_id": benchmark_id,
                    "target_elo": DEFAULT_STOCKFISH_ELO,
                    "result": result_str,
                    "winner": winner,
                    "termination": termination,
                    "plies": len(transcript),
                    "duration_sec": round(time.time() - started, 4),
                    "json_path": paths["json"],
                    "md_path": paths["md"],
                    "pgn_path": paths["pgn"],
                    "txt_path": paths["txt"],
                }
                self.benchmark_history = ([history_entry] + self.benchmark_history)[:12]
                atomic_json(BENCHMARK_HISTORY_PATH, {"history": self.benchmark_history})
                self.benchmark.latest_report = report
                self.benchmark.latest_result = result_str
                self.benchmark.latest_plies = str(len(transcript))
                self.benchmark.log_path = paths["json"]
                self.benchmark.pgn_path = paths["pgn"]
                self.benchmark.status_main = f"Benchmark complete: {result_str}"
                self.benchmark.status_sub = f"Winner {winner} · {termination} · log saved locally."
                self._set_benchmark_live_state(board, transcript, last_move, None)
                self._log_event("benchmark_complete", **history_entry)
            finally:
                engine.quit()
        except Exception as exc:
            self.benchmark.status_main = "Benchmark failed"
            self.benchmark.status_sub = f"{type(exc).__name__}: {exc}"
            error_report = {
                "benchmark_id": benchmark_id,
                "started_at_utc": utc_now(),
                "duration_sec": round(time.time() - started, 4),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            error_path = LOGS_DIR / f"stockfish_1100_{benchmark_id}_failed.json"
            atomic_json(error_path, error_report)
            self.benchmark.log_path = str(error_path)
            self._log_event("benchmark_failed", benchmark_id=benchmark_id, error_type=type(exc).__name__, error=str(exc), log_path=str(error_path))
        finally:
            self.benchmark.running = False

    def start_benchmark(self) -> Dict[str, Any]:
        with self.lock:
            if self.benchmark.running:
                raise ValueError("Benchmark is already running")
            self._log_event("benchmark_started", stockfish_path=self.stockfish_path or "", elo=DEFAULT_STOCKFISH_ELO)
            worker = threading.Thread(target=self._benchmark_worker, daemon=True)
            worker.start()
            return self.state_payload()

    def set_mode_preferences(
        self,
        *,
        mode: Optional[str] = None,
        teaching_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            if mode is not None:
                self.ui_mode = normalize_gui_mode(self.onefile, str(mode))
            if teaching_level is not None:
                self.teaching_level = normalize_gui_teaching_level(self.onefile, str(teaching_level))
            self._log_event("mode_preferences_updated", ui_mode=self.ui_mode, teaching_level=self.teaching_level)
            return self.state_payload()

    def state_payload(self) -> Dict[str, Any]:
        legal_targets, legal_groups = self._legal_maps()
        result = self._result_payload()
        return {
            "run_id": self.summary.run_id,
            "rating_claim_status": self.summary.rating_claim_status,
            "device": str(self.device),
            "ui_mode": self.ui_mode,
            "teaching_level": self.teaching_level,
            "fen": self.board.fen(),
            "turn": self._actor_color_name(self.board.turn),
            "human_color": self._actor_color_name(self.human_color),
            "transcript": self.transcript,
            "last_trace": self.last_trace,
            "last_move": self.last_move,
            "status_main": self.status_main,
            "status_sub": self.status_sub,
            "legal_targets": legal_targets,
            "legal_groups": legal_groups,
            "check_square": self._check_square(),
            "summary": {
                "steps_completed": self.summary.steps_completed,
                "holdout_masked_accuracy": self.summary.holdout_masked_accuracy,
                "holdout_masked_top5": self.summary.holdout_masked_top5,
                "positions_total": self.summary.positions_total,
                "best_val_loss": self.summary.best_val_loss,
            },
            "benchmark": {
                "running": self.benchmark.running,
                "status_main": self.benchmark.status_main,
                "status_sub": self.benchmark.status_sub,
                "log_path": self.benchmark.log_path,
                "pgn_path": self.benchmark.pgn_path,
                "latest_result": self.benchmark.latest_result,
                "latest_plies": self.benchmark.latest_plies,
            },
            "benchmark_live": {
                "fen": self.benchmark.live_fen,
                "turn": self.benchmark.live_turn,
                "last_move": self.benchmark.live_last_move,
                "check_square": self.benchmark.live_check_square,
                "transcript": self.benchmark.live_transcript,
                "last_trace": self.benchmark.live_last_trace,
                "game_over": self.benchmark.live_game_over,
                "result": self.benchmark.live_result,
                "winner": self.benchmark.live_winner,
            },
            "benchmark_history": self.benchmark_history,
            **result,
        }


class ArenaHandler(BaseHTTPRequestHandler):
    state: ArenaState

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8")) if raw else {}

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(HTML)
            return
        if self.path == "/api/state":
            self._send_json(self.state.state_payload())
            return
        if self.path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "run_id": self.state.summary.run_id,
                    "device": str(self.state.device),
                    "stockfish_path": self.state.stockfish_path or "",
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        try:
            if self.path == "/api/new_game":
                payload = self._read_json()
                color = str(payload.get("human_color", "white")).lower()
                if color not in {"white", "black"}:
                    raise ValueError("human_color must be white or black")
                self._send_json(self.state.new_game(color))
                return
            if self.path == "/api/move":
                payload = self._read_json()
                move = str(payload.get("move", "")).strip()
                if not move:
                    raise ValueError("move is required")
                self._send_json(self.state.play_human_move(move))
                return
            if self.path == "/api/mode":
                payload = self._read_json()
                self._send_json(
                    self.state.set_mode_preferences(
                        mode=payload.get("mode"),
                        teaching_level=payload.get("teaching_level"),
                    )
                )
                return
            if self.path == "/api/undo":
                self._send_json(self.state.undo_full_turn())
                return
            if self.path == "/api/benchmark/start":
                self._send_json(self.state.start_benchmark())
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def pick_port(base_port: int) -> int:
    for port in range(base_port, base_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("Unable to find an open local port")


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_state(preferred_device: Optional[str]) -> ArenaState:
    if not ONEFILE_PATH.exists():
        raise FileNotFoundError(f"Missing onefile source: {ONEFILE_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing run summary: {SUMMARY_PATH}")

    onefile = load_module(ONEFILE_PATH)
    runtime_summary = load_summary(SUMMARY_PATH)
    cfg = dict(runtime_summary.get("config", {}))
    cfg["device"] = choose_device(onefile, preferred_device)
    cfg["compile_policy"] = "off"
    onefile.deterministic_seed(int(cfg.get("seed", 42)), strict=bool(cfg.get("determinism_strict", True)))
    device = onefile.pick_device(cfg)
    model = onefile.ChessPolicyValueNet(cfg, len(onefile.MOVE_VOCAB)).to(device)
    onefile.load_checkpoint(CHECKPOINT_PATH, model, optimizer=None, restore_optimizer=False)
    model.eval()

    summary = ArenaSummary(
        run_id=str(runtime_summary.get("run_id", "unknown")),
        rating_claim_status=str(runtime_summary.get("rating_claim_status", "no_claim")),
        steps_completed=int(runtime_summary.get("training_summary", {}).get("steps_completed", 0)),
        holdout_masked_accuracy=float(runtime_summary.get("holdout_validation", {}).get("metrics", {}).get("masked_policy_accuracy", 0.0)),
        holdout_masked_top5=float(runtime_summary.get("holdout_validation", {}).get("metrics", {}).get("masked_top5_accuracy", 0.0)),
        positions_total=int(runtime_summary.get("dataset_provenance", {}).get("data_stats", {}).get("positions_total", 0)),
        best_val_loss=float(runtime_summary.get("training_summary", {}).get("best_val_loss", 0.0)),
    )
    fast_arena_cfg = {
        "search_enabled": False,
        "search_auto_budget": False,
        "search_candidate_topk": 1,
        "search_reply_topk": 1,
    }
    state = ArenaState(
        onefile=onefile,
        model=model,
        device=device,
        summary=summary,
        human_color=onefile.chess.WHITE,
        fast_arena_cfg=fast_arena_cfg,
    )
    state._set_benchmark_live_state(state.onefile.chess.Board(), [], None, None)
    if BENCHMARK_HISTORY_PATH.exists():
        try:
            raw_history = list(json.loads(BENCHMARK_HISTORY_PATH.read_text(encoding="utf-8")).get("history", []))
            state.benchmark_history = [
                item
                for item in raw_history
                if item.get("target_elo") == DEFAULT_STOCKFISH_ELO
                or "stockfish_1100_" in str(item.get("json_path", ""))
            ][:12]
        except Exception as e:
            print(f"[warn] benchmark history load failed: {type(e).__name__}: {e}", file=sys.stderr)
            state.benchmark_history = []
    state.stockfish_path = detect_stockfish_path()
    if state.stockfish_path:
        state.benchmark.status_sub = f"Stockfish detected at {state.stockfish_path}. Model will play White at local 1100 benchmark settings."
    else:
        state.benchmark.status_sub = "Stockfish not found yet. Install it, then rerun the launcher."
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Local web GUI for the trained MertFormer Chess checkpoint")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    state = build_state(args.device)
    handler_cls = type("ArenaHandlerBound", (ArenaHandler,), {"state": state})
    port = pick_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    url = f"http://127.0.0.1:{port}"
    print(f"MertFormer Chess GUI ready on {url}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {state.device}")
    print(f"Stockfish: {state.stockfish_path or 'not found'}")
    if not args.no_browser and os.environ.get("MERTFORMER_CHESS_NO_BROWSER", "0") != "1":
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MertFormer Chess GUI…")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
