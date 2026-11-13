"""
Веб-версия классического инженерного стенда Nengo-FIM Simulator:
- Полная настройка параметров спайковой сети, каждого уровня (L1..L9)
  и каждого отдельного ФИМа (число синапсов Б1/Б2/Б3, веса, пороги V_th, окна W_L);
- Классический научный интерфейс в стиле Nengo GUI (белые графики, схема ансамблей,
  спайковый растр, осциллограф мембранных потенциалов V(t), трассировка последовательностей);
- Пул памяти 10 000 000 ФИМ (по 1 млн на уровень), обучение на корпусе русского языка.

Запуск:
    python3 fim_ui.py
Откройте в браузере: http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from fim_algorithm import InstinctNetwork, normalize_text
from fim_nengo import FIMNengoSimulator
from train_fim import train_default_network


NET: InstinctNetwork = train_default_network()
SIM: FIMNengoSimulator = FIMNengoSimulator(NET)


def analyze_input(text: str, learn: bool = False) -> Dict[str, Any]:
    """
    Прогоняет введённый пользователем текст через сеть и возвращает:
    - ответ на самом высоком узнанном уровне + подъём по исходящим синапсам;
    - полную трассировку спайков (включая 2-из-3 ассоциативное узнавание);
    - совпадения подстрок в памяти по уровням 1..9;
    - Nengo-симуляцию (импульсы 252 мс, растр, V(t) и sources_detail).
    """
    clean = normalize_text(text)
    if not clean:
        return {
            "clean_text": "",
            "top_level": 0,
            "top_response_text": "",
            "top_fims": [],
            "recalled_level": 0,
            "recalled_fim_id": None,
            "recalled_text": "",
            "top_explanation": None,
            "recalled_explanation": None,
            "stream_events": [],
            "substring_matches": {},
            "simulation": None,
            "summary": NET.summary(),
            "config": NET.get_config(),
        }

    raw_events = (
        NET.feed_stream(clean, flush_end=True)
        if learn
        else NET.probe_stream(clean)
    )
    stream_events: List[Dict[str, Any]] = [
        {
            "level": ev.level,
            "fim_id": ev.node_id,
            "type": ev.event_type,
            "text": NET.decode_fim(ev.level, ev.node_id),
            "recognitions": NET.levels[ev.level].fims[ev.node_id].recognitions,
            "confidence": round(float(ev.confidence), 2),
            "time_ms": ev.time_ms,
            "active_slots": list(ev.active_slots),
        }
        for ev in raw_events
    ]

    clean_len = len(clean)
    substring_matches: Dict[int, List[Dict[str, Any]]] = {}
    for lvl in range(1, NET.num_levels + 1):
        matches = []
        for fim in NET.levels[lvl].fims:
            if not fim.consolidated:
                continue
            decoded = fim.decoded_text or NET.decode_fim(lvl, fim.fim_id)
            if len(decoded) <= clean_len and decoded in clean:
                matches.append(
                    {
                        "fim_id": fim.fim_id,
                        "text": decoded,
                        "recognitions": fim.recognitions,
                    }
                )
        if matches:
            matches.sort(key=lambda m: (len(m["text"]), m["recognitions"]), reverse=True)
            substring_matches[lvl] = matches[:25]

    stream_top_lvl = max((ev["level"] for ev in stream_events), default=0)
    sub_top_lvl = max(substring_matches.keys(), default=0)

    if stream_top_lvl > 0 and stream_top_lvl >= sub_top_lvl:
        top_level = stream_top_lvl
        top_fims = [ev for ev in stream_events if ev["level"] == top_level]
        top_response_text = "".join(ev["text"] for ev in top_fims)
    elif sub_top_lvl > 0:
        top_level = sub_top_lvl
        top_fims = substring_matches[top_level]
        top_response_text = top_fims[0]["text"]
    else:
        top_level = 0
        top_fims = []
        top_response_text = ""

    recalled_level = 0
    recalled_fim_id = None
    recalled_text = ""
    top_explanation = None
    recalled_explanation = None
    if top_level > 0 and top_fims:
        first_fid = top_fims[0]["fim_id"]
        top_explanation = NET.explain_fim(top_level, first_fid, max_depth=4)
        recalled_level, recalled_fim_id, recalled_text = NET.recall_highest_from(
            top_level, first_fid
        )
        if recalled_level > top_level and recalled_fim_id is not None:
            recalled_explanation = NET.explain_fim(recalled_level, recalled_fim_id, max_depth=4)

    sim_target_lvl = top_level if top_level > 0 else None
    sim_target_fid = top_fims[0]["fim_id"] if (top_level > 0 and top_fims) else None
    simulation = SIM.run(
        clean,
        target_level=sim_target_lvl,
        target_fim_id=sim_target_fid,
        learn=False,
    )

    return {
        "clean_text": clean,
        "top_level": top_level,
        "top_response_text": top_response_text,
        "top_fims": top_fims,
        "recalled_level": recalled_level,
        "recalled_fim_id": recalled_fim_id,
        "recalled_text": recalled_text,
        "top_explanation": top_explanation,
        "recalled_explanation": recalled_explanation,
        "stream_events": stream_events,
        "substring_matches": substring_matches,
        "simulation": simulation,
        "summary": NET.summary(),
        "config": NET.get_config(),
    }


def get_level_fims(level: int, limit: int = 80, query: str = "") -> Dict[str, Any]:
    """Возвращает консолидированные ФИМы выбранного уровня с поиском по тексту или #ID."""
    if level < 1 or level > NET.num_levels:
        return {"level": level, "total_consolidated": 0, "matched_count": 0, "max_fims": 0, "fims": []}

    pool = NET.levels[level]
    cons = [f for f in pool.fims if f.consolidated]
    q = query.strip().lower()

    if q:
        if q.startswith("#") and q[1:].isdigit():
            target_id = int(q[1:])
            filtered = [f for f in cons if f.fim_id == target_id]
        else:
            id_num = int(q) if q.isdigit() else None
            filtered = [
                f
                for f in cons
                if (id_num is not None and f.fim_id == id_num)
                or q in (f.decoded_text or NET.decode_fim(level, f.fim_id)).lower()
                or q in (f.training_stage or "").lower()
            ]
    else:
        filtered = cons

    filtered.sort(key=lambda f: (f.recognitions, -f.fim_id), reverse=True)
    items = [
        {
            "fim_id": f.fim_id,
            "level": level,
            "text": f.decoded_text or NET.decode_fim(level, f.fim_id),
            "recognitions": f.recognitions,
            "synapse_slots": list(f.synapse_slots),
            "synapse_counts": list(f.get_counts()),
            "synapse_weights": [round(w, 3) for w in f.get_weights()],
            "v_threshold": round(f.v_threshold, 3),
            "window_ms": f.effective_window_ms(pool.window_ms),
            "sources": list(f.sources),
            "created_at_ms": f.created_at_ms,
            "training_context": f.training_context,
            "training_stage": f.training_stage,
        }
        for f in filtered[:limit]
    ]
    return {
        "level": level,
        "total_consolidated": len(cons),
        "matched_count": len(filtered),
        "max_fims": pool.max_fims,
        "fims": items,
    }


HTML_PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nengo-FIM Workbench — Настройка синапсов и спайковая симуляция (10M)</title>
  <style>
    :root {
      --desk-bg: #dde2ea;
      --pane-bg: #f4f6f9;
      --surface: #ffffff;
      --toolbar: #2b313a;
      --toolbar-border: #1e232a;
      --border: #b6c0ce;
      --grid-line: #dce1ea;
      --text: #181c22;
      --muted: #556070;
      --accent: #1d5fbf;
      --accent-hover: #174da0;
      --ok: #157347;
      --warn: #b45309;
      --danger: #b91c1c;
      --mono: "DejaVu Sans Mono", "Consolas", "Liberation Mono", monospace;
      --sans: "Segoe UI", Tahoma, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      background: var(--desk-bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 12px;
      line-height: 1.35;
    }

    /* Classic Nengo Top Toolbar */
    .nengo-toolbar {
      background: var(--toolbar);
      color: #f1f5f9;
      border-bottom: 2px solid var(--toolbar-border);
      padding: 6px 12px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .tb-left {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .tb-title {
      font-weight: 700;
      font-size: 13px;
      letter-spacing: 0.02em;
      color: #ffffff;
    }
    .tb-sub {
      font-family: var(--mono);
      font-size: 11px;
      color: #b7c3d4;
      border-left: 1px solid #485262;
      padding-left: 10px;
    }
    .tb-transport {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 5px;
      background: #1f242b;
      border: 1px solid #454f5e;
      padding: 3px 6px;
    }
    .tb-stats {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-family: var(--mono);
      font-size: 11px;
    }
    .tb-stat {
      background: #1f242b;
      border: 1px solid #454f5e;
      padding: 3px 7px;
      color: #d6dfeb;
    }
    .tb-stat b { color: #7cc4ff; }

    /* Buttons & Inputs: flat classic engineering style */
    button {
      font-family: var(--sans);
      font-size: 11.5px;
      font-weight: 600;
      padding: 4px 9px;
      border: 1px solid #96a2b3;
      background: #eceff4;
      color: #181c22;
      border-radius: 2px;
      cursor: pointer;
    }
    button:hover { background: #dfe5ee; border-color: #6b788b; }
    button.primary {
      background: var(--accent);
      border-color: #13448c;
      color: #ffffff;
    }
    button.primary:hover { background: var(--accent-hover); }
    button.good {
      background: #1e7e4f;
      border-color: #145b38;
      color: #ffffff;
    }
    button.good:hover { background: #18663f; }
    button.danger {
      background: #c53030;
      border-color: #9b2222;
      color: #ffffff;
    }
    button.danger:hover { background: #9b2222; }
    button.dark-btn {
      background: #3a424e;
      border-color: #5a6575;
      color: #f8fafc;
      padding: 3px 8px;
    }
    button.dark-btn:hover { background: #4b5563; }
    button.tab-btn.active {
      background: var(--accent);
      border-color: #13448c;
      color: #ffffff;
    }

    input[type="text"], input[type="number"], select, textarea {
      font-family: var(--mono);
      font-size: 11.5px;
      background: #ffffff;
      color: #111827;
      border: 1px solid #9ba6b7;
      border-radius: 2px;
      padding: 3px 6px;
    }
    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px rgba(29,95,191,0.2);
    }

    /* 3-Column Desktop Workbench */
    .workbench {
      display: grid;
      grid-template-columns: 350px minmax(520px, 1fr) 390px;
      gap: 6px;
      padding: 6px;
      min-height: calc(100vh - 46px);
      align-items: start;
    }
    @media (max-width: 1280px) {
      .workbench { grid-template-columns: 330px 1fr; }
      .right-dock { grid-column: 1 / -1; }
    }
    @media (max-width: 860px) {
      .workbench { grid-template-columns: 1fr; }
    }

    .col {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
    }
    .win {
      background: var(--pane-bg);
      border: 1px solid var(--border);
      border-radius: 2px;
    }
    .win-head {
      background: #e2e7ef;
      border-bottom: 1px solid var(--border);
      padding: 5px 8px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
    }
    .win-title {
      font-weight: 700;
      font-size: 11.5px;
      color: #1e242c;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .win-body {
      padding: 8px;
    }

    /* Parameter Forms */
    .param-grid {
      display: grid;
      grid-template-columns: 1fr 86px;
      gap: 5px 8px;
      align-items: center;
    }
    .param-grid label {
      color: #27303c;
      font-size: 11.5px;
    }
    .param-grid input[type="number"], .param-grid select {
      width: 100%;
      text-align: right;
    }
    .chk-row {
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 6px 0;
      font-size: 11.5px;
      color: #1f2730;
    }
    .btn-row {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      align-items: center;
    }

    /* Tables */
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      font-size: 11.5px;
    }
    th, td {
      border: 1px solid var(--grid-line);
      padding: 4px 6px;
      text-align: left;
      vertical-align: middle;
    }
    th {
      background: #ebeff5;
      color: #27303c;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    td input[type="number"], td input[type="text"], td select {
      width: 100%;
      padding: 2px 4px;
      font-size: 11px;
    }
    tr.clickable:hover { background: #eef4fc; cursor: pointer; }
    tr.selected-row { background: #dceafb !important; }
    .scroll-box {
      max-height: 270px;
      overflow-y: auto;
      border: 1px solid var(--border);
      background: #ffffff;
    }
    .scroll-tall {
      max-height: 360px;
      overflow-y: auto;
      border: 1px solid var(--border);
      background: #ffffff;
    }

    /* Center Plots & Nengo Schematic */
    .probe-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    @media (max-width: 1050px) {
      .probe-grid { grid-template-columns: 1fr; }
    }
    canvas.nengo-canvas {
      display: block;
      width: 100%;
      background: #ffffff;
      border: 1px solid var(--border);
    }
    .readout-strip {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 6px;
    }
    .readout-box {
      background: #ffffff;
      border: 1px solid var(--border);
      padding: 6px 8px;
    }
    .readout-label {
      font-size: 10.5px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
    }
    .readout-val {
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 700;
      margin-top: 2px;
      color: #111827;
      word-break: break-word;
    }

    /* Tags & Badges */
    .tag {
      display: inline-block;
      font-family: var(--mono);
      font-size: 10.5px;
      padding: 1px 5px;
      border: 1px solid #b6c0ce;
      background: #f1f4f8;
      border-radius: 2px;
      margin: 1px 2px 1px 0;
    }
    .tag.hit { background: #e6f4ea; border-color: #6ab586; color: #0f5132; }
    .tag.miss { background: #fff4e5; border-color: #d99b42; color: #7c3f00; text-decoration: line-through; }
    .tag.assoc { background: #fff3cd; border-color: #e0b137; color: #7a5200; }
    .mono { font-family: var(--mono); }
    .muted { color: var(--muted); font-size: 11px; }
    .status-msg {
      font-family: var(--mono);
      font-size: 11px;
      padding: 4px 6px;
      margin-top: 6px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      display: none;
    }
    .status-msg.show { display: block; }
    .status-msg.ok { border-color: #5ca97a; background: #edf7f0; color: #14532d; }
    .status-msg.err { border-color: #d96b6b; background: #fdf2f2; color: #7f1d1d; }

    /* Decomposition tree */
    .tree-node {
      margin-left: 12px;
      padding-left: 8px;
      border-left: 1px solid #9ba6b7;
      margin-top: 4px;
    }
    .tree-root { margin-left: 0; border-left: none; padding-left: 0; }
    .tree-card {
      background: #ffffff;
      border: 1px solid var(--grid-line);
      padding: 4px 6px;
      cursor: pointer;
    }
    .tree-card:hover { border-color: var(--accent); background: #f3f7fd; }
  </style>
</head>
<body>

<div class="nengo-toolbar">
  <div class="tb-left">
    <div class="tb-title">Nengo-FIM Workbench</div>
    <div class="tb-sub">Симулятор и конструктор спайковой памяти ФИМ (пул 10M, L0–L9)</div>
  </div>
  <div class="tb-transport">
    <button class="dark-btn" onclick="resetPlayhead()" title="В начало (0 мс)">|&lt;&lt;</button>
    <button class="dark-btn" onclick="stepPlayhead(-1)" title="Назад на 1 такт">-Такт</button>
    <button class="dark-btn" id="playPauseBtn" onclick="togglePlay()">&#9654; Пуск</button>
    <button class="dark-btn" onclick="stepPlayhead(1)" title="Вперёд на 1 такт">+Такт</button>
    <input type="range" id="timeSlider" min="0" max="252" step="12" value="252" oninput="scrubTime(this.value)" style="width:140px;" />
    <span class="mono" id="clockLabel" style="color:#7cc4ff; min-width:85px;">t = 0 мс</span>
    <select id="speedSelect" style="background:#2b313a; color:#f1f5f9; border-color:#525d6d;">
      <option value="0.25">0.25x</option>
      <option value="0.5" selected>0.5x</option>
      <option value="1">1.0x</option>
      <option value="2">2.0x</option>
    </select>
  </div>
  <div class="tb-stats">
    <div class="tb-stat">ФИМ в памяти: <b id="statTotalFims">—</b> / 9 000 000</div>
    <div class="tb-stat">Такт L0: <b id="statTickMs">84 мс</b></div>
    <div class="tb-stat">Инстинкт: <b id="statInstinctMs">252 мс</b></div>
    <div class="tb-stat">V_th: <b id="statVth">2.00</b></div>
  </div>
</div>

<div class="workbench">
  <!-- LEFT DOCK: Global Network Config, Level Configs, FIM Memory Explorer -->
  <div class="col left-dock">
    <div class="win">
      <div class="win-head">
        <span class="win-title">1. Параметры спайковой сети и синапсов</span>
      </div>
      <div class="win-body">
        <div class="param-grid">
          <label for="cfgMinSyn">Мин. синапсов на ФИМ (шт.):</label>
          <select id="cfgMinSyn"><option value="2">2</option><option value="3">3</option></select>

          <label for="cfgMaxSyn">Макс. синапсов на ФИМ (шт.):</label>
          <select id="cfgMaxSyn"><option value="3">3</option><option value="2">2</option></select>

          <label for="cfgDefCount">Синапсов на 1 вход БН (шт.):</label>
          <input type="number" id="cfgDefCount" min="1" max="32" step="1" value="1" />

          <label for="cfgDefWeight">Вес одного синапса w:</label>
          <input type="number" id="cfgDefWeight" min="0.05" max="10" step="0.05" value="1.05" />

          <label for="cfgDefVth">Порог головного нейрона V_th:</label>
          <input type="number" id="cfgDefVth" min="0.2" max="20" step="0.05" value="2.00" />

          <label for="cfgTickMs">Такт уровня L0 (мс):</label>
          <input type="number" id="cfgTickMs" min="12" max="1000" step="12" value="84" />

          <label for="cfgInstinctMs">Цикл инстинкта (мс):</label>
          <input type="number" id="cfgInstinctMs" min="24" max="5000" step="12" value="252" />

          <label for="cfgDtMs">Шаг интегратора dt (мс):</label>
          <input type="number" id="cfgDtMs" min="1" max="84" step="1" value="12" />

          <label for="cfgTau">Коэфф. утечки мембраны τ:</label>
          <input type="number" id="cfgTau" min="0.1" max="20" step="0.1" value="1.4" />

          <label for="cfgL2Chars">Макс. символов слова (L2):</label>
          <input type="number" id="cfgL2Chars" min="2" max="18" step="1" value="6" />
        </div>
        <label class="chk-row">
          <input type="checkbox" id="cfgAssoc23" checked />
          Разрешить ассоциативное узнавание 2-из-3 синапсов
        </label>
        <label class="chk-row">
          <input type="checkbox" id="cfgApplyExisting" checked />
          Применить веса, кол-во и V_th ко всем ФИМ сети
        </label>
        <div class="btn-row">
          <button class="primary" onclick="saveGlobalConfig()">Применить и пересчитать</button>
          <button onclick="loadConfig()">Сбросить к текущим</button>
        </div>
        <div id="globalCfgStatus" class="status-msg"></div>
      </div>
    </div>

    <div class="win">
      <div class="win-head">
        <span class="win-title">2. Параметры уровней (L1–L9, пул 1M/ур.)</span>
        <button onclick="saveLevelConfigs()">Сохранить окна W_L и пороги</button>
      </div>
      <div class="win-body" style="padding:4px;">
        <div class="scroll-box" style="max-height:245px;">
          <table>
            <thead>
              <tr>
                <th>Ур.</th>
                <th>Окно W_L (мс)</th>
                <th>V_th</th>
                <th>ФИМ в памяти</th>
              </tr>
            </thead>
            <tbody id="levelsConfigBody"></tbody>
          </table>
        </div>
        <div id="levelCfgStatus" class="status-msg"></div>
      </div>
    </div>

    <div class="win">
      <div class="win-head">
        <span class="win-title">3. База ФИМ уровня L<span id="memCurLvl">3</span></span>
        <span class="mono muted" id="memCountInfo">—</span>
      </div>
      <div class="win-body">
        <div class="btn-row" id="levelTabs" style="margin-bottom:6px;"></div>
        <div class="btn-row" style="margin-bottom:6px;">
          <input type="text" id="memSearch" placeholder="Поиск текста или #ID..." style="flex:1;" oninput="searchCurrentLevel()" />
          <button onclick="searchCurrentLevel()">Найти</button>
        </div>
        <div class="scroll-box" style="max-height:290px;">
          <table>
            <thead>
              <tr>
                <th>#ID</th>
                <th>Текст ФИМ</th>
                <th>Син.</th>
                <th>V_th</th>
                <th>Узн.</th>
              </tr>
            </thead>
            <tbody id="memListBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- CENTER WORKSPACE: Input Stream, Nengo Schematic, Probes (Raster + V(t)), Trace Table -->
  <div class="col center-pane">
    <div class="win">
      <div class="win-head">
        <span class="win-title">Входной поток элементов (L0 = 41 символ, такт <span id="hdrTick">84</span> мс)</span>
        <span class="mono muted" id="latencyBadge">Готов</span>
      </div>
      <div class="win-body">
        <textarea id="inputText" rows="2" placeholder="Введите слово, фразу или предложение...">мама раму</textarea>
        <div class="btn-row" style="margin-top:6px;">
          <button class="primary" onclick="sendText(false)">Запустить симуляцию (Проверка)</button>
          <button class="good" onclick="sendText(true)">Подать с обучением (+ Записать ФИМ)</button>
          <button onclick="setExample('мама мыла раму')">мама мыла раму</button>
          <button onclick="setExample('мама раму')">мама раму (2 из 3)</button>
          <button onclick="setExample('дом у берега')">дом у берега</button>
          <button onclick="setExample('Только Иван выразил готовность помочь.')">Корпус RuCoLA</button>
        </div>
        <div class="readout-strip">
          <div class="readout-box">
            <div class="readout-label">Узнавание на верхнем активном уровне</div>
            <div class="readout-val" id="topAnswerMain">—</div>
            <div id="topAnswerMeta" style="margin-top:3px;"></div>
          </div>
          <div class="readout-box">
            <div class="readout-label">Всплытие вверх по исходящим синапсам (достройка)</div>
            <div class="readout-val" id="recallMain">—</div>
            <div id="recallMeta" style="margin-top:3px;"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="win">
      <div class="win-head">
        <span class="win-title">Схема ансамбля Nengo и синапсов активного ФИМа (клик по узлу — выбор ФИМа)</span>
        <span class="mono muted" id="dagFocusLabel">Нет выбранного модуля</span>
      </div>
      <div class="win-body" style="padding:6px;">
        <canvas id="dagCanvas" class="nengo-canvas" height="235"></canvas>
      </div>
    </div>

    <div class="probe-grid">
      <div class="win">
        <div class="win-head">
          <span class="win-title">Nengo Probe: Спайковый растр (L0..L9 + Инстинкт)</span>
          <span class="mono muted" id="rasterMeta">—</span>
        </div>
        <div class="win-body" style="padding:6px;">
          <canvas id="rasterCanvas" class="nengo-canvas" height="220"></canvas>
        </div>
      </div>

      <div class="win">
        <div class="win-head">
          <span class="win-title">Nengo Probe: Осциллограф V(t) нейронов Б1, Б2, Б3 и Г</span>
          <span class="mono" id="oscReadout" style="color:#1d5fbf; font-size:11px;">V_Г=0.00 / V_th=2.00</span>
        </div>
        <div class="win-body" style="padding:6px;">
          <canvas id="oscCanvas" class="nengo-canvas" height="220"></canvas>
        </div>
      </div>
    </div>

    <div class="win">
      <div class="win-head">
        <span class="win-title">Трассировка каждой последовательности: Б1 + Б2 + Б3 &rarr; Г (<span id="traceCount">0</span> срабатываний)</span>
        <div class="btn-row">
          <div class="btn-row" id="traceLevelFilters"></div>
          <input type="text" id="traceFilterInput" placeholder="Фильтр текста..." oninput="renderTraceTable()" style="width:140px;" />
        </div>
      </div>
      <div class="win-body" style="padding:4px;">
        <div class="scroll-tall">
          <table>
            <thead>
              <tr>
                <th>t (мс)</th>
                <th>Ур. / #ID</th>
                <th>Текст ФИМ</th>
                <th>Синапсы Б1 + Б2 + Б3 (источник · кол-во · вес)</th>
                <th>Окно / V_th</th>
                <th>Режим</th>
                <th>Настройка</th>
              </tr>
            </thead>
            <tbody id="traceBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT DOCK: Per-FIM Synapse & Threshold Editor + Hierarchy Tree -->
  <div class="col right-dock">
    <div class="win">
      <div class="win-head">
        <span class="win-title" id="editorTitle">Редактор ФИМа (выберите ФИМ)</span>
        <button onclick="openNewFimMode()">+ Создать новый ФИМ</button>
      </div>
      <div class="win-body">
        <div class="param-grid" style="grid-template-columns: 1fr 120px;">
          <label for="edLevel">Уровень (L1–L9):</label>
          <select id="edLevel"></select>

          <label for="edFimId">ID модуля:</label>
          <input type="text" id="edFimId" readonly value="—" style="background:#eef1f6;" />

          <label>Декодированный текст:</label>
          <div id="edDecodedText" class="mono" style="font-weight:700; color:#1d5fbf; text-align:right; overflow:hidden; text-overflow:ellipsis;">—</div>

          <label for="edSynMode">Число активных слотов ФИМа:</label>
          <select id="edSynMode" onchange="toggleB2Slot()">
            <option value="3">3 синапса (Б1, Б2, Б3)</option>
            <option value="2">2 синапса (Б1, Б3)</option>
          </select>
        </div>

        <div style="margin-top:8px; font-weight:700; font-size:11px; color:#27303c; text-transform:uppercase;">
          Синапсы базовых нейронов (источник, шт., вес)
        </div>
        <table style="margin-top:4px;">
          <thead>
            <tr>
              <th style="width:36px;">БН</th>
              <th>Источник (L-1 #ID или текст)</th>
              <th style="width:52px;">Шт.</th>
              <th style="width:62px;">Вес w</th>
              <th style="width:52px;">ΔV</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="mono"><b>Б1</b></td>
              <td><input type="text" id="edSrc1" placeholder="символ или #ID/текст" /></td>
              <td><input type="number" id="edCnt1" min="1" max="64" step="1" value="1" oninput="recalcDrive()" /></td>
              <td><input type="number" id="edWgt1" min="0.05" max="10" step="0.05" value="1.05" oninput="recalcDrive()" /></td>
              <td class="mono" id="edDrv1">1.05</td>
            </tr>
            <tr id="edRowB2">
              <td class="mono"><b>Б2</b></td>
              <td><input type="text" id="edSrc2" placeholder="символ или #ID/текст" /></td>
              <td><input type="number" id="edCnt2" min="1" max="64" step="1" value="1" oninput="recalcDrive()" /></td>
              <td><input type="number" id="edWgt2" min="0.05" max="10" step="0.05" value="1.05" oninput="recalcDrive()" /></td>
              <td class="mono" id="edDrv2">1.05</td>
            </tr>
            <tr>
              <td class="mono"><b>Б3</b></td>
              <td><input type="text" id="edSrc3" placeholder="символ или #ID/текст" /></td>
              <td><input type="number" id="edCnt3" min="1" max="64" step="1" value="1" oninput="recalcDrive()" /></td>
              <td><input type="number" id="edWgt3" min="0.05" max="10" step="0.05" value="1.05" oninput="recalcDrive()" /></td>
              <td class="mono" id="edDrv3">1.05</td>
            </tr>
          </tbody>
        </table>

        <div class="param-grid" style="margin-top:8px; grid-template-columns: 1fr 120px;">
          <label for="edVth">Порог срабатывания Г (V_th):</label>
          <input type="number" id="edVth" min="0.2" max="20" step="0.05" value="2.00" oninput="recalcDrive()" />

          <label for="edWin">Индив. окно W (мс, пусто=ур.):</label>
          <input type="number" id="edWin" min="24" max="2000000" step="12" placeholder="по уровню" />

          <label for="edRecs">Счётчик узнаваний:</label>
          <input type="number" id="edRecs" min="0" step="1" value="1" />

          <label>Суммарный вклад ΣΔV:</label>
          <div class="mono" id="edSumDrive" style="text-align:right; font-weight:700;">3.15 &ge; 2.00</div>
        </div>

        <label class="chk-row">
          <input type="checkbox" id="edConsolidated" checked />
          Модуль активен и консолидирован в памяти
        </label>

        <div class="btn-row" style="margin-top:6px;">
          <button class="primary" id="btnSaveFim" onclick="saveCurrentFim()">Сохранить изменения ФИМ</button>
          <button class="good" onclick="createNewFim()">Создать новый ФИМ</button>
          <button class="danger" id="btnDelFim" onclick="disableCurrentFim()">Отключить</button>
        </div>
        <div id="fimEditStatus" class="status-msg"></div>
      </div>
    </div>

    <div class="win">
      <div class="win-head">
        <span class="win-title">Причинная иерархия выбранного ФИМа (вниз до L0 и вверх)</span>
      </div>
      <div class="win-body">
        <div class="muted" id="edStageMeta">Выберите ФИМ для анализа обучения и синаптической цепи</div>
        <div id="parentsBlock" style="margin-top:6px;"></div>
        <div id="treeBlock" style="margin-top:8px; max-height:360px; overflow-y:auto;"></div>
      </div>
    </div>
  </div>
</div>

<script>
let currentSim = null;
let currentTraceFilterLvl = 0;
let selectedSpikeKey = null;
let playheadMs = 0;
let isPlaying = false;
let playTimer = null;
let currentMemLevel = 3;
let netConfig = null;
let editorMode = 'edit'; // 'edit' | 'create'
let currentEditedFim = { level: 3, fim_id: null };

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showMsg(elId, msg, isErr=false) {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.className = 'status-msg show ' + (isErr ? 'err' : 'ok');
  setTimeout(() => { el.className = 'status-msg'; }, 4500);
}

// --- Init Level Selectors ---
(function initControls() {
  const edLvl = document.getElementById('edLevel');
  for (let l = 1; l <= 9; l++) {
    const opt = document.createElement('option');
    opt.value = l;
    opt.textContent = 'L' + l;
    if (l === 3) opt.selected = true;
    edLvl.appendChild(opt);
  }
  const tabs = document.getElementById('levelTabs');
  for (let l = 1; l <= 9; l++) {
    const b = document.createElement('button');
    b.className = 'tab-btn' + (l === 3 ? ' active' : '');
    b.textContent = 'L' + l;
    b.onclick = () => inspectLevel(l);
    tabs.appendChild(b);
  }
  const flt = document.getElementById('traceLevelFilters');
  const allBtn = document.createElement('button');
  allBtn.className = 'tab-btn active';
  allBtn.textContent = 'Все';
  allBtn.onclick = () => setTraceFilter(0, allBtn);
  flt.appendChild(allBtn);
  for (let l = 1; l <= 9; l++) {
    const b = document.createElement('button');
    b.className = 'tab-btn';
    b.textContent = 'L' + l;
    b.onclick = () => setTraceFilter(l, b);
    flt.appendChild(b);
  }
})();

function setTraceFilter(lvl, btn) {
  currentTraceFilterLvl = lvl;
  document.querySelectorAll('#traceLevelFilters button').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderTraceTable();
}

function setExample(text) {
  document.getElementById('inputText').value = text;
  sendText(false);
}

// --- Network & Level Configuration ---
async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();
  applyConfigToUI(cfg);
}

function applyConfigToUI(cfg) {
  if (!cfg) return;
  netConfig = cfg;
  document.getElementById('cfgDefCount').value = cfg.default_synapse_count;
  document.getElementById('cfgDefWeight').value = cfg.default_synapse_weight;
  document.getElementById('cfgDefVth').value = cfg.default_v_threshold;
  document.getElementById('cfgTickMs').value = cfg.tick_ms;
  document.getElementById('cfgInstinctMs').value = cfg.instinct_cycle_ms;
  document.getElementById('cfgDtMs').value = cfg.dt_ms;
  document.getElementById('cfgTau').value = cfg.tau_leak_factor;
  document.getElementById('cfgL2Chars').value = cfg.level2_max_chars;
  document.getElementById('cfgAssoc23').checked = !!cfg.allow_associative_2of3;

  const l1 = cfg.levels && (cfg.levels['1'] || cfg.levels[1]);
  if (l1) {
    document.getElementById('cfgMinSyn').value = String(l1.min_synapses);
    document.getElementById('cfgMaxSyn').value = String(l1.max_synapses);
  }
  document.getElementById('statTickMs').textContent = cfg.tick_ms + ' мс';
  document.getElementById('hdrTick').textContent = cfg.tick_ms;
  document.getElementById('statInstinctMs').textContent = cfg.instinct_cycle_ms + ' мс';
  document.getElementById('statVth').textContent = Number(cfg.default_v_threshold).toFixed(2);
  renderLevelsConfigTable(cfg.levels);
}

function renderLevelsConfigTable(levelsObj) {
  if (!levelsObj) return;
  const tbody = document.getElementById('levelsConfigBody');
  let html = '';
  let totalFims = 0;
  for (let l = 1; l <= 9; l++) {
    const lv = levelsObj[String(l)] || levelsObj[l];
    if (!lv) continue;
    totalFims += lv.consolidated_fims;
    html += `<tr class="clickable ${l === currentMemLevel ? 'selected-row' : ''}" onclick="if(event.target.tagName!=='INPUT') inspectLevel(${l})">
      <td class="mono"><b>L${l}</b></td>
      <td><input type="number" id="lvlWin_${l}" value="${lv.window_ms}" min="24" step="12" /></td>
      <td><input type="number" id="lvlVth_${l}" value="${lv.v_threshold}" min="0.2" max="20" step="0.05" /></td>
      <td class="mono">${lv.consolidated_fims.toLocaleString('ru-RU')}</td>
    </tr>`;
  }
  tbody.innerHTML = html;
  document.getElementById('statTotalFims').textContent = totalFims.toLocaleString('ru-RU');
}

async function saveGlobalConfig() {
  const payload = {
    min_synapses: Number(document.getElementById('cfgMinSyn').value),
    max_synapses: Number(document.getElementById('cfgMaxSyn').value),
    default_synapse_count: Number(document.getElementById('cfgDefCount').value),
    default_synapse_weight: Number(document.getElementById('cfgDefWeight').value),
    default_v_threshold: Number(document.getElementById('cfgDefVth').value),
    tick_ms: Number(document.getElementById('cfgTickMs').value),
    instinct_cycle_ms: Number(document.getElementById('cfgInstinctMs').value),
    dt_ms: Number(document.getElementById('cfgDtMs').value),
    tau_leak_factor: Number(document.getElementById('cfgTau').value),
    level2_max_chars: Number(document.getElementById('cfgL2Chars').value),
    allow_associative_2of3: document.getElementById('cfgAssoc23').checked,
    apply_to_existing: document.getElementById('cfgApplyExisting').checked,
  };
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (res.ok) {
    applyConfigToUI(data.config);
    showMsg('globalCfgStatus', 'Параметры сети применены и пересчитаны.');
    await sendText(false);
    await searchCurrentLevel();
  } else {
    showMsg('globalCfgStatus', data.error || 'Ошибка сохранения', true);
  }
}

async function saveLevelConfigs() {
  const levels = {};
  for (let l = 1; l <= 9; l++) {
    const wEl = document.getElementById('lvlWin_' + l);
    const vEl = document.getElementById('lvlVth_' + l);
    if (wEl && vEl) {
      levels[l] = {
        window_ms: Number(wEl.value),
        v_threshold: Number(vEl.value),
      };
    }
  }
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      levels,
      apply_to_existing: document.getElementById('cfgApplyExisting').checked
    })
  });
  const data = await res.json();
  if (res.ok) {
    applyConfigToUI(data.config);
    showMsg('levelCfgStatus', 'Окна W_L и пороги уровней обновлены.');
    await sendText(false);
  } else {
    showMsg('levelCfgStatus', data.error || 'Ошибка', true);
  }
}

// --- Memory Browser ---
async function inspectLevel(lvl) {
  currentMemLevel = lvl;
  document.getElementById('memCurLvl').textContent = lvl;
  document.querySelectorAll('#levelTabs button').forEach((b, i) => {
    b.classList.toggle('active', i + 1 === lvl);
  });
  if (netConfig && netConfig.levels) renderLevelsConfigTable(netConfig.levels);
  await searchCurrentLevel();
}

async function searchCurrentLevel() {
  const q = document.getElementById('memSearch').value || '';
  const res = await fetch(`/api/level?lvl=${currentMemLevel}&limit=80&q=${encodeURIComponent(q)}`);
  const data = await res.json();
  document.getElementById('memCountInfo').textContent = `${data.matched_count} / ${data.total_consolidated.toLocaleString('ru-RU')}`;
  const tbody = document.getElementById('memListBody');
  if (!data.fims || !data.fims.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">Ничего не найдено</td></tr>';
    return;
  }
  tbody.innerHTML = data.fims.map(f => {
    const isSel = currentEditedFim.level === currentMemLevel && currentEditedFim.fim_id === f.fim_id;
    const cntStr = (f.synapse_counts || []).join('/');
    return `<tr class="clickable ${isSel ? 'selected-row' : ''}" onclick="inspectFim(${currentMemLevel}, ${f.fim_id})">
      <td class="mono">#${f.fim_id}</td>
      <td class="mono"><b>${esc(f.text)}</b></td>
      <td class="mono">${f.synapse_slots.length} (${cntStr})</td>
      <td class="mono">${Number(f.v_threshold).toFixed(2)}</td>
      <td class="mono">${f.recognitions}</td>
    </tr>`;
  }).join('');
}

// --- Per-FIM Editor ---
function toggleB2Slot() {
  const mode = Number(document.getElementById('edSynMode').value);
  document.getElementById('edRowB2').style.display = (mode === 3) ? '' : 'none';
  recalcDrive();
}

function slotDrive(cnt, wgt) {
  return Number(wgt) * (1.0 + 0.15 * Math.max(0, Number(cnt) - 1));
}

function recalcDrive() {
  const mode = Number(document.getElementById('edSynMode').value);
  const d1 = slotDrive(document.getElementById('edCnt1').value, document.getElementById('edWgt1').value);
  const d2 = mode === 3 ? slotDrive(document.getElementById('edCnt2').value, document.getElementById('edWgt2').value) : 0;
  const d3 = slotDrive(document.getElementById('edCnt3').value, document.getElementById('edWgt3').value);
  document.getElementById('edDrv1').textContent = d1.toFixed(2);
  document.getElementById('edDrv2').textContent = d2.toFixed(2);
  document.getElementById('edDrv3').textContent = d3.toFixed(2);
  const sum = d1 + d2 + d3;
  const vth = Number(document.getElementById('edVth').value || 2.0);
  const el = document.getElementById('edSumDrive');
  el.textContent = `${sum.toFixed(2)} ${sum + 1e-9 >= vth ? '≥' : '<'} ${vth.toFixed(2)}`;
  el.style.color = sum + 1e-9 >= vth ? '#157347' : '#b91c1c';
}

function openNewFimMode() {
  editorMode = 'create';
  document.getElementById('editorTitle').textContent = 'Создание нового ФИМа';
  document.getElementById('edLevel').disabled = false;
  document.getElementById('edLevel').value = String(currentMemLevel);
  document.getElementById('edFimId').value = 'Новый';
  document.getElementById('edDecodedText').textContent = '(по источникам)';
  document.getElementById('edSynMode').value = '3';
  document.getElementById('edSrc1').value = '';
  document.getElementById('edSrc2').value = '';
  document.getElementById('edSrc3').value = '';
  document.getElementById('edCnt1').value = 1;
  document.getElementById('edCnt2').value = 1;
  document.getElementById('edCnt3').value = 1;
  document.getElementById('edWgt1').value = 1.05;
  document.getElementById('edWgt2').value = 1.05;
  document.getElementById('edWgt3').value = 1.05;
  document.getElementById('edVth').value = netConfig ? netConfig.default_v_threshold : 2.0;
  document.getElementById('edWin').value = '';
  document.getElementById('edRecs').value = 1;
  document.getElementById('edConsolidated').checked = true;
  document.getElementById('btnSaveFim').disabled = true;
  document.getElementById('btnDelFim').disabled = true;
  toggleB2Slot();
}

function loadFimIntoEditor(exp) {
  if (!exp || exp.level < 1) return;
  editorMode = 'edit';
  currentEditedFim = { level: exp.level, fim_id: exp.fim_id };
  document.getElementById('editorTitle').textContent = `Редактор ФИМ L${exp.level} #${exp.fim_id}`;
  document.getElementById('edLevel').value = String(exp.level);
  document.getElementById('edLevel').disabled = true;
  document.getElementById('edFimId').value = '#' + exp.fim_id;
  document.getElementById('edDecodedText').textContent = '«' + exp.text + '»';

  const ch = exp.children || [];
  const isTwo = ch.length === 2;
  document.getElementById('edSynMode').value = isTwo ? '2' : '3';

  const fillSlot = (idx, sIn, cIn, wIn) => {
    const item = ch[idx];
    if (!item) {
      document.getElementById(sIn).value = '';
      document.getElementById(cIn).value = 1;
      document.getElementById(wIn).value = 1.05;
      return;
    }
    const label = item.level === 0 ? item.text : ('#' + item.fim_id + ' (' + item.text + ')');
    document.getElementById(sIn).value = item.level === 0 ? item.text : ('#' + item.fim_id);
    document.getElementById(cIn).value = item.synapse_count || 1;
    document.getElementById(wIn).value = item.weight || 1.05;
  };

  if (isTwo) {
    fillSlot(0, 'edSrc1', 'edCnt1', 'edWgt1');
    document.getElementById('edSrc2').value = '';
    fillSlot(1, 'edSrc3', 'edCnt3', 'edWgt3');
  } else {
    fillSlot(0, 'edSrc1', 'edCnt1', 'edWgt1');
    fillSlot(1, 'edSrc2', 'edCnt2', 'edWgt2');
    fillSlot(2, 'edSrc3', 'edCnt3', 'edWgt3');
  }

  document.getElementById('edVth').value = exp.v_threshold ?? 2.0;
  document.getElementById('edWin').value = exp.custom_window_ms || '';
  document.getElementById('edRecs').value = exp.recognitions ?? 0;
  document.getElementById('edConsolidated').checked = exp.consolidated !== false;
  document.getElementById('btnSaveFim').disabled = false;
  document.getElementById('btnDelFim').disabled = false;
  toggleB2Slot();
  renderExplanation(exp);
}

function collectEditorPayload() {
  const lvl = Number(document.getElementById('edLevel').value);
  const mode = Number(document.getElementById('edSynMode').value);
  const s1 = document.getElementById('edSrc1').value;
  const s2 = document.getElementById('edSrc2').value;
  const s3 = document.getElementById('edSrc3').value;
  const c1 = Number(document.getElementById('edCnt1').value);
  const c2 = Number(document.getElementById('edCnt2').value);
  const c3 = Number(document.getElementById('edCnt3').value);
  const w1 = Number(document.getElementById('edWgt1').value);
  const w2 = Number(document.getElementById('edWgt2').value);
  const w3 = Number(document.getElementById('edWgt3').value);
  const winRaw = document.getElementById('edWin').value.trim();

  const sources = mode === 2 ? [s1, s3] : [s1, s2, s3];
  const synapse_slots = mode === 2 ? [0, 2] : [0, 1, 2];
  const synapse_counts = mode === 2 ? [c1, c3] : [c1, c2, c3];
  const synapse_weights = mode === 2 ? [w1, w3] : [w1, w2, w3];

  return {
    level: lvl,
    fim_id: currentEditedFim.fim_id,
    sources,
    synapse_slots,
    synapse_counts,
    synapse_weights,
    v_threshold: Number(document.getElementById('edVth').value),
    custom_window_ms: winRaw ? Number(winRaw) : null,
    recognitions: Number(document.getElementById('edRecs').value),
    consolidated: document.getElementById('edConsolidated').checked,
    probe_text: document.getElementById('inputText').value,
  };
}

async function saveCurrentFim() {
  if (currentEditedFim.fim_id === null) return;
  const payload = collectEditorPayload();
  const res = await fetch('/api/fim/update', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) {
    showMsg('fimEditStatus', data.error || 'Ошибка обновления ФИМ', true);
    return;
  }
  loadFimIntoEditor(data.fim);
  showMsg('fimEditStatus', `ФИМ L${data.fim.level}#${data.fim.fim_id} сохранён. Осциллограф пересчитан.`);
  await sendText(false, data.fim.level, data.fim.fim_id);
  await searchCurrentLevel();
}

async function createNewFim() {
  const payload = collectEditorPayload();
  const res = await fetch('/api/fim/create', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) {
    showMsg('fimEditStatus', data.error || 'Ошибка создания ФИМ', true);
    return;
  }
  loadFimIntoEditor(data.fim);
  if (data.config) applyConfigToUI(data.config);
  showMsg('fimEditStatus', `Создан ФИМ L${data.fim.level}#${data.fim.fim_id} «${data.fim.text}»`);
  await inspectLevel(data.fim.level);
  await sendText(false, data.fim.level, data.fim.fim_id);
}

async function disableCurrentFim() {
  if (currentEditedFim.fim_id === null) return;
  const res = await fetch('/api/fim/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(currentEditedFim)
  });
  const data = await res.json();
  if (!res.ok) {
    showMsg('fimEditStatus', data.error || 'Ошибка отключения', true);
    return;
  }
  showMsg('fimEditStatus', `ФИМ L${currentEditedFim.level}#${currentEditedFim.fim_id} отключён.`);
  await loadConfig();
  await searchCurrentLevel();
  await sendText(false);
}

async function inspectFim(lvl, fid) {
  if (lvl < 1) return;
  const probe = document.getElementById('inputText').value || '';
  const res = await fetch(`/api/fim?lvl=${lvl}&id=${fid}&probe=${encodeURIComponent(probe)}`);
  const exp = await res.json();
  loadFimIntoEditor(exp);
  if (exp.simulation) {
    loadSimulation(exp.simulation, false);
  }
  if (lvl !== currentMemLevel) {
    inspectLevel(lvl);
  } else {
    searchCurrentLevel();
  }
}

function renderExplanation(exp) {
  if (!exp) return;
  document.getElementById('edStageMeta').innerHTML =
    `Этап: <b>${esc(exp.training_stage || '—')}</b> · Окно: <b>${exp.window_ms} мс</b> · t_созд: <b>${exp.created_at_ms} мс</b>` +
    (exp.training_context ? `<br>Контекст: <span class="mono">«${esc(exp.training_context)}»</span>` : '');

  const parentsHtml = (exp.parents && exp.parents.length)
    ? '<div style="font-weight:700; font-size:11px; margin-bottom:3px;">Исходящие синапсы вверх (L' + (exp.level + 1) + '):</div>' +
      exp.parents.map(p =>
        `<span class="tag hit" style="cursor:pointer;" onclick="inspectFim(${p.level}, ${p.fim_id})">L${p.level}#${p.fim_id} «${esc(p.text)}» (${p.recognitions})</span>`
      ).join('')
    : '<span class="muted">Нет исходящих синапсов на уровень выше</span>';
  document.getElementById('parentsBlock').innerHTML = parentsHtml;

  function buildTree(node, isRoot) {
    const slotTag = node.slot ? `<span class="tag">${esc(node.slot)}</span>` : '';
    const synMeta = node.synapse_count ? `<span class="muted mono">[${node.synapse_count}шт, w=${node.weight}]</span>` : '';
    const clickAttr = node.level > 0 ? `onclick="inspectFim(${node.level}, ${node.fim_id})"` : '';
    let inner = `<div class="${isRoot ? 'tree-root' : 'tree-node'}">
      <div class="tree-card" ${clickAttr}>
        ${slotTag}<b class="mono">L${node.level}#${node.fim_id}</b> <span class="mono">«${esc(node.text)}»</span> ${synMeta}
      </div>`;
    if (node.children && node.children.length) {
      inner += node.children.map(ch => buildTree(ch, false)).join('');
    }
    inner += '</div>';
    return inner;
  }
  document.getElementById('treeBlock').innerHTML =
    '<div style="font-weight:700; font-size:11px; margin-bottom:3px;">Дерево входов вниз до L0:</div>' + buildTree(exp, true);
}

// --- Simulation & Trace Table ---
async function sendText(learn, focusLvl=null, focusFid=null) {
  const t0 = performance.now();
  const text = document.getElementById('inputText').value;
  const res = await fetch('/api/check', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, learn, focus_level: focusLvl, focus_fim_id: focusFid})
  });
  const data = await res.json();
  const dt = (performance.now() - t0).toFixed(1);
  document.getElementById('latencyBadge').textContent = `${dt} мс · Вход: «${data.clean_text || '—'}»`;

  const ansMain = document.getElementById('topAnswerMain');
  const ansMeta = document.getElementById('topAnswerMeta');
  if (!data.top_level) {
    ansMain.textContent = 'Нет сработавших ФИМ (можно обучить)';
    ansMeta.innerHTML = '';
  } else {
    ansMain.textContent = `«${data.top_response_text}»`;
    ansMeta.innerHTML = `Ур. ${data.top_level}: ` + data.top_fims.map(f => {
      const conf = (f.confidence && f.confidence < 0.99) ? ` [2/3: ${f.confidence}]` : '';
      return `<span class="tag hit" style="cursor:pointer;" onclick="inspectFim(${data.top_level}, ${f.fim_id})">L${data.top_level}#${f.fim_id} «${esc(f.text)}»${conf}</span>`;
    }).join('');
  }

  const recMain = document.getElementById('recallMain');
  const recMeta = document.getElementById('recallMeta');
  if (data.recalled_level > data.top_level && data.recalled_text) {
    recMain.style.color = '#157347';
    recMain.textContent = `«${data.recalled_text}»`;
    recMeta.innerHTML = `<span class="tag hit" style="cursor:pointer;" onclick="inspectFim(${data.recalled_level}, ${data.recalled_fim_id})">L${data.recalled_level}#${data.recalled_fim_id}</span>`;
  } else {
    recMain.style.color = '#556070';
    recMain.textContent = 'Совпадает с верхним узнанным уровнем';
    recMeta.innerHTML = '';
  }

  if (data.config) applyConfigToUI(data.config);
  const expToLoad = data.recalled_explanation || data.top_explanation;
  if (expToLoad && focusLvl === null) {
    loadFimIntoEditor(expToLoad);
  }
  if (data.simulation) {
    loadSimulation(data.simulation, true);
  }
}

function loadSimulation(sim, resetTime=true) {
  currentSim = sim;
  const slider = document.getElementById('timeSlider');
  slider.max = sim.duration_ms;
  slider.step = sim.dt_ms;
  if (resetTime) {
    stopPlay();
    playheadMs = sim.duration_ms;
  } else {
    playheadMs = Math.min(playheadMs, sim.duration_ms);
  }
  slider.value = playheadMs;
  document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;

  if (sim.target_fim) {
    selectedSpikeKey = `${sim.target_fim.level}:${sim.target_fim.fim_id}`;
    document.getElementById('dagFocusLabel').textContent =
      `Фокус: L${sim.target_fim.level}#${sim.target_fim.fim_id} «${sim.target_fim.text}» | V_th=${Number(sim.target_fim.v_threshold).toFixed(2)} | W=${sim.target_fim.window_ms} мс`;
  } else {
    selectedSpikeKey = null;
    document.getElementById('dagFocusLabel').textContent = 'Нет активного модуля';
  }
  document.getElementById('traceCount').textContent = (sim.fim_spikes || []).length;
  renderTraceTable();
  drawNengoAll();
}

function renderTraceTable() {
  const tbody = document.getElementById('traceBody');
  if (!currentSim || !currentSim.fim_spikes || !currentSim.fim_spikes.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">Нет сработавших последовательностей.</td></tr>';
    return;
  }
  const q = (document.getElementById('traceFilterInput').value || '').trim().toLowerCase();
  const rows = currentSim.fim_spikes.filter(sp => {
    if (currentTraceFilterLvl > 0 && sp.level !== currentTraceFilterLvl) return false;
    if (q && !sp.text.toLowerCase().includes(q) && !('#' + sp.fim_id).includes(q)) return false;
    return true;
  });
  tbody.innerHTML = rows.map(sp => {
    const key = `${sp.level}:${sp.fim_id}`;
    const isSel = key === selectedSpikeKey;
    const isAssoc = sp.confidence < 0.99;
    const synHtml = (sp.sources_detail || []).map(sd => {
      const cls = sd.fired ? 'hit' : 'miss';
      const prefix = sd.child_level === 0 ? 'L0' : `L${sd.child_level}#${sd.child_id}`;
      return `<span class="tag ${cls}">${sd.neuron}:${prefix} «${esc(sd.child_text)}» (${sd.synapse_count}шт·w=${sd.weight})</span>`;
    }).join(' ');
    const modeTag = isAssoc
      ? `<span class="tag assoc">2-из-3 (${sp.confidence})</span>`
      : `<span class="tag hit">${sp.synapse_slots.length}-из-${sp.synapse_slots.length} (1.0)</span>`;
    return `<tr class="clickable ${isSel ? 'selected-row' : ''}" onclick="inspectFim(${sp.level}, ${sp.fim_id})">
      <td class="mono">${sp.time_ms}</td>
      <td class="mono"><b>L${sp.level}#${sp.fim_id}</b></td>
      <td class="mono"><b>«${esc(sp.text)}»</b></td>
      <td>${synHtml}</td>
      <td class="mono">${sp.window_ms}мс / ${Number(sp.v_threshold).toFixed(2)}</td>
      <td>${modeTag}</td>
      <td><button onclick="event.stopPropagation(); inspectFim(${sp.level}, ${sp.fim_id})">Настроить</button></td>
    </tr>`;
  }).join('');
}

// --- Transport Playback ---
function resetPlayhead() {
  stopPlay();
  playheadMs = 0;
  document.getElementById('timeSlider').value = 0;
  document.getElementById('clockLabel').textContent = 't = 0 мс';
  drawNengoAll();
}
function stepPlayhead(dir) {
  if (!currentSim) return;
  stopPlay();
  playheadMs = Math.max(0, Math.min(currentSim.duration_ms, playheadMs + dir * currentSim.tick_ms));
  document.getElementById('timeSlider').value = playheadMs;
  document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;
  drawNengoAll();
}
function scrubTime(val) {
  playheadMs = Number(val);
  document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;
  drawNengoAll();
}
function togglePlay() {
  if (!currentSim) return;
  if (isPlaying) { stopPlay(); return; }
  if (playheadMs >= currentSim.duration_ms) playheadMs = 0;
  isPlaying = true;
  document.getElementById('playPauseBtn').textContent = '|| Пауза';
  playTimer = setInterval(() => {
    const spd = Number(document.getElementById('speedSelect').value || 0.5);
    playheadMs = Math.min(currentSim.duration_ms, playheadMs + Math.round(currentSim.dt_ms * Math.max(1, spd * 2)));
    document.getElementById('timeSlider').value = playheadMs;
    document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;
    drawNengoAll();
    if (playheadMs >= currentSim.duration_ms) stopPlay();
  }, 55);
}
function stopPlay() {
  isPlaying = false;
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  document.getElementById('playPauseBtn').textContent = '▶ Пуск';
}

// --- Classic Nengo GUI Canvas Renderers (White scientific surfaces) ---
function setupCanvas(id) {
  const c = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 500;
  const H = Number(c.getAttribute('height')) || 220;
  c.width = Math.round(W * dpr);
  c.height = Math.round(H * dpr);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W, H, c };
}

let dagHitboxes = [];
document.getElementById('dagCanvas').addEventListener('click', (e) => {
  const rect = e.target.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  for (const hb of dagHitboxes) {
    if (x >= hb.x && x <= hb.x + hb.w && y >= hb.y && y <= hb.y + hb.h) {
      inspectFim(hb.level, hb.fim_id);
      return;
    }
  }
});

function drawNengoAll() {
  if (!currentSim) return;
  drawDag();
  drawRaster();
  drawOscilloscope();
}

function drawDag() {
  const { ctx, W, H } = setupCanvas('dagCanvas');
  dagHitboxes = [];
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Nengo background engineering grid
  ctx.strokeStyle = '#eef1f6';
  ctx.lineWidth = 1;
  for (let gx = 20; gx < W; gx += 20) {
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
  }
  for (let gy = 20; gy < H; gy += 20) {
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
  }

  const tf = currentSim.target_fim;
  if (!tf) {
    ctx.fillStyle = '#556070';
    ctx.font = '12px Tahoma, sans-serif';
    ctx.fillText('Нет активного модуля ФИМ для построения схемы ансамбля Nengo.', 16, 30);
    return;
  }

  // Left-to-right classic Nengo Network schematic for target FIM:
  // [Sources L-1] --> [Base Ensembles Б1, Б2, Б3] --> [Head Ensemble Г (V_th)]
  const srcs = tf.sources || [];
  const headX = Math.max(W - 210, 420);
  const headY = H / 2 - 36;
  const bnX = Math.max(headX * 0.52, 230);
  const srcX = 24;

  // Draw Head Ensemble Г
  ctx.fillStyle = '#fff9db';
  ctx.strokeStyle = '#2d3748';
  ctx.lineWidth = 1.6;
  ctx.fillRect(headX, headY, 175, 72);
  ctx.strokeRect(headX, headY, 175, 72);
  ctx.fillStyle = '#181c22';
  ctx.font = 'bold 11.5px "DejaVu Sans Mono", monospace';
  ctx.fillText(`Г: L${tf.level}#${tf.fim_id}`, headX + 8, headY + 18);
  ctx.fillStyle = '#1d5fbf';
  ctx.fillText(`«${tf.text.slice(0, 18)}»`, headX + 8, headY + 36);
  ctx.fillStyle = '#556070';
  ctx.font = '10.5px "DejaVu Sans Mono", monospace';
  ctx.fillText(`V_th=${Number(tf.v_threshold).toFixed(2)} | W=${tf.window_ms}мс`, headX + 8, headY + 56);
  dagHitboxes.push({ x: headX, y: headY, w: 175, h: 72, level: tf.level, fim_id: tf.fim_id });

  // Active spike lookup at playhead
  const activeHead = (currentSim.voltage_traces.spike_times_head || []).some(t => Math.abs(t - playheadMs) <= 24);
  if (activeHead) {
    ctx.strokeStyle = '#157347';
    ctx.lineWidth = 3;
    ctx.strokeRect(headX - 2, headY - 2, 179, 76);
  }

  const slotYs = srcs.length === 2 ? [H * 0.28, H * 0.72] : [H * 0.20, H * 0.50, H * 0.80];

  srcs.forEach((s, idx) => {
    const cy = slotYs[idx];
    const srcW = 160, srcH = 40;
    const bnW = 128, bnH = 44;
    const sy = cy - srcH / 2;
    const by = cy - bnH / 2;

    // Source box (level - 1)
    ctx.fillStyle = s.level === 0 ? '#f1f5f9' : '#edf4ff';
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.2;
    ctx.fillRect(srcX, sy, srcW, srcH);
    ctx.strokeRect(srcX, sy, srcW, srcH);
    ctx.fillStyle = '#181c22';
    ctx.font = 'bold 11px "DejaVu Sans Mono", monospace';
    ctx.fillText(`L${s.level}#${s.node_id} «${s.text.slice(0, 12)}»`, srcX + 7, sy + 17);
    ctx.fillStyle = '#556070';
    ctx.font = '10px Tahoma, sans-serif';
    ctx.fillText(s.level === 0 ? 'Вход L0 (элемент)' : 'Клик -> открыть ФИМ', srcX + 7, sy + 32);
    if (s.level > 0) dagHitboxes.push({ x: srcX, y: sy, w: srcW, h: srcH, level: s.level, fim_id: s.node_id });

    // Arrow Source -> BN
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(srcX + srcW, cy);
    ctx.lineTo(bnX, cy);
    ctx.stroke();

    // Base Neuron ensemble Б1/Б2/Б3
    ctx.fillStyle = '#f8fafc';
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1.4;
    ctx.fillRect(bnX, by, bnW, bnH);
    ctx.strokeRect(bnX, by, bnW, bnH);
    ctx.fillStyle = '#181c22';
    ctx.font = 'bold 11px "DejaVu Sans Mono", monospace';
    ctx.fillText(`${s.neuron} (слот ${s.slot})`, bnX + 7, by + 17);
    ctx.fillStyle = '#157347';
    ctx.font = '10.5px "DejaVu Sans Mono", monospace';
    ctx.fillText(`${s.synapse_count}шт · w=${s.weight}`, bnX + 7, by + 34);

    // Synapse connection BN -> Г with bundle line thickness
    ctx.strokeStyle = '#1d5fbf';
    ctx.lineWidth = Math.min(4.5, 1.2 + 0.5 * Number(s.synapse_count || 1));
    ctx.beginPath();
    ctx.moveTo(bnX + bnW, cy);
    ctx.lineTo(headX, headY + 36);
    ctx.stroke();

    // Synapse label on wire
    const mx = (bnX + bnW + headX) / 2 - 22;
    const my = (cy + headY + 36) / 2 - 4;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(mx - 2, my - 10, 56, 13);
    ctx.fillStyle = '#1d5fbf';
    ctx.font = '10px "DejaVu Sans Mono", monospace';
    ctx.fillText(`ΔV=${s.drive}`, mx, my);
  });
}

function drawRaster() {
  const { ctx, W, H } = setupCanvas('rasterCanvas');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const dur = Math.max(currentSim.duration_ms, 252);
  const padL = 36, padR = 12, padT = 14, padB = 22;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xOf = t => padL + (t / dur) * plotW;

  // Axes
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  // Y bands: L0..L9 (10 rows)
  for (let l = 0; l <= 9; l++) {
    const y = padT + ((9 - l) + 0.5) * (plotH / 10);
    ctx.strokeStyle = '#eef1f6';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#475569';
    ctx.font = '10px "DejaVu Sans Mono", monospace';
    ctx.fillText('L' + l, 6, y + 3);
  }

  // Instinct cycle dashed lines
  ctx.strokeStyle = '#d97706';
  ctx.setLineDash([3, 3]);
  (currentSim.instinct_pulses_ms || []).forEach(tp => {
    const x = xOf(tp);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
  });
  ctx.setLineDash([]);

  // Element spikes on L0
  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 1.5;
  (currentSim.element_spikes || []).forEach(es => {
    if (es.time_ms > playheadMs) return;
    const x = xOf(es.time_ms);
    const y = padT + 9.5 * (plotH / 10);
    ctx.beginPath(); ctx.moveTo(x, y - 6); ctx.lineTo(x, y + 6); ctx.stroke();
  });

  // FIM spikes L1..L9
  (currentSim.fim_spikes || []).forEach(sp => {
    if (sp.time_ms > playheadMs) return;
    const x = xOf(sp.time_ms);
    const y = padT + ((9 - sp.level) + 0.5) * (plotH / 10);
    ctx.strokeStyle = sp.confidence < 0.99 ? '#d97706' : '#1d5fbf';
    ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7); ctx.stroke();
  });

  // Playhead cursor
  ctx.strokeStyle = '#dc2626';
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(xOf(playheadMs), padT); ctx.lineTo(xOf(playheadMs), padT + plotH); ctx.stroke();

  ctx.fillStyle = '#334155';
  ctx.font = '10px "DejaVu Sans Mono", monospace';
  ctx.fillText('0 мс', padL, H - 6);
  ctx.fillText(dur + ' мс', W - 52, H - 6);
  document.getElementById('rasterMeta').textContent =
    `L0: ${(currentSim.element_spikes || []).length} | ФИМ: ${(currentSim.fim_spikes || []).length}`;
}

function drawOscilloscope() {
  const { ctx, W, H } = setupCanvas('oscCanvas');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const tr = currentSim.voltage_traces;
  if (!tr || !tr.t_grid || !tr.t_grid.length) return;
  const dur = Math.max(currentSim.duration_ms, 252);
  const padL = 36, padR = 12, padT = 16, padB = 22;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const yMax = Math.max(3.6, tr.v_threshold + 0.8, ...tr.v_head);
  const xOf = t => padL + (t / dur) * plotW;
  const yOf = v => padT + plotH - (v / yMax) * plotH;

  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  // Threshold V_th dashed line
  const yTh = yOf(tr.v_threshold);
  ctx.strokeStyle = '#dc2626';
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(padL, yTh); ctx.lineTo(W - padR, yTh); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#dc2626';
  ctx.font = '10px "DejaVu Sans Mono", monospace';
  ctx.fillText('Vth=' + tr.v_threshold.toFixed(2), padL + 4, yTh - 3);

  let maxStep = 0;
  for (let i = 0; i < tr.t_grid.length; i++) {
    if (tr.t_grid[i] <= playheadMs) maxStep = i;
  }

  const curves = [
    { arr: tr.v_b1, color: '#0284c7', width: 1.3 },
    { arr: tr.v_b2, color: '#9333ea', width: 1.3 },
    { arr: tr.v_b3, color: '#d97706', width: 1.3 },
    { arr: tr.v_head, color: '#15803d', width: 2.2 },
  ];
  for (const c of curves) {
    ctx.strokeStyle = c.color;
    ctx.lineWidth = c.width;
    ctx.beginPath();
    for (let i = 0; i <= maxStep; i++) {
      const x = xOf(tr.t_grid[i]), y = yOf(c.arr[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Cursor
  ctx.strokeStyle = '#181c22';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(xOf(playheadMs), padT); ctx.lineTo(xOf(playheadMs), padT + plotH); ctx.stroke();

  // Legend
  ctx.font = '10px "DejaVu Sans Mono", monospace';
  ctx.fillStyle = '#0284c7'; ctx.fillText('Б1', W - 130, 12);
  ctx.fillStyle = '#9333ea'; ctx.fillText('Б2', W - 102, 12);
  ctx.fillStyle = '#d97706'; ctx.fillText('Б3', W - 74, 12);
  ctx.fillStyle = '#15803d'; ctx.fillText('Г(sum)', W - 46, 12);

  const b1 = (tr.v_b1[maxStep] || 0).toFixed(2);
  const b2 = (tr.v_b2[maxStep] || 0).toFixed(2);
  const b3 = (tr.v_b3[maxStep] || 0).toFixed(2);
  const vh = (tr.v_head[maxStep] || 0).toFixed(2);
  document.getElementById('oscReadout').textContent =
    `Б1=${b1} Б2=${b2} Б3=${b3} | V_Г=${vh}/${tr.v_threshold.toFixed(2)}`;
}

window.addEventListener('resize', () => { if (currentSim) drawNengoAll(); });
loadConfig().then(() => inspectLevel(3)).then(() => setExample('мама раму'));
</script>
</body>
</html>
"""


class FIMHandler(BaseHTTPRequestHandler):
    def _send_nocache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._send_nocache_headers()
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            self._send_json(NET.summary())
            return
        if parsed.path == "/api/config":
            self._send_json(NET.get_config())
            return
        if parsed.path == "/api/level":
            qs = parse_qs(parsed.query)
            lvl = int(qs.get("lvl", ["1"])[0])
            limit = int(qs.get("limit", ["80"])[0])
            query = qs.get("q", [""])[0]
            self._send_json(get_level_fims(lvl, limit=limit, query=query))
            return
        if parsed.path == "/api/fim":
            qs = parse_qs(parsed.query)
            lvl = int(qs.get("lvl", ["1"])[0])
            fid = int(qs.get("id", ["0"])[0])
            probe_txt = qs.get("probe", [""])[0]
            exp = NET.explain_fim(lvl, fid, max_depth=5)
            sim_input = probe_txt if probe_txt.strip() else NET.decode_fim(lvl, fid)
            exp["simulation"] = SIM.run(sim_input, target_level=lvl, target_fim_id=fid, learn=False)
            self._send_json(exp)
            return

        raw = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._send_nocache_headers()
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        req = json.loads(body or "{}")

        try:
            if self.path == "/api/check":
                result = analyze_input(req.get("text", ""), learn=bool(req.get("learn", False)))
                fl = req.get("focus_level")
                ff = req.get("focus_fim_id")
                if fl is not None and ff is not None and result.get("clean_text"):
                    result["simulation"] = SIM.run(
                        result["clean_text"],
                        target_level=int(fl),
                        target_fim_id=int(ff),
                        learn=False,
                    )
                self._send_json(result)
                return

            if self.path == "/api/config":
                updated = NET.update_config(req)
                self._send_json({"config": updated, "summary": NET.summary()})
                return

            if self.path == "/api/fim/update":
                lvl = int(req.get("level", 1))
                fid = int(req.get("fim_id", 0))
                exp = NET.update_fim(lvl, fid, req)
                probe_txt = str(req.get("probe_text") or "").strip() or exp["text"]
                sim = SIM.run(probe_txt, target_level=lvl, target_fim_id=fid, learn=False)
                self._send_json({"fim": exp, "simulation": sim, "summary": NET.summary()})
                return

            if self.path == "/api/fim/create":
                lvl = int(req.get("level", 1))
                exp = NET.create_fim(lvl, req)
                probe_txt = str(req.get("probe_text") or "").strip() or exp["text"]
                sim = SIM.run(probe_txt, target_level=lvl, target_fim_id=exp["fim_id"], learn=False)
                self._send_json({
                    "fim": exp,
                    "simulation": sim,
                    "summary": NET.summary(),
                    "config": NET.get_config(),
                })
                return

            if self.path == "/api/fim/delete":
                lvl = int(req.get("level", 1))
                fid = int(req.get("fim_id", 0))
                info = NET.delete_fim(lvl, fid)
                self._send_json({"deleted": info, "summary": NET.summary()})
                return

        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer((host, port), FIMHandler)
    print(f"ФИМ-интерфейс запущен: http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
