"""
Веб-интерфейс и инженерный стенд ФИМ-Nengo для проверки, трассировки
каждой последовательности, спайковой визуализации и инспекции памяти (до 10 млн ФИМ).

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
  <title>ФИМ-Nengo Studio — Спайковый симулятор и трассировщик (10M)</title>
  <style>
    :root {
      --bg: #070b14;
      --panel: #0d1424;
      --panel-2: #111a2e;
      --border: #1e293b;
      --border-hi: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --cyan: #38bdf8;
      --green: #4ade80;
      --amber: #fbbf24;
      --red: #f87171;
      --violet: #c084fc;
      --mono: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, system-ui, -apple-system, sans-serif;
      font-size: 13px;
      line-height: 1.45;
    }
    .topbar {
      background: #0b1120;
      border-bottom: 1px solid var(--border);
      padding: 10px 18px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      position: sticky;
      top: 0;
      z-index: 30;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 15px;
      letter-spacing: -0.01em;
    }
    .brand-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--cyan);
      box-shadow: 0 0 10px var(--cyan);
    }
    .metrics-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-family: var(--mono);
      font-size: 11.5px;
    }
    .metric-chip {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 9px;
      color: var(--muted);
    }
    .metric-chip b { color: var(--text); }
    .metric-chip.accent b { color: var(--cyan); }
    .metric-chip.green b { color: var(--green); }

    .workspace {
      max-width: 1620px;
      margin: 0 auto;
      padding: 14px 18px 36px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }
    .panel-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      padding-bottom: 7px;
      border-bottom: 1px solid var(--border);
    }
    .panel-title {
      font-size: 12.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--cyan);
    }
    .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }

    textarea, input[type="text"] {
      background: #060911;
      color: var(--text);
      border: 1px solid var(--border-hi);
      border-radius: 6px;
      padding: 8px 10px;
      font-family: var(--mono);
      font-size: 13px;
    }
    textarea { width: 100%; min-height: 54px; resize: vertical; }
    textarea:focus, input[type="text"]:focus {
      outline: none;
      border-color: var(--cyan);
    }
    button {
      background: #0284c7;
      color: #fff;
      border: 1px solid #0369a1;
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: #0ea5e9; }
    button.sec {
      background: #0f172a;
      color: var(--text);
      border: 1px solid var(--border-hi);
    }
    button.sec:hover { border-color: var(--cyan); color: var(--cyan); }
    button.sec.active {
      background: #0c4a6e;
      border-color: var(--cyan);
      color: #fff;
    }

    .transport {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      background: #060911;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 10px;
      margin-top: 8px;
      font-family: var(--mono);
      font-size: 12px;
    }
    .transport button { padding: 4px 10px; font-size: 12px; }
    .clock-pill {
      background: #0f172a;
      border: 1px solid var(--border-hi);
      border-radius: 5px;
      padding: 3px 8px;
      color: var(--cyan);
    }

    .readout-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    @media (max-width: 960px) { .readout-grid { grid-template-columns: 1fr; } }
    .readout-box {
      background: #060911;
      border: 1px solid var(--border);
      border-left: 3px solid var(--cyan);
      border-radius: 6px;
      padding: 10px 12px;
    }
    .readout-box.recall { border-left-color: var(--green); }
    .readout-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
    .readout-main {
      font-size: 17px;
      font-weight: 700;
      color: #f8fafc;
      margin: 4px 0;
      word-break: break-word;
    }

    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    @media (max-width: 1080px) { .two-col { grid-template-columns: 1fr; } }

    .canvas-wrap {
      background: #05080f;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px;
    }
    .canvas-head {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      margin-bottom: 6px;
      font-size: 12px;
      font-weight: 600;
      color: #cbd5e1;
    }
    canvas {
      width: 100%;
      display: block;
      background: #03060c;
      border-radius: 4px;
      cursor: crosshair;
    }
    .legend { font-size: 11px; color: var(--muted); display: flex; gap: 10px; flex-wrap: wrap; font-family: var(--mono); }

    .trace-scroll {
      max-height: 350px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #060911;
    }
    table.trace-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    table.trace-table th {
      position: sticky;
      top: 0;
      background: #0b1120;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      text-align: left;
      padding: 7px 9px;
      border-bottom: 1px solid var(--border-hi);
      z-index: 2;
    }
    table.trace-table td {
      padding: 6px 9px;
      border-bottom: 1px solid #111827;
      vertical-align: top;
    }
    table.trace-table tr { cursor: pointer; }
    table.trace-table tbody tr:hover { background: #101b33; }
    table.trace-table tbody tr.selected {
      background: #0c2749;
      outline: 1px solid var(--cyan);
    }
    table.trace-table tbody tr.future { opacity: 0.42; }

    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 600;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .badge.rec { background: rgba(74,222,128,0.12); color: var(--green); border-color: rgba(74,222,128,0.35); }
    .badge.p23 { background: rgba(251,191,36,0.12); color: var(--amber); border-color: rgba(251,191,36,0.35); }
    .badge.cons { background: rgba(56,189,248,0.12); color: var(--cyan); border-color: rgba(56,189,248,0.35); }
    .badge.inst { background: rgba(248,113,113,0.15); color: var(--red); border-color: rgba(248,113,113,0.4); }

    .slot-chip {
      display: inline-block;
      padding: 2px 6px;
      margin: 1px 4px 1px 0;
      border-radius: 4px;
      font-family: var(--mono);
      font-size: 11px;
      border: 1px solid #1e293b;
      background: #0b1220;
      color: #cbd5e1;
      cursor: pointer;
    }
    .slot-chip.hit { border-color: rgba(56,189,248,0.45); color: #bae6fd; }
    .slot-chip.miss { border-color: rgba(251,191,36,0.35); color: #94a3b8; text-decoration: line-through; }
    .slot-chip:hover { border-color: var(--cyan); }

    .lvl-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 7px;
      margin-bottom: 10px;
    }
    .lvl-card {
      background: #060911;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 9px;
      cursor: pointer;
      font-family: var(--mono);
      font-size: 11.5px;
    }
    .lvl-card:hover, .lvl-card.active { border-color: var(--cyan); background: #0c172b; }
    .lvl-card .lvl-head { font-weight: 700; color: var(--cyan); margin-bottom: 3px; display: flex; justify-content: space-between; }

    .tree-box {
      max-height: 430px;
      overflow: auto;
      background: #060911;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
    }
    .tree-node {
      margin-left: 16px;
      padding-left: 10px;
      border-left: 1px solid #334155;
      margin-top: 5px;
      font-size: 12px;
    }
    .tree-head {
      background: #0d1526;
      border: 1px solid #263449;
      border-radius: 5px;
      padding: 4px 8px;
      display: inline-block;
      cursor: pointer;
    }
    .tree-head:hover { border-color: var(--cyan); }

    .fim-list {
      max-height: 320px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .fim-item {
      background: #060911;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 9px;
      cursor: pointer;
      font-size: 12px;
    }
    .fim-item:hover { border-color: var(--cyan); background: #0d172a; }
    .mono { font-family: var(--mono); }
    .muted { color: var(--muted); }
  </style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <span class="brand-dot"></span>
    <span>ФИМ-Nengo Studio · Иерархическая спайковая сеть (L0–L9)</span>
  </div>
  <div class="metrics-strip">
    <div class="metric-chip accent">Ёмкость пула: <b id="hdrCap">1 000 000 ФИМ / ур. (10M сеть)</b></div>
    <div class="metric-chip green">Консолидировано: <b id="hdrTotalCons">—</b></div>
    <div class="metric-chip">Узнаваний в базе: <b id="hdrTotalRec">—</b></div>
    <div class="metric-chip">Такт L0: <b>84 мс</b> · Инстинкт: <b>252 мс</b> · Порог: <b>V_th = 2.0</b></div>
  </div>
</div>

<div class="workspace">

  <!-- Блок ввода и Nengo-транспорта -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">Входной поток (Уровень 0: 41 элемент) &amp; Транспорт симуляции Nengo</div>
      <div class="row">
        <button class="sec" onclick="setExample('привет')">привет</button>
        <button class="sec" onclick="setExample('ну привет')">ну привет</button>
        <button class="sec" onclick="setExample('мама мыла раму')">мама мыла раму</button>
        <button class="sec" onclick="setExample('мама раму')">мама раму (2-из-3)</button>
        <button class="sec" onclick="setExample('дом у берега')">дом у берега</button>
        <button class="sec" onclick="setExample('Только Иван выразил готовность помочь.')">RuCoLA #1</button>
        <button class="sec" onclick="setExample('На поверку вся теория оказалась полной чепухой.')">RuCoLA #2</button>
      </div>
    </div>
    <textarea id="inputText" placeholder="Введите фразу или предложение для потактовой трассировки спайков..."></textarea>
    <div class="row" style="margin-top:8px;">
      <button onclick="sendText(false)">▶ Проверить и трассировать (Read-Only)</button>
      <button class="sec" onclick="sendText(true)">+ Подать в память и дообучить</button>
      <span id="latencyBadge" class="mono muted" style="font-size:11.5px;"></span>
    </div>

    <div class="transport">
      <button class="sec" onclick="simSeek(0)" title="В начало">⏮</button>
      <button class="sec" onclick="simStep(-84)" title="Назад на 1 такт (-84 мс)">⏪ -84мс</button>
      <button onclick="simPlay()">▶ Пуск</button>
      <button class="sec" onclick="simPause()">⏸ Пауза</button>
      <button class="sec" onclick="simStep(84)" title="Вперёд на 1 такт (+84 мс)">⏩ +84мс</button>
      <button class="sec" onclick="simEnd()" title="В конец">⏭ В конец</button>
      <select id="simSpeed" style="background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:5px;padding:4px 6px;font-family:var(--mono);font-size:11.5px;">
        <option value="0.35">0.35x</option>
        <option value="1" selected>1.0x</option>
        <option value="2">2.0x</option>
      </select>
      <input type="range" id="simScrub" min="0" max="504" step="12" value="0" style="flex:1;min-width:160px;" oninput="onScrub(this.value)" />
      <span class="clock-pill" id="simClock">t = 0 / 0 мс</span>
      <span class="clock-pill" id="simTicks">Такт L0: 0 · Инстинкт: 0</span>
    </div>
  </div>

  <!-- Ответ системы (прямой вывод и подъём) -->
  <div class="readout-grid">
    <div class="readout-box">
      <div class="readout-label">Максимально узнанная конструкция на входе</div>
      <div id="topAnswerMain" class="readout-main muted">Подайте текст или выберите пример выше</div>
      <div id="topAnswerMeta" class="mono muted" style="font-size:11.5px;"></div>
    </div>
    <div class="readout-box recall">
      <div class="readout-label">Ассоциативный подъём по исходящим синапсам (верхний контекст)</div>
      <div id="recallMain" class="readout-main muted">—</div>
      <div id="recallMeta" class="mono muted" style="font-size:11.5px;"></div>
    </div>
  </div>

  <!-- 1. Таблица трассировки каждой последовательности -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">1. Хронологическая трассировка каждой последовательности (Б1 + Б2 + Б3 → Г)</div>
      <div class="row">
        <div id="traceFilterBar" class="row"></div>
        <label class="mono muted" style="font-size:11.5px;display:flex;align-items:center;gap:5px;cursor:pointer;">
          <input type="checkbox" id="chkShowAllTimes" checked onchange="renderTraceTable()" /> Показать весь поток (включая будущие шаги)
        </label>
      </div>
    </div>
    <div class="trace-scroll">
      <table class="trace-table">
        <thead>
          <tr>
            <th style="width:110px;">Время / Такт</th>
            <th style="width:105px;">Уровень / Окно</th>
            <th style="width:125px;">Тип срабатывания</th>
            <th style="width:220px;">ФИМ и распознанный текст</th>
            <th>Формула сборки (синапсы Б1, Б2, Б3 от предыдущего уровня)</th>
            <th style="width:190px;">Источник в корпусе</th>
          </tr>
        </thead>
        <tbody id="traceBody">
          <tr><td colspan="6" class="muted" style="padding:14px;">Нет данных симуляции. Запустите проверку текста.</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- 2. Каскадный граф последовательностей (DAG + LIF Ансамбль) -->
  <div class="panel">
    <div class="panel-header">
      <div class="panel-title">2. Причинный граф каскада спайков (L0 → L1 → … → L9) &amp; LIF-ансамбль целевого ФИМ</div>
      <span id="simTargetBadge" class="mono" style="font-size:11.5px;color:#7dd3fc;">Цель осциллографа: —</span>
    </div>
    <div class="canvas-wrap">
      <div class="canvas-head">
        <span>Кликните по любому узлу графа, чтобы переключить осциллограф и дерево синапсов на этот ФИМ</span>
        <span class="legend">
          <span style="color:#f87171;">● Импульс инстинкта 252 мс</span>
          <span style="color:#38bdf8;">● Базовый нейрон Б</span>
          <span style="color:#4ade80;">● Полный спайк (1.0)</span>
          <span style="color:#fbbf24;">● Ассоциация 2-из-3 (0.67)</span>
        </span>
      </div>
      <canvas id="networkCanvas" style="height:290px;"></canvas>
    </div>
  </div>

  <!-- 3. Растр спайков и Осциллограф V(t) -->
  <div class="two-col">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">3А. Многоканальный растр спайков (Инстинкт 252 мс + L0..L9)</div>
      </div>
      <div class="canvas-wrap">
        <canvas id="rasterCanvas" style="height:255px;"></canvas>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">3Б. Осциллограф мембранных потенциалов V(t) целевого ФИМ</div>
        <div id="oscVoltagesReadout" class="mono" style="font-size:11.5px;">V_Б1=0.00 V_Б2=0.00 V_Б3=0.00 | V_Г=0.00 / 2.00</div>
      </div>
      <div class="canvas-wrap">
        <div class="canvas-head">
          <span class="legend">
            <span style="color:#60a5fa;">— V_Б1</span>
            <span style="color:#c084fc;">— V_Б2</span>
            <span style="color:#34d399;">— V_Б3</span>
            <span style="color:#fbbf24;">— V_Г (головной)</span>
            <span style="color:#f87171;">--- V_th = 2.0</span>
          </span>
        </div>
        <canvas id="oscCanvas" style="height:230px;"></canvas>
      </div>
    </div>
  </div>

  <!-- 4. Полное дерево синапсов и Поиск по корпусу (1 000 000 ФИМ / уровень) -->
  <div class="two-col">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">4А. Структурное дерево сборки ФИМ (до букв L0) и исходящие связи</div>
      </div>
      <div id="analyticsBox" class="tree-box muted">Выберите любую строку в таблице трассировки, узел графа или ФИМ в базе памяти.</div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">4Б. Инспектор пула памяти (1 000 000 ФИМ на уровень · Корпус СЧР + RuCoLA)</div>
      </div>
      <div id="summaryGrid" class="lvl-grid"></div>
      <div class="row" style="margin-bottom:8px;">
        <input type="text" id="levelSearchInput" placeholder="Поиск подстроки или #ID на выбранном уровне..." style="flex:1;" onkeydown="if(event.key==='Enter') searchActiveLevel()" />
        <button onclick="searchActiveLevel()">Найти</button>
        <button class="sec" onclick="document.getElementById('levelSearchInput').value=''; searchActiveLevel()">Сброс</button>
      </div>
      <div id="levelBrowser" class="fim-list">
        <div class="muted" style="padding:10px;">Нажмите на карточку уровня (1–9) выше, чтобы просмотреть или найти ФИМы в памяти.</div>
      </div>
    </div>
  </div>

</div>

<script>
let activeLevel = 3;
let currentData = null;
let currentSim = null;
let playheadMs = 0;
let simPlaying = false;
let simRafId = null;
let lastFrameTs = null;
let traceLevelFilter = 0; // 0 = все уровни
let selectedTarget = null; // {level, fim_id}
let graphHitBoxes = []; // кликабельные узлы графа

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function fmtNum(n) {
  return Number(n || 0).toLocaleString('ru-RU');
}

function setupHiDpiCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(300, Math.round(rect.width));
  const h = Math.max(180, Math.round(rect.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, W: w, H: h};
}

function setExample(txt) {
  document.getElementById('inputText').value = txt;
  sendText(false);
}

function renderSummary(summary) {
  let totCons = 0, totRec = 0;
  const grid = document.getElementById('summaryGrid');
  grid.innerHTML = '';
  for (let lvl = 1; lvl <= 9; lvl++) {
    const s = summary[lvl];
    totCons += s.consolidated_fims;
    totRec += s.total_recognitions;
    const div = document.createElement('div');
    div.className = 'lvl-card' + (activeLevel === lvl ? ' active' : '');
    div.onclick = () => inspectLevel(lvl);
    div.innerHTML = `<div class="lvl-head"><span>Ур.${lvl}</span><span>${s.window_ms}мс</span></div>
      <div>ФИМ: <b>${fmtNum(s.consolidated_fims)}</b> / 1M</div>
      <div class="muted">Узн: ${fmtNum(s.total_recognitions)}</div>`;
    grid.appendChild(div);
  }
  document.getElementById('hdrTotalCons').textContent = fmtNum(totCons);
  document.getElementById('hdrTotalRec').textContent = fmtNum(totRec);
}

async function loadSummary() {
  const res = await fetch('/api/summary');
  const data = await res.json();
  renderSummary(data);
}

async function inspectLevel(lvl) {
  activeLevel = lvl;
  loadSummary();
  await searchActiveLevel();
}

async function searchActiveLevel() {
  const q = document.getElementById('levelSearchInput').value.trim();
  const res = await fetch(`/api/level?lvl=${activeLevel}&limit=80&q=${encodeURIComponent(q)}`);
  const data = await res.json();
  const box = document.getElementById('levelBrowser');
  if (!data.fims.length) {
    box.innerHTML = `<div class="muted" style="padding:10px;">На уровне ${activeLevel} по запросу «${esc(q)}» совпадений не найдено (всего на уровне: ${fmtNum(data.total_consolidated)}).</div>`;
    return;
  }
  let html = `<div class="mono muted" style="font-size:11px;padding:2px 4px;">Уровень ${activeLevel}: показано ${data.fims.length} из ${fmtNum(data.matched_count)} (всего в пуле ${fmtNum(data.total_consolidated)} / ${fmtNum(data.max_fims)})</div>`;
  for (const f of data.fims) {
    const preview = f.text.length > 160 ? f.text.slice(0, 160) + '…' : f.text;
    const stageBadge = f.training_stage ? ` · ${esc(f.training_stage)}` : '';
    html += `<div class="fim-item" onclick="inspectFim(${activeLevel}, ${f.fim_id})">
      <div class="mono" style="font-size:11px;color:#7dd3fc;">
        <b>#${f.fim_id}</b> · узнан ${fmtNum(f.recognitions)}x · слоты ${JSON.stringify(f.synapse_slots)} · t=${f.created_at_ms}мс${stageBadge}
      </div>
      <div style="color:#f8fafc;margin-top:2px;">«${esc(preview)}»</div>
    </div>`;
  }
  box.innerHTML = html;
}

function renderTree(node) {
  const slotPrefix = node.slot ? `<b style="color:#38bdf8;">${esc(node.slot)}</b> ← ` : '';
  const slotsStr = node.synapse_slots && node.synapse_slots.length ? ` · слоты ${JSON.stringify(node.synapse_slots)}` : '';
  const clickAttr = node.level > 0 ? `onclick="event.stopPropagation(); inspectFim(${node.level}, ${node.fim_id})"` : '';
  let html = `<div class="tree-node">
    <div class="tree-head" ${clickAttr}>
      ${slotPrefix}<b>L${node.level} #${node.fim_id}</b> «${esc(node.text)}»
      <span class="muted mono" style="font-size:11px;">(${fmtNum(node.recognitions)}x${slotsStr})</span>
    </div>`;
  if (node.children && node.children.length) {
    for (const ch of node.children) html += renderTree(ch);
  }
  html += `</div>`;
  return html;
}

function renderExplanation(exp) {
  if (!exp) return;
  const box = document.getElementById('analyticsBox');
  let parentsHtml = '<span class="muted">Нет исходящих синапсов выше (верхний узел).</span>';
  if (exp.parents && exp.parents.length) {
    parentsHtml = exp.parents.map(p =>
      `<span class="slot-chip hit" onclick="inspectFim(${p.level}, ${p.fim_id})">L${p.level} #${p.fim_id} «${esc(p.text.slice(0, 48))}» (${fmtNum(p.recognitions)}x)</span>`
    ).join('');
  }
  box.innerHTML = `
    <div style="font-size:14px;font-weight:700;color:#7dd3fc;margin-bottom:6px;">
      Уровень ${exp.level}, ФИМ #${exp.fim_id} — «${esc(exp.text)}»
    </div>
    <div class="mono" style="background:#0b1220;border:1px solid #1e293b;border-radius:5px;padding:7px 9px;font-size:11.5px;margin-bottom:8px;">
      <div><b>Этап обучения:</b> ${esc(exp.training_stage || '—')} | <b>Консолидация:</b> ${exp.created_at_ms} мс | <b>Слоты:</b> ${JSON.stringify(exp.synapse_slots)} | <b>Узнаваний:</b> ${fmtNum(exp.recognitions)}</div>
      <div style="margin-top:3px;color:#94a3b8;"><b>Контекст при записи:</b> «${esc(exp.training_context || '—')}»</div>
    </div>
    <div style="margin-bottom:8px;"><b style="font-size:11.5px;color:#cbd5e1;">Куда идёт спайк дальше (Уровень ${exp.level + 1}):</b><div style="margin-top:4px;">${parentsHtml}</div></div>
    <div><b style="font-size:11.5px;color:#cbd5e1;">Иерархия сборки от Б1, Б2, Б3 вниз до Уровня 0:</b>${renderTree(exp)}</div>
  `;
}

async function inspectFim(lvl, fid) {
  if (lvl <= 0) return;
  selectedTarget = {level: lvl, fim_id: fid};
  const currentText = document.getElementById('inputText').value.trim();
  const res = await fetch(`/api/fim?lvl=${lvl}&id=${fid}`);
  const exp = await res.json();
  renderExplanation(exp);
  if (exp.simulation) {
    // Если сейчас нет активного текста или пользователь кликнул внешний ФИМ из базы, показываем его симуляцию
    if (!currentSim || currentText !== (currentSim.clean_text || '')) {
      loadSimulation(exp.simulation, false);
    } else {
      // Иначе фокусируемся на этом ФИМе, если он участвовал, или показываем его собственную симуляцию
      const matched = currentSim.fim_spikes.some(s => s.level === lvl && s.fim_id === fid);
      if (matched) {
        renderTraceTable();
        drawNengoAll();
      } else {
        loadSimulation(exp.simulation, false);
      }
    }
  }
}

function renderTraceFilters() {
  const bar = document.getElementById('traceFilterBar');
  if (!currentSim) { bar.innerHTML = ''; return; }
  const lvls = Array.from(new Set(currentSim.fim_spikes.map(s => s.level))).sort((a,b) => a - b);
  let html = `<button class="sec ${traceLevelFilter === 0 ? 'active' : ''}" style="padding:3px 8px;font-size:11px;" onclick="setTraceFilter(0)">Все (${currentSim.fim_spikes.length})</button>`;
  for (const l of lvls) {
    const cnt = currentSim.fim_spikes.filter(s => s.level === l).length;
    html += `<button class="sec ${traceLevelFilter === l ? 'active' : ''}" style="padding:3px 8px;font-size:11px;" onclick="setTraceFilter(${l})">L${l} (${cnt})</button>`;
  }
  bar.innerHTML = html;
}

function setTraceFilter(lvl) {
  traceLevelFilter = lvl;
  renderTraceFilters();
  renderTraceTable();
}

function renderTraceTable() {
  const tbody = document.getElementById('traceBody');
  if (!currentSim || !currentSim.fim_spikes.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted" style="padding:14px;">Последовательных срабатываний ФИМ на этом входе не обнаружено.</td></tr>`;
    return;
  }
  const showAll = document.getElementById('chkShowAllTimes').checked;
  const rows = currentSim.fim_spikes.filter(sp => {
    if (traceLevelFilter > 0 && sp.level !== traceLevelFilter) return false;
    if (!showAll && sp.time_ms > playheadMs) return false;
    return true;
  });

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted" style="padding:14px;">Нет событий для выбранного фильтра / момента времени t=${Math.round(playheadMs)} мс.</td></tr>`;
    return;
  }

  let html = '';
  for (const sp of rows) {
    const isFuture = sp.time_ms > playheadMs;
    const isSelected = selectedTarget && selectedTarget.level === sp.level && selectedTarget.fim_id === sp.fim_id;
    const isPartial = sp.confidence < 0.99;
    const badgeCls = sp.event_type === 'consolidation' ? 'cons' : (isPartial ? 'p23' : 'rec');
    const badgeText = sp.event_type === 'consolidation'
      ? 'ЗАПИСЬ 1.0'
      : (isPartial ? `2-ИЗ-3 (${sp.confidence})` : 'УЗНАН 1.0');
    const tickNum = Math.round(sp.time_ms / currentSim.tick_ms);
    const instBadge = (sp.time_ms % currentSim.instinct_cycle_ms === 0)
      ? ` <span class="badge inst" title="Импульс осциллятора инстинкта 252 мс">252мс</span>` : '';

    const formulaHtml = (sp.sources_detail || []).map(sd => {
      const cls = sd.fired ? 'hit' : 'miss';
      const mark = sd.fired ? '✓' : '✗';
      const click = sd.child_level > 0 ? `event.stopPropagation(); inspectFim(${sd.child_level}, ${sd.child_id})` : 'event.stopPropagation()';
      return `<span class="slot-chip ${cls}" onclick="${click}" title="Синапс ${sd.neuron} от L${sd.child_level}#${sd.child_id}"><b>${sd.neuron}</b>: L${sd.child_level}#${sd.child_id} «${esc(sd.child_text)}» ${mark}</span>`;
    }).join(' ');

    html += `<tr class="${isSelected ? 'selected' : ''} ${isFuture ? 'future' : ''}" onclick="selectSpikeRow(${sp.level}, ${sp.fim_id}, ${sp.time_ms})">
      <td class="mono"><b>${sp.time_ms} мс</b><div class="muted" style="font-size:10.5px;">такт #${tickNum}${instBadge}</div></td>
      <td class="mono"><b>L${sp.level}</b><div class="muted" style="font-size:10.5px;">W=${sp.window_ms} мс</div></td>
      <td><span class="badge ${badgeCls}">${badgeText}</span></td>
      <td><div class="mono" style="color:#7dd3fc;font-size:11px;">#${sp.fim_id} (${fmtNum(sp.recognitions)}x)</div><div><b>«${esc(sp.text)}»</b></div></td>
      <td>${formulaHtml}</td>
      <td class="muted" style="font-size:11px;">${esc(sp.training_stage || '—')}</td>
    </tr>`;
  }
  tbody.innerHTML = html;
}

async function selectSpikeRow(lvl, fid, tMs) {
  simPause();
  playheadMs = Math.max(playheadMs, tMs);
  updateClockUi();
  await inspectFim(lvl, fid);
}

function updateClockUi() {
  if (!currentSim) return;
  document.getElementById('simScrub').value = Math.round(playheadMs);
  document.getElementById('simClock').textContent = `t = ${Math.round(playheadMs)} / ${currentSim.duration_ms} мс`;
  const tickN = Math.floor(playheadMs / currentSim.tick_ms);
  const instN = Math.floor(playheadMs / currentSim.instinct_cycle_ms);
  document.getElementById('simTicks').textContent = `Такт L0: #${tickN} · Инстинкт: #${instN}`;
}

function loadSimulation(sim, autoPlay = true) {
  currentSim = sim;
  traceLevelFilter = 0;
  if (!sim) {
    playheadMs = 0;
    document.getElementById('simClock').textContent = 't = 0 / 0 мс';
    document.getElementById('simTargetBadge').textContent = 'Цель осциллографа: —';
    renderTraceFilters();
    renderTraceTable();
    drawNengoAll();
    return;
  }
  document.getElementById('simScrub').max = sim.duration_ms;
  const tgt = sim.target_fim;
  if (tgt) {
    selectedTarget = {level: tgt.level, fim_id: tgt.fim_id};
    document.getElementById('simTargetBadge').textContent = `Цель осциллографа: L${tgt.level} #${tgt.fim_id} «${tgt.text.slice(0, 34)}» (окно ${tgt.window_ms} мс)`;
  } else {
    document.getElementById('simTargetBadge').textContent = 'Цель осциллографа: —';
  }
  renderTraceFilters();

  if (autoPlay) {
    playheadMs = 0;
    updateClockUi();
    renderTraceTable();
    simPlay();
  } else {
    simPause();
    playheadMs = sim.duration_ms;
    updateClockUi();
    renderTraceTable();
    drawNengoAll();
  }
}

function simSeek(ms) {
  if (!currentSim) return;
  simPause();
  playheadMs = Math.max(0, Math.min(currentSim.duration_ms, ms));
  updateClockUi();
  renderTraceTable();
  drawNengoAll();
}

function simStep(deltaMs) {
  if (!currentSim) return;
  simSeek(playheadMs + deltaMs);
}

function simPlay() {
  if (!currentSim) return;
  if (playheadMs >= currentSim.duration_ms) playheadMs = 0;
  simPlaying = true;
  lastFrameTs = null;
  if (simRafId) cancelAnimationFrame(simRafId);
  simRafId = requestAnimationFrame(stepPlayback);
}

function simPause() {
  simPlaying = false;
  if (simRafId) cancelAnimationFrame(simRafId);
  simRafId = null;
}

function simEnd() {
  if (!currentSim) return;
  simPause();
  playheadMs = currentSim.duration_ms;
  updateClockUi();
  renderTraceTable();
  drawNengoAll();
}

function onScrub(val) {
  simSeek(Number(val));
}

function stepPlayback(ts) {
  if (!simPlaying || !currentSim) return;
  if (lastFrameTs === null) lastFrameTs = ts;
  const elapsed = ts - lastFrameTs;
  lastFrameTs = ts;
  const mult = Number(document.getElementById('simSpeed').value || 1);
  const speed = Math.max(0.55, currentSim.duration_ms / 1600) * mult;
  playheadMs = Math.min(currentSim.duration_ms, playheadMs + elapsed * speed);
  updateClockUi();
  drawNengoAll();
  if (playheadMs < currentSim.duration_ms) {
    simRafId = requestAnimationFrame(stepPlayback);
  } else {
    simPlaying = false;
    renderTraceTable();
  }
}

function drawNengoAll() {
  drawNetworkGraph();
  drawRasterPlot();
  drawOscilloscope();
}

function drawNetworkGraph() {
  const canvas = document.getElementById('networkCanvas');
  const {ctx, W, H} = setupHiDpiCanvas(canvas);
  ctx.clearRect(0, 0, W, H);
  graphHitBoxes = [];

  if (!currentSim) {
    ctx.fillStyle = '#64748b';
    ctx.font = '13px system-ui';
    ctx.fillText('Подайте фразу или выберите пример выше, чтобы увидеть причинный граф спайков и ансамбль Б1/Б2/Б3 → Г.', 20, H / 2);
    return;
  }

  const ensembleW = 290;
  const dagW = W - ensembleW - 22;

  // Группируем узлы по уровням L0 .. maxLvl (реальные узлы каскада!)
  const maxLvl = Math.max(1, ...currentSim.fim_spikes.map(s => s.level));
  const rowCount = maxLvl + 1;
  const rowGap = Math.min(54, (H - 34) / Math.max(2, rowCount));

  // Индекс позиций узлов (key = `${lvl}:${id}:${t}`) и последняя позиция для каждого (lvl, id)
  const nodePos = {};
  const nodesToDraw = [];

  // L0: входные символы
  const elems = currentSim.element_spikes;
  const maxElemSpan = Math.max(1, elems.length);
  const xPadL = 62, xPadR = 24;
  const availW = Math.max(120, dagW - xPadL - xPadR);
  const yL0 = H - 28;

  for (let i = 0; i < elems.length; i++) {
    const es = elems[i];
    const x = xPadL + ((i + 0.5) / maxElemSpan) * availW;
    const pos = {x, y: yL0, lvl: 0, fim_id: es.node_id, time_ms: es.time_ms, text: es.char, confidence: 1.0};
    nodePos[`0:${es.node_id}:${es.time_ms}`] = pos;
    nodePos[`0:${es.node_id}:last`] = pos;
    nodesToDraw.push(pos);
  }

  // L1..maxLvl: реальные спайки ФИМ по времени
  const dur = Math.max(1, currentSim.duration_ms);
  for (const sp of currentSim.fim_spikes) {
    const y = yL0 - sp.level * rowGap;
    const x = xPadL + Math.min(0.97, Math.max(0.04, sp.time_ms / dur)) * availW;
    const pos = {
      x, y, lvl: sp.level, fim_id: sp.fim_id, time_ms: sp.time_ms,
      text: sp.text, confidence: sp.confidence, sources_detail: sp.sources_detail || [],
      event_type: sp.event_type,
    };
    nodePos[`${sp.level}:${sp.fim_id}:${sp.time_ms}`] = pos;
    nodePos[`${sp.level}:${sp.fim_id}:last`] = pos;
    nodesToDraw.push(pos);
  }

  // Подписи уровней слева в DAG
  ctx.font = 'bold 10.5px monospace';
  for (let l = 0; l <= maxLvl; l++) {
    const y = yL0 - l * rowGap;
    ctx.strokeStyle = 'rgba(30,41,59,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(46, y);
    ctx.lineTo(dagW - 8, y);
    ctx.stroke();
    ctx.fillStyle = l === 0 ? '#38bdf8' : '#94a3b8';
    ctx.fillText(`L${l}`, 12, y + 4);
  }

  // Линии синаптической связи от дочерних узлов к родительским ФИМ
  for (const node of nodesToDraw) {
    if (node.lvl === 0) continue;
    const activeNode = node.time_ms <= playheadMs;
    const partial = node.confidence < 0.99;
    for (const sd of node.sources_detail) {
      // Ищем ближайший по времени <= node.time_ms дочерний узел
      let childPos = null;
      for (const cand of nodesToDraw) {
        if (cand.lvl === sd.child_level && cand.fim_id === sd.child_id && cand.time_ms <= node.time_ms) {
          if (!childPos || cand.time_ms > childPos.time_ms) childPos = cand;
        }
      }
      if (!childPos) continue;
      ctx.strokeStyle = activeNode
        ? (sd.fired ? (partial ? 'rgba(251,191,36,0.75)' : 'rgba(74,222,128,0.7)') : 'rgba(248,113,113,0.35)')
        : 'rgba(51,65,85,0.28)';
      ctx.lineWidth = activeNode && sd.fired ? 1.7 : 1;
      if (!sd.fired) ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(childPos.x, childPos.y - 9);
      ctx.lineTo(node.x, node.y + 9);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Рисуем узлы
  for (const n of nodesToDraw) {
    const fired = n.time_ms <= playheadMs;
    const fresh = fired && (playheadMs - n.time_ms <= 96);
    const isSel = selectedTarget && selectedTarget.level === n.lvl && selectedTarget.fim_id === n.fim_id;
    if (n.lvl === 0) {
      const r = fresh ? 10 : 8;
      ctx.fillStyle = fired ? (fresh ? '#0284c7' : '#0c4a6e') : '#0f172a';
      ctx.strokeStyle = fired ? '#38bdf8' : '#334155';
      ctx.lineWidth = fresh ? 2 : 1;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = fired ? '#f8fafc' : '#64748b';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(n.text === ' ' ? '␠' : n.text, n.x, n.y + 3.5);
      ctx.textAlign = 'left';
    } else {
      const label = n.text.length > 14 ? n.text.slice(0, 13) + '…' : n.text;
      const bw = Math.max(54, Math.min(130, label.length * 6.8 + 18));
      const bh = 21;
      const partial = n.confidence < 0.99;
      ctx.fillStyle = isSel ? '#0c2749' : (fresh ? (partial ? '#78350f' : '#064e3b') : '#0b1220');
      ctx.strokeStyle = isSel ? '#38bdf8' : (fired ? (partial ? '#fbbf24' : '#4ade80') : '#334155');
      ctx.lineWidth = isSel || fresh ? 2.2 : 1.1;
      ctx.beginPath();
      ctx.roundRect(n.x - bw / 2, n.y - bh / 2, bw, bh, 5);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = fired ? '#f8fafc' : '#64748b';
      ctx.font = '10.5px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(label, n.x, n.y + 3.5);
      ctx.textAlign = 'left';
      graphHitBoxes.push({x1: n.x - bw/2, y1: n.y - bh/2, x2: n.x + bw/2, y2: n.y + bh/2, lvl: n.lvl, fim_id: n.fim_id, time_ms: n.time_ms});
    }
  }

  // Правая панель: целевой LIF ансамбль ФИМ (Б1, Б2, Б3 -> Г) + Инстинкт 252 мс
  const px = W - ensembleW - 6, py = 8, pw = ensembleW, ph = H - 16;
  ctx.fillStyle = '#090f1d';
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.roundRect(px, py, pw, ph, 8);
  ctx.fill(); ctx.stroke();

  let instinctPulse = false;
  for (const tp of currentSim.instinct_pulses_ms) {
    if (Math.abs(playheadMs - tp) <= 45 && playheadMs >= tp - 10) instinctPulse = true;
  }

  ctx.fillStyle = instinctPulse ? '#450a0a' : '#111827';
  ctx.strokeStyle = instinctPulse ? '#f87171' : '#7f1d1d';
  ctx.lineWidth = instinctPulse ? 2.2 : 1.1;
  ctx.beginPath();
  ctx.roundRect(px + 10, py + 10, pw - 20, 26, 5);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = instinctPulse ? '#fca5a5' : '#cbd5e1';
  ctx.font = 'bold 11px monospace';
  ctx.fillText(instinctPulse ? '⚡ ИНСТИНКТ 252 мс: ИМПУЛЬС' : 'Осциллятор инстинкта 252 мс', px + 18, py + 27);

  const tgt = currentSim.target_fim;
  if (tgt) {
    ctx.fillStyle = '#7dd3fc';
    ctx.font = 'bold 11.5px monospace';
    ctx.fillText(`Ансамбль L${tgt.level}#${tgt.fim_id} «${tgt.text.slice(0, 16)}»`, px + 10, py + 54);

    const tr = currentSim.voltage_traces;
    let stepIdx = Math.min(tr.t_grid.length - 1, Math.floor(playheadMs / currentSim.dt_ms));
    if (stepIdx < 0) stepIdx = 0;

    const slotsMap = {};
    for (const s of tgt.sources) slotsMap[s.slot] = s;
    const bVals = [tr.v_b1[stepIdx] || 0, tr.v_b2[stepIdx] || 0, tr.v_b3[stepIdx] || 0];
    const headVal = tr.v_head[stepIdx] || 0;
    const headFired = tr.spike_times_head.some(t => t <= playheadMs);
    const gx = px + pw - 52, gy = py + 150;

    for (let s = 0; s < 3; s++) {
      const bx = px + 36, by = py + 88 + s * 64;
      const src = slotsMap[s];
      const active = bVals[s] > 0.15;

      ctx.strokeStyle = src ? (active ? '#38bdf8' : '#334155') : '#111827';
      ctx.lineWidth = active ? 2.4 : 1.1;
      ctx.beginPath();
      ctx.moveTo(bx + 17, by);
      ctx.lineTo(gx - 24, gy);
      ctx.stroke();

      ctx.fillStyle = src ? (active ? '#0c4a6e' : '#0f172a') : '#060911';
      ctx.strokeStyle = src ? (active ? '#38bdf8' : '#475569') : '#1e293b';
      ctx.beginPath();
      ctx.arc(bx, by, 17, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = src ? '#f8fafc' : '#475569';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`Б${s+1}`, bx, by + 4);
      ctx.textAlign = 'left';

      if (src) {
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '10.5px monospace';
        ctx.fillText(`${src.text.slice(0, 12)} (${bVals[s].toFixed(2)})`, bx + 22, by - 3);
      }
    }

    ctx.fillStyle = headFired ? '#064e3b' : (headVal >= 1.5 ? '#451a03' : '#0f172a');
    ctx.strokeStyle = headFired ? '#4ade80' : '#fbbf24';
    ctx.lineWidth = headFired ? 2.8 : 1.8;
    ctx.beginPath();
    ctx.arc(gx, gy, 24, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Г', gx, gy + 4);
    ctx.font = '10.5px monospace';
    ctx.fillStyle = headFired ? '#4ade80' : '#fbbf24';
    ctx.fillText(`V=${headVal.toFixed(2)}`, gx, gy + 38);
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Vth=2.0', gx, gy + 51);
    ctx.textAlign = 'left';
  }
}

document.getElementById('networkCanvas').addEventListener('click', function(e) {
  const rect = this.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  for (let i = graphHitBoxes.length - 1; i >= 0; i--) {
    const b = graphHitBoxes[i];
    if (mx >= b.x1 && mx <= b.x2 && my >= b.y1 && my <= b.y2) {
      selectSpikeRow(b.lvl, b.fim_id, b.time_ms);
      break;
    }
  }
});

function drawRasterPlot() {
  const canvas = document.getElementById('rasterCanvas');
  const {ctx, W, H} = setupHiDpiCanvas(canvas);
  ctx.clearRect(0, 0, W, H);
  if (!currentSim) return;

  const padL = 78, padR = 14, padT = 12, padB = 24;
  const pw = W - padL - padR, ph = H - padT - padB;
  const dur = Math.max(1, currentSim.duration_ms);
  const rows = 11;
  const rowH = ph / rows;
  const xOf = t => padL + (t / dur) * pw;

  for (const tp of currentSim.instinct_pulses_ms) {
    const x = xOf(tp);
    ctx.strokeStyle = 'rgba(248,113,113,0.22)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, H - padB);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  const labels = ['Инстинкт', 'Ур.0 (41)', 'Ур.1', 'Ур.2', 'Ур.3', 'Ур.4', 'Ур.5', 'Ур.6', 'Ур.7', 'Ур.8', 'Ур.9'];
  ctx.font = '10px monospace';
  for (let r = 0; r < rows; r++) {
    const yMid = padT + (rows - 1 - r + 0.5) * rowH;
    ctx.strokeStyle = '#0f172a';
    ctx.beginPath();
    ctx.moveTo(padL, yMid + rowH * 0.48);
    ctx.lineTo(W - padR, yMid + rowH * 0.48);
    ctx.stroke();
    ctx.fillStyle = r === 0 ? '#f87171' : '#94a3b8';
    ctx.fillText(labels[r], 6, yMid + 3);
  }

  for (const tp of currentSim.instinct_pulses_ms) {
    if (tp > playheadMs) continue;
    const x = xOf(tp), yMid = padT + (rows - 1 + 0.5) * rowH;
    ctx.strokeStyle = '#f87171';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(x, yMid - rowH * 0.36);
    ctx.lineTo(x, yMid + rowH * 0.36);
    ctx.stroke();
  }

  for (const es of currentSim.element_spikes) {
    if (es.time_ms > playheadMs) continue;
    const x = xOf(es.time_ms), yMid = padT + (rows - 2 + 0.5) * rowH;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(x, yMid - rowH * 0.34);
    ctx.lineTo(x, yMid + rowH * 0.34);
    ctx.stroke();
  }

  for (const fs of currentSim.fim_spikes) {
    if (fs.time_ms > playheadMs) continue;
    const r = fs.level + 1;
    if (r >= rows) continue;
    const x = xOf(fs.time_ms), yMid = padT + (rows - 1 - r + 0.5) * rowH;
    const partial = fs.confidence < 0.99;
    ctx.strokeStyle = partial ? '#fbbf24' : '#4ade80';
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(x, yMid - rowH * 0.38);
    ctx.lineTo(x, yMid + rowH * 0.38);
    ctx.stroke();
    if (fs.level >= 2) {
      ctx.fillStyle = partial ? '#fde68a' : '#bbf7d0';
      ctx.font = '9.5px monospace';
      ctx.fillText(fs.text.slice(0, 12), Math.min(W - 82, x + 3), yMid + 3);
    }
  }

  const px = xOf(playheadMs);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(px, padT);
  ctx.lineTo(px, H - padB);
  ctx.stroke();

  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px monospace';
  ctx.fillText('0 мс', padL, H - 6);
  ctx.fillText(`${dur} мс`, W - 56, H - 6);
}

function drawOscilloscope() {
  const canvas = document.getElementById('oscCanvas');
  const {ctx, W, H} = setupHiDpiCanvas(canvas);
  ctx.clearRect(0, 0, W, H);
  if (!currentSim || !currentSim.voltage_traces) return;

  const tr = currentSim.voltage_traces;
  const padL = 42, padR = 14, padT = 14, padB = 24;
  const pw = W - padL - padR, ph = H - padT - padB;
  const dur = Math.max(1, currentSim.duration_ms);
  const vMax = 3.5;

  const xOf = t => padL + (t / dur) * pw;
  const yOf = v => padT + ph - (Math.min(vMax, Math.max(0, v)) / vMax) * ph;

  ctx.font = '10px monospace';
  for (const val of [0, 1.0, 2.0, 3.0]) {
    const y = yOf(val);
    ctx.strokeStyle = val === 2.0 ? '#ef4444' : '#111827';
    ctx.lineWidth = val === 2.0 ? 1.4 : 1;
    if (val === 2.0) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = val === 2.0 ? '#f87171' : '#64748b';
    ctx.fillText(val.toFixed(1), 8, y + 3);
  }

  for (const tp of currentSim.instinct_pulses_ms) {
    const x = xOf(tp);
    ctx.strokeStyle = 'rgba(248,113,113,0.18)';
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, H - padB);
    ctx.stroke();
  }

  const maxStep = Math.min(tr.t_grid.length - 1, Math.floor(playheadMs / currentSim.dt_ms));
  const series = [
    {arr: tr.v_b1, color: '#60a5fa', width: 1.6},
    {arr: tr.v_b2, color: '#c084fc', width: 1.6},
    {arr: tr.v_b3, color: '#34d399', width: 1.6},
    {arr: tr.v_head, color: '#fbbf24', width: 2.3},
  ];
  for (const s of series) {
    if (maxStep < 1) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width;
    ctx.beginPath();
    for (let i = 0; i <= maxStep; i++) {
      const x = xOf(tr.t_grid[i]), y = yOf(s.arr[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const px = xOf(playheadMs);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(px, padT);
  ctx.lineTo(px, H - padB);
  ctx.stroke();

  const b1 = (tr.v_b1[maxStep] || 0).toFixed(2);
  const b2 = (tr.v_b2[maxStep] || 0).toFixed(2);
  const b3 = (tr.v_b3[maxStep] || 0).toFixed(2);
  const vh = (tr.v_head[maxStep] || 0).toFixed(2);
  document.getElementById('oscVoltagesReadout').textContent =
    `V_Б1=${b1}  V_Б2=${b2}  V_Б3=${b3}  |  V_Г=${vh} / ${tr.v_threshold.toFixed(2)}`;

  ctx.fillStyle = '#94a3b8';
  ctx.fillText('0 мс', padL, H - 6);
  ctx.fillText(`${dur} мс`, W - 56, H - 6);
}

async function sendText(learn) {
  const t0 = performance.now();
  const text = document.getElementById('inputText').value;
  const res = await fetch('/api/check', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, learn})
  });
  const data = await res.json();
  const dt = (performance.now() - t0).toFixed(1);
  document.getElementById('latencyBadge').textContent = `Обработка: ${dt} мс · Вход: «${data.clean_text || '—'}»`;

  const ansMain = document.getElementById('topAnswerMain');
  const ansMeta = document.getElementById('topAnswerMeta');
  if (!data.top_level) {
    ansMain.className = 'readout-main muted';
    ansMain.textContent = 'На входе пока нет узнанных ФИМов (можно дообучить).';
    ansMeta.textContent = '';
  } else {
    ansMain.className = 'readout-main';
    ansMain.textContent = `«${data.top_response_text}»`;
    ansMeta.innerHTML = `Уровень ${data.top_level} · ` + data.top_fims.map(f => {
      const confBadge = (f.confidence && f.confidence < 0.99) ? ` [2-из-3: ${f.confidence}]` : '';
      return `<span class="slot-chip hit" onclick="inspectFim(${data.top_level}, ${f.fim_id})">L${data.top_level}#${f.fim_id} «${esc(f.text)}»${confBadge}</span>`;
    }).join(' ');
  }

  const recMain = document.getElementById('recallMain');
  const recMeta = document.getElementById('recallMeta');
  if (data.recalled_level > data.top_level && data.recalled_text) {
    recMain.className = 'readout-main';
    recMain.style.color = '#86efac';
    recMain.textContent = `«${data.recalled_text}»`;
    recMeta.innerHTML = `Максимальный связанный узел: <span class="slot-chip hit" onclick="inspectFim(${data.recalled_level}, ${data.recalled_fim_id})">L${data.recalled_level}#${data.recalled_fim_id}</span>`;
  } else {
    recMain.className = 'readout-main muted';
    recMain.style.color = '';
    recMain.textContent = 'Совпадает с верхним узнанным уровнем';
    recMeta.textContent = '';
  }

  if (data.recalled_explanation) renderExplanation(data.recalled_explanation);
  else if (data.top_explanation) renderExplanation(data.top_explanation);

  renderSummary(data.summary);
  if (data.simulation) {
    loadSimulation(data.simulation, true);
  }
}

window.addEventListener('resize', () => { if (currentSim) drawNengoAll(); });
inspectLevel(3);
setExample('мама мыла раму');
</script>
</body>
</html>
"""


class FIMHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            self._send_json(NET.summary())
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
            exp = NET.explain_fim(lvl, fid, max_depth=5)
            decoded = NET.decode_fim(lvl, fid)
            exp["simulation"] = SIM.run(decoded, target_level=lvl, target_fim_id=fid, learn=False)
            self._send_json(exp)
            return

        raw = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        if self.path == "/api/check":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            req = json.loads(body or "{}")
            result = analyze_input(req.get("text", ""), learn=bool(req.get("learn", False)))
            self._send_json(result)
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
