"""
Веб-стенд 9-уровневой ФИМ-модели со схематической диаграммой 9 уровней и 41 элемента,
перетаскиванием и подписью количества синапсов (строго по ТЗ без выдуманных параметров).

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
    - узнавание на верхнем уровне и всплытие по исходящим синапсам;
    - полную трассировку спайков (3-из-3, 2-из-2 и 2-из-3) с кол-вом синапсов;
    - совпадения подстрок в памяти;
    - временну́ю симуляцию (такт 84 мс, цикл инстинкта 252 мс, растр L0..L9).
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
    """Возвращает элементы L0 (при level=0) или консолидированные ФИМы уровня 1..9."""
    if level == 0:
        cfg = NET.get_config()
        elems = cfg["l0_elements"]
        q = query.strip().lower()
        if q:
            elems = [
                e for e in elems
                if q in e["char"].lower() or q == str(e["node_id"]) or q == f"#{e['node_id']}"
            ]
        return {
            "level": 0,
            "total_consolidated": cfg["l0_count"],
            "matched_count": len(elems),
            "max_fims": cfg["l0_count"],
            "fims": [
                {
                    "fim_id": e["node_id"],
                    "level": 0,
                    "text": e["char"],
                    "recognitions": 0,
                    "synapse_slots": [],
                    "synapse_counts": [e["b1"], e["b2"], e["b3"]],
                    "outgoing_synapses_per_fim": e["quota"],
                    "actual_out_synapses": e["out_synapses"],
                    "window_ms": NET.tick_ms,
                }
                for e in elems[:limit]
            ],
        }

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
            "outgoing_synapses_per_fim": (
                f.outgoing_synapses_per_fim
                if f.outgoing_synapses_per_fim is not None
                else pool.synapses_per_fim_out
            ),
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
  <title>Схематическая диаграмма 9 уровней ФИМ и настройка синапсов (ТЗ)</title>
  <style>
    :root {
      --desk: #e5e9f0;
      --panel: #f6f8fb;
      --white: #ffffff;
      --bar: #262c34;
      --border: #b3becd;
      --grid: #dde3ec;
      --text: #161a20;
      --muted: #525d6d;
      --blue: #1a5bb8;
      --green: #167948;
      --amber: #b45309;
      --red: #b91c1c;
      --mono: "DejaVu Sans Mono", "Consolas", monospace;
      --sans: "Segoe UI", Tahoma, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      background: var(--desk);
      color: var(--text);
      font-family: var(--sans);
      font-size: 12px;
      line-height: 1.35;
    }

    .topbar {
      background: var(--bar);
      color: #f8fafc;
      padding: 7px 14px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 2px solid #181c22;
      position: sticky;
      top: 0;
      z-index: 60;
    }
    .tb-title {
      font-weight: 700;
      font-size: 13.5px;
      letter-spacing: 0.01em;
    }
    .tb-rules {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-family: var(--mono);
      font-size: 11px;
    }
    .tb-chip {
      background: #1b2027;
      border: 1px solid #444e5e;
      padding: 3px 8px;
      color: #dbe3ee;
    }
    .tb-chip b { color: #7cc4ff; }

    button {
      font-family: var(--sans);
      font-size: 11.5px;
      font-weight: 600;
      padding: 4px 10px;
      border: 1px solid #8c99ab;
      background: #eceff4;
      color: #161a20;
      border-radius: 2px;
      cursor: pointer;
    }
    button:hover { background: #dce3ed; }
    button.primary { background: var(--blue); border-color: #12438a; color: #fff; }
    button.primary:hover { background: #154d9e; }
    button.good { background: var(--green); border-color: #0e5934; color: #fff; }
    button.danger { background: #c53030; border-color: #932020; color: #fff; }
    button.tab-btn.active { background: var(--blue); border-color: #12438a; color: #fff; }

    input[type="text"], input[type="number"], select, textarea {
      font-family: var(--mono);
      font-size: 11.5px;
      background: #ffffff;
      color: #111827;
      border: 1px solid #96a2b3;
      border-radius: 2px;
      padding: 3px 6px;
    }
    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--blue);
    }

    .container {
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-width: 1900px;
      margin: 0 auto;
    }
    .win {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 2px;
    }
    .win-head {
      background: #e1e6ef;
      border-bottom: 1px solid var(--border);
      padding: 6px 10px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
    }
    .win-title {
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: #1b2028;
    }
    .win-body { padding: 8px; }

    /* 9-Level Interactive Schematic Container */
    .diagram-scroll {
      overflow-x: auto;
      background: #ffffff;
      border: 1px solid var(--border);
      position: relative;
      user-select: none;
    }
    .diagram-stage {
      position: relative;
      width: 2440px;
      height: 410px;
      background-image:
        linear-gradient(to right, #f2f4f8 1px, transparent 1px),
        linear-gradient(to bottom, #f2f4f8 1px, transparent 1px);
      background-size: 20px 20px;
    }
    .diagram-overlay-input {
      position: absolute;
      background: #fffdf5;
      border: 1px solid #1a5bb8;
      padding: 2px 4px;
      font-family: var(--mono);
      font-size: 10.5px;
      text-align: center;
      z-index: 10;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .diagram-overlay-input input {
      width: 58px;
      text-align: center;
      padding: 1px 3px;
      font-weight: 700;
      color: #1a5bb8;
      border: 1px solid #9bb7e0;
    }
    .diagram-drag-hint {
      font-family: var(--mono);
      font-size: 11px;
      color: #1a5bb8;
      background: #edf4ff;
      border: 1px solid #9bb7e0;
      padding: 3px 8px;
    }

    /* Two-column Middle Section */
    .mid-grid {
      display: grid;
      grid-template-columns: 540px minmax(480px, 1fr);
      gap: 8px;
      align-items: start;
    }
    @media (max-width: 1150px) {
      .mid-grid { grid-template-columns: 1fr; }
    }

    /* Bottom Two-Column Section */
    .bot-grid {
      display: grid;
      grid-template-columns: 1.35fr 0.95fr;
      gap: 8px;
      align-items: start;
    }
    @media (max-width: 1150px) {
      .bot-grid { grid-template-columns: 1fr; }
    }

    canvas.plot {
      display: block;
      width: 100%;
      background: #ffffff;
      border: 1px solid var(--border);
    }
    .param-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      background: #ffffff;
      border: 1px solid var(--border);
      padding: 6px 8px;
      margin-bottom: 8px;
    }
    .param-item {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 11.5px;
    }
    .param-item input[type="number"] { width: 72px; text-align: right; }

    table {
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      font-size: 11.5px;
    }
    th, td {
      border: 1px solid var(--grid);
      padding: 4px 6px;
      text-align: left;
      vertical-align: middle;
    }
    th {
      background: #ebeff5;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    tr.clickable:hover { background: #edf4ff; cursor: pointer; }
    tr.selected-row { background: #dbeafe !important; }
    .scroll-box {
      max-height: 360px;
      overflow-y: auto;
      border: 1px solid var(--border);
      background: #ffffff;
    }

    .tag {
      display: inline-block;
      font-family: var(--mono);
      font-size: 10.5px;
      padding: 1px 5px;
      border: 1px solid #b3becd;
      background: #f1f4f9;
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
      padding: 4px 7px;
      margin-top: 6px;
      border: 1px solid #5ca97a;
      background: #edf7f0;
      color: #14532d;
      display: none;
    }
    .status-msg.show { display: block; }
    .status-msg.err { border-color: #d96b6b; background: #fdf2f2; color: #7f1d1d; }

    .l0-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
    }
    .l0-cell {
      border: 1px solid var(--border);
      background: #ffffff;
      padding: 3px 6px;
      cursor: pointer;
      font-family: var(--mono);
      font-size: 11px;
      min-width: 78px;
    }
    .l0-cell:hover { border-color: var(--blue); background: #edf4ff; }
    .l0-cell.active { border-color: var(--blue); background: #dbeafe; font-weight: 700; }

    .readout-pair {
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
    .tree-node {
      margin-left: 12px;
      padding-left: 8px;
      border-left: 1px solid #96a2b3;
      margin-top: 3px;
    }
    .tree-root { margin-left: 0; border-left: none; padding-left: 0; }
    .tree-card {
      background: #ffffff;
      border: 1px solid var(--grid);
      padding: 3px 6px;
      cursor: pointer;
      font-size: 11px;
    }
    .tree-card:hover { border-color: var(--blue); background: #f3f7fd; }
  </style>
</head>
<body>

<div class="topbar">
  <div>
    <div class="tb-title">ФИМ-Сеть (10 уровней: L0 + L1–L9) — Визуальная схема и конструктор синапсов</div>
  </div>
  <div class="tb-rules">
    <div class="tb-chip">Элементы L0: <b id="topL0Count">41</b></div>
    <div class="tb-chip">Такт L0: <b id="topTickMs">84 мс</b></div>
    <div class="tb-chip">Цикл инстинкта: <b id="topInstMs">252 мс</b></div>
    <div class="tb-chip">Правило ФИМ: <b>мин 2 / макс 3 синапса (Б1,Б2,Б3 &rarr; Г)</b></div>
    <div class="tb-chip">Лимит L2: <b id="topL2Limit">&le; 6 симв.</b></div>
    <div class="tb-chip">Всего ФИМ в памяти: <b id="topTotalFims">—</b> / <b>9 000 000</b></div>
  </div>
</div>

<div class="container">

  <!-- 1. SCHEMATIC DIAGRAM OF ALL 9 LEVELS + L0 + INSTINCT NEURON -->
  <div class="win">
    <div class="win-head">
      <span class="win-title">1. Схематическая диаграмма 9 уровней, Уровня 0 (41 элемент) и Нейрона инстинкта (252 мс)</span>
      <span class="diagram-drag-hint">Потяните мышкой от (Г) / L0 к Б1-Б3 следующего уровня или впишите число синапсов прямо на стрелке</span>
    </div>
    <div class="win-body">
      <div class="param-bar">
        <div class="param-item">
          <label>Такт подачи L0 (мс):</label>
          <input type="number" id="cfgTickMs" min="12" max="1000" step="12" value="84" />
        </div>
        <div class="param-item">
          <label>Цикл нейрона инстинкта (мс):</label>
          <input type="number" id="cfgInstMs" min="24" max="5000" step="12" value="252" />
        </div>
        <div class="param-item">
          <label>Мин. сигналов:</label>
          <select id="cfgMinSyn"><option value="2">2 (Б1, Б3)</option><option value="3">3 (Б1, Б2, Б3)</option></select>
        </div>
        <div class="param-item">
          <label>Макс. сигналов:</label>
          <select id="cfgMaxSyn"><option value="3">3 (Б1, Б2, Б3)</option><option value="2">2 (Б1, Б3)</option></select>
        </div>
        <div class="param-item">
          <label>Макс. символов на L2:</label>
          <input type="number" id="cfgL2Chars" min="2" max="18" step="1" value="6" />
        </div>
        <div class="param-item">
          <input type="checkbox" id="cfgAssoc23" checked />
          <label for="cfgAssoc23">Узнавание 2-из-3 на L2–L9</label>
        </div>
        <button class="primary" onclick="saveGlobalParams()">Сохранить параметры ТЗ</button>
        <span id="dragLiveStatus" class="mono" style="color:var(--blue); font-weight:700;"></span>
      </div>

      <div class="diagram-scroll" id="diagramScroll">
        <div class="diagram-stage" id="diagramStage">
          <svg id="diagramSvg" width="2440" height="410" style="display:block;"></svg>
          <div id="diagramInputsLayer"></div>
        </div>
      </div>
      <div id="globalCfgMsg" class="status-msg"></div>
    </div>
  </div>

  <!-- 2. MIDDLE SECTION: SINGLE-FIM / L0 SYNAPSE CONSTRUCTOR + INPUT STREAM & SPIKE RASTER -->
  <div class="mid-grid">
    <!-- Left: Interactive Single-FIM Synapse Schematic -->
    <div class="win">
      <div class="win-head">
        <span class="win-title" id="fimBoxTitle">2. Схема синапсов выбранного ФИМ / элемента L0 (перетягивайте синапсы)</span>
        <button onclick="openCreateFim()">+ Новый ФИМ</button>
      </div>
      <div class="win-body">
        <div style="position:relative; border:1px solid var(--border); background:#fff; margin-bottom:8px;">
          <svg id="singleFimSvg" viewBox="0 0 520 215" style="width:100%; height:215px; display:block; user-select:none;"></svg>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 120px; gap:5px 8px; align-items:center; margin-bottom:6px;">
          <label>Уровень / ID модуля:</label>
          <div class="btn-row">
            <select id="edLevel" style="width:54px;"></select>
            <input type="text" id="edFimId" readonly value="—" style="width:60px; background:#edf1f7;" />
          </div>
          <label>Число синапсов в этом ФИМ:</label>
          <select id="edSynMode" onchange="onSynModeChange()">
            <option value="3">3 синапса (Б1, Б2, Б3)</option>
            <option value="2">2 синапса (Б1, Б3 — без Б2)</option>
          </select>
        </div>

        <table id="edSlotsTable">
          <thead>
            <tr>
              <th>Вход БН</th>
              <th>От какого элемента/ФИМ (L-1)</th>
              <th style="width:95px;">Синапсов (шт.)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="mono"><b>Б1</b> (слот 0)</td>
              <td><input type="text" id="edSrc1" placeholder="символ или #ID/текст" /></td>
              <td><input type="number" id="edCnt1" min="1" max="10000" value="1" oninput="redrawSingleFimSvg()" /></td>
            </tr>
            <tr id="edRowB2">
              <td class="mono"><b>Б2</b> (слот 1)</td>
              <td><input type="text" id="edSrc2" placeholder="символ или #ID/текст" /></td>
              <td><input type="number" id="edCnt2" min="1" max="10000" value="1" oninput="redrawSingleFimSvg()" /></td>
            </tr>
            <tr>
              <td class="mono"><b>Б3</b> (слот 2)</td>
              <td><input type="text" id="edSrc3" placeholder="символ или #ID/текст" /></td>
              <td><input type="number" id="edCnt3" min="1" max="10000" value="1" oninput="redrawSingleFimSvg()" /></td>
            </tr>
          </tbody>
        </table>

        <div style="display:grid; grid-template-columns: 1fr 110px; gap:5px 8px; align-items:center; margin-top:6px;">
          <label for="edOutQuota">Исходящих синапсов от этого (Г) вверх (на разные БН L+1):</label>
          <input type="number" id="edOutQuota" min="1" max="1000000" value="500" oninput="redrawSingleFimSvg()" />

          <label for="edWin">Окно прослушивания этого ФИМ (мс, пусто = окно уровня):</label>
          <input type="number" id="edWin" min="24" step="12" placeholder="по уровню" />
        </div>

        <div class="btn-row" style="margin-top:8px;">
          <button class="primary" id="btnSaveFim" onclick="saveFimChanges()">Сохранить синапсы ФИМ</button>
          <button class="good" onclick="createNewFim()">Создать новый ФИМ</button>
          <button class="danger" id="btnDisableFim" onclick="disableCurrentFim()">Отключить</button>
        </div>
        <div id="fimEditMsg" class="status-msg"></div>

        <div id="fimRelationsBox" style="margin-top:8px; border-top:1px solid var(--border); padding-top:6px;">
          <div class="muted" id="edMetaText">Нажмите на любой ФИМ или элемент L0 чтобы увидеть его синапсы и цепочку.</div>
          <div id="edParentsList" style="margin-top:4px;"></div>
          <div id="edTreeList" style="margin-top:6px; max-height:190px; overflow-y:auto;"></div>
        </div>
      </div>
    </div>

    <!-- Right: Input Flow & Time Spike Raster -->
    <div class="win">
      <div class="win-head">
        <span class="win-title">3. Подача входного потока (84 мс / символ) и временна́я диаграмма спайков</span>
        <span class="mono muted" id="latencyBadge">Готов</span>
      </div>
      <div class="win-body">
        <textarea id="inputText" rows="2" style="width:100%; resize:vertical;" placeholder="Введите текст для прогона по 9 уровням...">мама раму</textarea>
        <div class="btn-row" style="margin-top:6px;">
          <button class="primary" onclick="sendText(false)">Проверить (без записи)</button>
          <button class="good" onclick="sendText(true)">Обучить (+ Записать новые ФИМ)</button>
          <button onclick="setExample('мама мыла раму')">мама мыла раму</button>
          <button onclick="setExample('мама раму')">мама раму (2 из 3)</button>
          <button onclick="setExample('дом у берега')">дом у берега</button>
          <button onclick="setExample('Только Иван выразил готовность помочь.')">Корпус RuCoLA</button>
        </div>

        <div class="readout-pair">
          <div class="readout-box">
            <div class="muted" style="font-weight:700;">Узнавание на верхнем сработавшем уровне:</div>
            <div class="mono" id="topAnswerMain" style="font-size:13px; font-weight:700; margin-top:2px;">—</div>
            <div id="topAnswerMeta" style="margin-top:3px;"></div>
          </div>
          <div class="readout-box">
            <div class="muted" style="font-weight:700;">Выход по исходящим синапсам Г вверх:</div>
            <div class="mono" id="recallMain" style="font-size:13px; font-weight:700; margin-top:2px;">—</div>
            <div id="recallMeta" style="margin-top:3px;"></div>
          </div>
        </div>

        <div class="param-bar" style="margin-top:8px; margin-bottom:6px;">
          <button onclick="resetPlayhead()">|&lt;&lt; 0 мс</button>
          <button onclick="stepPlayhead(-1)">-84 мс</button>
          <button id="playBtn" onclick="togglePlay()">&#9654; Пуск</button>
          <button onclick="stepPlayhead(1)">+84 мс</button>
          <input type="range" id="timeSlider" min="0" max="252" step="84" value="252" oninput="scrubTime(this.value)" style="flex:1; min-width:120px;" />
          <span class="mono" id="clockLabel" style="font-weight:700; color:var(--blue);">t = 0 мс</span>
        </div>

        <canvas id="rasterCanvas" class="plot" height="290"></canvas>
        <div class="muted" style="margin-top:4px;">
          Серые штрихи (L0) — элементы каждые <span id="lblTick1">84</span> мс ·
          Оранжевый пунктир — такты нейрона инстинкта каждые <span id="lblInst1">252</span> мс ·
          Синие импульсы (L1–L9) — срабатывание Г (3/3 и 2/2) · Жёлтые — 2-из-3.
        </div>
      </div>
    </div>
  </div>

  <!-- 3. BOTTOM SECTION: SEQUENCE TRACE TABLE + L0 41 ELEMENTS & L1..L9 MEMORY -->
  <div class="bot-grid">
    <div class="win">
      <div class="win-head">
        <span class="win-title">4. Пошаговая трассировка последовательностей (Б1 + Б2 + Б3 &rarr; Г)</span>
        <div class="btn-row">
          <div class="btn-row" id="traceLevelFilters"></div>
          <input type="text" id="traceSearch" placeholder="Фильтр..." oninput="renderTraceTable()" style="width:120px;" />
        </div>
      </div>
      <div class="win-body" style="padding:4px;">
        <div class="scroll-box">
          <table>
            <thead>
              <tr>
                <th>Время / Такт / Инстинкт</th>
                <th>ФИМ</th>
                <th>Текст</th>
                <th>Входы Б1, Б2, Б3 (источник и число синапсов)</th>
                <th>Окно W_L</th>
                <th>Выход Г на L+1</th>
                <th>Правило</th>
              </tr>
            </thead>
            <tbody id="traceBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="win">
      <div class="win-head">
        <span class="win-title">5. Элементы Уровня 0 (41 шт.) и память уровней L1–L9</span>
        <span class="mono muted" id="memCountInfo">—</span>
      </div>
      <div class="win-body">
        <div class="btn-row" id="memTabs" style="margin-bottom:6px;"></div>
        <div class="btn-row" style="margin-bottom:6px;">
          <input type="text" id="memSearch" placeholder="Поиск элемента, текста или #ID..." oninput="searchMem()" style="flex:1;" />
          <button onclick="searchMem()">Найти</button>
        </div>
        <div id="l0MatrixWrap" style="display:none; max-height:310px; overflow-y:auto;"></div>
        <div class="scroll-box" id="memTableWrap" style="max-height:310px;">
          <table>
            <thead>
              <tr>
                <th>#ID</th>
                <th>Текст ФИМ</th>
                <th>Синапсы (Б1/Б2/Б3)</th>
                <th>Окно</th>
                <th>Узнав.</th>
              </tr>
            </thead>
            <tbody id="memTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
let netConfig = null;
let currentSim = null;
let currentTraceFilterLvl = 0;
let currentMemLvl = 0;
let playheadMs = 0;
let isPlaying = false;
let playTimer = null;
let currentEdited = { level: 3, fim_id: 1, mode: 'edit', explain: null };

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showMsg(elId, msg, isErr=false) {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.className = 'status-msg show' + (isErr ? ' err' : '');
  setTimeout(() => { el.className = 'status-msg'; }, 4200);
}

// --- Init Static Selectors ---
(function initUI() {
  const edLvl = document.getElementById('edLevel');
  for (let l = 1; l <= 9; l++) {
    const o = document.createElement('option');
    o.value = l; o.textContent = 'L' + l;
    edLvl.appendChild(o);
  }
  const mt = document.getElementById('memTabs');
  for (let l = 0; l <= 9; l++) {
    const b = document.createElement('button');
    b.className = 'tab-btn' + (l === 0 ? ' active' : '');
    b.textContent = l === 0 ? 'L0 (41)' : 'L' + l;
    b.onclick = () => switchMemLevel(l);
    mt.appendChild(b);
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

function setExample(txt) {
  document.getElementById('inputText').value = txt;
  sendText(false);
}

// --- Config & Global 9-Level Schematic ---
async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();
  applyConfig(cfg);
}

function applyConfig(cfg) {
  if (!cfg) return;
  netConfig = cfg;
  document.getElementById('cfgTickMs').value = cfg.tick_ms;
  document.getElementById('cfgInstMs').value = cfg.instinct_cycle_ms;
  document.getElementById('cfgL2Chars').value = cfg.level2_max_chars;
  document.getElementById('cfgAssoc23').checked = !!cfg.allow_associative_2of3;

  document.getElementById('topTickMs').textContent = cfg.tick_ms + ' мс';
  document.getElementById('lblTick1').textContent = cfg.tick_ms;
  document.getElementById('topInstMs').textContent = cfg.instinct_cycle_ms + ' мс';
  document.getElementById('lblInst1').textContent = cfg.instinct_cycle_ms;
  document.getElementById('topL2Limit').textContent = '≤ ' + cfg.level2_max_chars + ' симв.';

  const l1 = cfg.levels['1'] || cfg.levels[1];
  if (l1) {
    document.getElementById('cfgMinSyn').value = String(l1.min_synapses);
    document.getElementById('cfgMaxSyn').value = String(l1.max_synapses);
  }
  let tot = 0;
  for (let l = 1; l <= 9; l++) {
    const lv = cfg.levels[String(l)] || cfg.levels[l];
    if (lv) tot += lv.consolidated_fims;
  }
  document.getElementById('topTotalFims').textContent = tot.toLocaleString('ru-RU');
  draw9LevelSchematic();
  if (currentMemLvl === 0) renderL0Matrix();
}

async function saveGlobalParams(extraPatch = {}) {
  const payload = Object.assign({
    tick_ms: Number(document.getElementById('cfgTickMs').value),
    instinct_cycle_ms: Number(document.getElementById('cfgInstMs').value),
    min_synapses: Number(document.getElementById('cfgMinSyn').value),
    max_synapses: Number(document.getElementById('cfgMaxSyn').value),
    level2_max_chars: Number(document.getElementById('cfgL2Chars').value),
    allow_associative_2of3: document.getElementById('cfgAssoc23').checked,
  }, extraPatch);
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (res.ok) {
    applyConfig(data.config);
    showMsg('globalCfgMsg', 'Параметры сети и синапсов обновлены.');
    await sendText(false);
  } else {
    showMsg('globalCfgMsg', data.error || 'Ошибка', true);
  }
}

async function patchBundleSynapses(fromLvl, newOutCount) {
  const cnt = Math.max(1, Math.min(1000000, Math.round(Number(newOutCount) || 1)));
  let patch = {};
  if (fromLvl === 0) {
    patch = { synapses_per_element_l0: cnt };
  } else {
    patch = { levels: { [fromLvl]: { synapses_per_fim_out: cnt } } };
  }
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(patch)
  });
  const data = await res.json();
  if (res.ok) {
    applyConfig(data.config);
    document.getElementById('dragLiveStatus').textContent =
      fromLvl === 0
        ? `Сохранено: от 1 элемента L0 → ${cnt} синапсов на БН L1`
        : `Сохранено: от 1 ФИМ L${fromLvl} (Г) → ${cnt} синапсов на БН L${fromLvl+1}`;
  }
}

async function patchLevelWindow(lvl, newWinMs) {
  const w = Math.max(24, Math.round(Number(newWinMs) || 252));
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ levels: { [lvl]: { window_ms: w } } })
  });
  const data = await res.json();
  if (res.ok) {
    applyConfig(data.config);
    await sendText(false);
  }
}

// --- Draw Interactive 9-Level + L0 + Instinct Diagram ---
let dragBundleState = null; // { fromLvl, startX, startVal }

function draw9LevelSchematic() {
  if (!netConfig) return;
  const svg = document.getElementById('diagramSvg');
  const layer = document.getElementById('diagramInputsLayer');
  const levels = netConfig.levels;

  const boxW = 166;
  const gapW = 72; // space for synapse bundle & label between levels
  const startX = 16;
  const instY = 14;
  const cardY = 82;
  const cardH = 306;

  let svgHtml = `
    <defs>
      <marker id="arrBlue" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
        <polygon points="0 0, 7 3.5, 0 7" fill="#1a5bb8"/>
      </marker>
      <marker id="arrAmber" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
        <polygon points="0 0, 6 3, 0 6" fill="#d97706"/>
      </marker>
    </defs>
  `;
  let htmlInputs = '';

  // Top Instinct Neuron bar spanning L1..L9
  const xL1 = startX + (boxW + gapW);
  const xL9Right = startX + 9 * (boxW + gapW) + boxW;
  svgHtml += `
    <rect x="${xL1}" y="${instY}" width="${xL9Right - xL1}" height="34" rx="2" fill="#fffbeb" stroke="#d97706" stroke-width="1.5"/>
    <text x="${xL1 + 12}" y="${instY + 21}" font-family="Segoe UI, Tahoma, sans-serif" font-size="12" font-weight="700" fill="#92400e">
      НЕЙРОН ИНСТИНКТА — цикл ${netConfig.instinct_cycle_ms} мс (связан со всеми головными нейронами Г уровней L1–L9)
    </text>
  `;

  // Draw Level 0 and Levels 1..9
  for (let l = 0; l <= 9; l++) {
    const x = startX + l * (boxW + gapW);
    const isSelMem = l === currentMemLvl;

    if (l === 0) {
      svgHtml += `
        <g style="cursor:pointer;" onclick="switchMemLevel(0)">
          <rect x="${x}" y="${cardY}" width="${boxW}" height="${cardH}" rx="2"
                fill="#f1f5f9" stroke="${isSelMem ? '#1a5bb8' : '#475569'}" stroke-width="${isSelMem ? 2.2 : 1.4}"/>
          <rect x="${x}" y="${cardY}" width="${boxW}" height="28" fill="#334155"/>
          <text x="${x + 8}" y="${cardY + 18}" font-family="Tahoma, sans-serif" font-size="11.5" font-weight="700" fill="#ffffff">
            УРОВЕНЬ 0 (${netConfig.l0_count} эл.)
          </text>
          <text x="${x + 8}" y="${cardY + 46}" font-family="DejaVu Sans Mono, monospace" font-size="10.5" fill="#1e293b">
            Такт подачи: ${netConfig.tick_ms} мс
          </text>
          <text x="${x + 8}" y="${cardY + 64}" font-family="Tahoma, sans-serif" font-size="10" fill="#475569">
            33 буквы + пробел + .,!?;:-
          </text>
          <rect x="${x + 10}" y="${cardY + 78}" width="${boxW - 20}" height="122" fill="#ffffff" stroke="#94a3b8"/>
          <text x="${x + 16}" y="${cardY + 96}" font-family="DejaVu Sans Mono, monospace" font-size="10" font-weight="700" fill="#1a5bb8">
            а б в г д е ё ж з и й
          </text>
          <text x="${x + 16}" y="${cardY + 114}" font-family="DejaVu Sans Mono, monospace" font-size="10" font-weight="700" fill="#1a5bb8">
            к л м н о п р с т у ф
          </text>
          <text x="${x + 16}" y="${cardY + 132}" font-family="DejaVu Sans Mono, monospace" font-size="10" font-weight="700" fill="#1a5bb8">
            х ц ч ш щ ъ ы ь э ю я
          </text>
          <text x="${x + 16}" y="${cardY + 150}" font-family="DejaVu Sans Mono, monospace" font-size="10" fill="#475569">
            [ ] . , ! ? ; : -
          </text>
          <text x="${x + 16}" y="${cardY + 174}" font-family="Tahoma, sans-serif" font-size="10" fill="#0f5132" font-weight="700">
            Факт выходов на L1:
          </text>
          <text x="${x + 16}" y="${cardY + 190}" font-family="DejaVu Sans Mono, monospace" font-size="10.5" fill="#0f5132" font-weight="700">
            ${netConfig.l0_total_out.toLocaleString('ru-RU')} синапсов
          </text>
          <text x="${x + 8}" y="${cardY + 230}" font-family="Tahoma, sans-serif" font-size="10" fill="#475569">
            От 1 элемента L0 к БН L1:
          </text>
          <text x="${x + 8}" y="${cardY + 248}" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="700" fill="#1a5bb8">
            ${netConfig.synapses_per_element_l0.toLocaleString('ru-RU')} синапсов
          </text>
          <text x="${x + 8}" y="${cardY + 286}" font-family="Tahoma, sans-serif" font-size="10" fill="#1a5bb8">
            Клик: открыть все 41 элемент
          </text>
        </g>
      `;
    } else {
      const lv = levels[String(l)] || levels[l];
      const cxHead = x + boxW - 22;
      // Sync line from Instinct Neuron to Head (Г)
      svgHtml += `
        <line x1="${cxHead}" y1="${instY + 34}" x2="${cxHead}" y2="${cardY + 136}"
              stroke="#d97706" stroke-width="1.3" stroke-dasharray="4,3" marker-end="url(#arrAmber)"/>
      `;
      svgHtml += `
        <g style="cursor:pointer;" onclick="if(event.target.tagName!=='INPUT') switchMemLevel(${l})">
          <rect x="${x}" y="${cardY}" width="${boxW}" height="${cardH}" rx="2"
                fill="#f8fafc" stroke="${isSelMem ? '#1a5bb8' : '#475569'}" stroke-width="${isSelMem ? 2.2 : 1.3}"/>
          <rect x="${x}" y="${cardY}" width="${boxW}" height="28" fill="${l === 2 ? '#1e3a5f' : '#2d3748'}"/>
          <text x="${x + 8}" y="${cardY + 18}" font-family="Tahoma, sans-serif" font-size="11.5" font-weight="700" fill="#ffffff">
            УРОВЕНЬ ${l} ${l === 2 ? '(≤' + netConfig.level2_max_chars + ' симв.)' : ''}
          </text>

          <!-- FIM Internal Diagram: [Б1] [Б2] [Б3] -> (Г) -->
          <rect x="${x + 8}" y="${cardY + 36}" width="${boxW - 16}" height="120" fill="#ffffff" stroke="#cbd5e1"/>
          <text x="${x + 12}" y="${cardY + 50}" font-family="Tahoma, sans-serif" font-size="9.5" fill="#475569">
            Устройство ФИМ (2..3 син.):
          </text>
          <rect x="${x + 14}" y="${cardY + 58}" width="32" height="22" fill="#e0f2fe" stroke="#0284c7"/>
          <text x="${x + 21}" y="${cardY + 73}" font-family="DejaVu Sans Mono, monospace" font-size="10.5" font-weight="700" fill="#0369a1">Б1</text>

          <rect x="${x + 14}" y="${cardY + 86}" width="32" height="22" fill="#f3e8ff" stroke="#7c3aed" stroke-dasharray="3,2"/>
          <text x="${x + 21}" y="${cardY + 101}" font-family="DejaVu Sans Mono, monospace" font-size="10.5" font-weight="700" fill="#6d28d9">Б2</text>

          <rect x="${x + 14}" y="${cardY + 114}" width="32" height="22" fill="#fef3c7" stroke="#d97706"/>
          <text x="${x + 21}" y="${cardY + 129}" font-family="DejaVu Sans Mono, monospace" font-size="10.5" font-weight="700" fill="#b45309">Б3</text>

          <line x1="${x + 46}" y1="${cardY + 69}" x2="${cxHead - 16}" y2="${cardY + 95}" stroke="#475569" stroke-width="1.2"/>
          <line x1="${x + 46}" y1="${cardY + 97}" x2="${cxHead - 16}" y2="${cardY + 97}" stroke="#475569" stroke-width="1.2" stroke-dasharray="3,2"/>
          <line x1="${x + 46}" y1="${cardY + 125}" x2="${cxHead - 16}" y2="${cardY + 99}" stroke="#475569" stroke-width="1.2"/>

          <circle cx="${cxHead}" cy="${cardY + 97}" r="16" fill="#dcfce7" stroke="#15803d" stroke-width="1.6"/>
          <text x="${cxHead - 5}" y="${cardY + 101}" font-family="DejaVu Sans Mono, monospace" font-size="12" font-weight="700" fill="#14532d">Г</text>
          <text x="${x + 54}" y="${cardY + 148}" font-family="DejaVu Sans Mono, monospace" font-size="9.5" fill="#475569">
            Б1:${lv.actual_in_b1} Б2:${lv.actual_in_b2} Б3:${lv.actual_in_b3}
          </text>

          <!-- Level Metrics & Pool -->
          <text x="${x + 8}" y="${cardY + 174}" font-family="Tahoma, sans-serif" font-size="10" fill="#475569">
            Записано ФИМ (пул 1M):
          </text>
          <text x="${x + 8}" y="${cardY + 190}" font-family="DejaVu Sans Mono, monospace" font-size="12" font-weight="700" fill="#161a20">
            ${lv.consolidated_fims.toLocaleString('ru-RU')}
          </text>
          <text x="${x + 8}" y="${cardY + 207}" font-family="DejaVu Sans Mono, monospace" font-size="9.5" fill="#0f5132">
            2-син(Б1,Б3): ${lv.two_synapse_fims.toLocaleString('ru-RU')}
          </text>
          <text x="${x + 8}" y="${cardY + 221}" font-family="DejaVu Sans Mono, monospace" font-size="9.5" fill="#1a5bb8">
            3-син(Б1..Б3): ${lv.three_synapse_fims.toLocaleString('ru-RU')}
          </text>

          <text x="${x + 8}" y="${cardY + 244}" font-family="Tahoma, sans-serif" font-size="10" fill="#475569">
            Окно W_${l} (мс):
          </text>
          <text x="${x + 8}" y="${cardY + 286}" font-family="DejaVu Sans Mono, monospace" font-size="9.5" fill="#525d6d">
            Вх. синапсов: ${lv.actual_in_total.toLocaleString('ru-RU')}
          </text>
          <text x="${x + 8}" y="${cardY + 299}" font-family="DejaVu Sans Mono, monospace" font-size="9.5" fill="#1a5bb8">
            Ср. выходов от 1 ФИМ: ${lv.avg_out_per_fim}
          </text>
        </g>
      `;

      // Editable window_ms inside Level box
      htmlInputs += `
        <div class="diagram-overlay-input" style="left:${x + 8}px; top:${cardY + 248}px; width:${boxW - 16}px; text-align:left; background:#fff; border-color:#cbd5e1;">
          W_${l}: <input type="number" value="${lv.window_ms}" min="24" step="12" style="width:76px;"
            onchange="patchLevelWindow(${l}, this.value)" /> мс
        </div>
      `;
    }

    // Draw Synapse Bundle Arrows between Level l and Level l + 1
    if (l < 9) {
      const xStart = x + boxW;
      const xEnd = x + boxW + gapW;
      const quotaVal = (l === 0)
        ? netConfig.synapses_per_element_l0
        : (levels[String(l)] || levels[l]).synapses_per_fim_out;
      const actualTot = (levels[String(l + 1)] || levels[l + 1]).actual_in_total;
      const numWires = Math.min(9, Math.max(2, Math.round(Math.log10(Math.max(1, quotaVal)) * 2.5) + 2));

      // Fan of parallel synapse lines connecting L's Head / Elements to L+1's Б1, Б2, Б3
      const ysTargets = [cardY + 69, cardY + 97, cardY + 125];
      for (let w = 0; w < numWires; w++) {
        const y0 = cardY + 90 + (w - (numWires - 1) / 2) * 5;
        const y1 = ysTargets[w % 3];
        svgHtml += `
          <line x1="${xStart}" y1="${y0}" x2="${xEnd}" y2="${y1}"
                stroke="#1a5bb8" stroke-width="1.35" opacity="0.72" marker-end="url(#arrBlue)"/>
        `;
      }

      // Draggable pull handle in the gap so user can drag left/right to pull synapse count
      svgHtml += `
        <rect x="${xStart - 2}" y="${cardY + 46}" width="${gapW + 4}" height="115" fill="transparent"
              style="cursor:ew-resize;" data-drag-from="${l}"
              onmousedown="startBundleDrag(event, ${l}, ${quotaVal})"/>
        <circle cx="${xStart + 2}" cy="${cardY + 97}" r="6" fill="#1a5bb8" stroke="#ffffff" stroke-width="1.5"
                style="cursor:ew-resize;" onmousedown="startBundleDrag(event, ${l}, ${quotaVal})"/>
      `;

      // Direct signed label box on the bundle arrow
      const boxLeft = xStart - 8;
      const boxTop = cardY + 168;
      const whoLabel = l === 0 ? 'от 1 эл. L0:' : `от 1 ФИМ L${l}:`;
      htmlInputs += `
        <div class="diagram-overlay-input" style="left:${boxLeft}px; top:${boxTop}px; width:${gapW + 16}px;">
          <div style="font-size:9px; color:#334155;">${whoLabel}</div>
          <input type="number" id="bundleInp_${l}" value="${quotaVal}" min="1" max="1000000"
                 onchange="patchBundleSynapses(${l}, this.value)" />
          <div style="font-size:9px; color:#0f5132; margin-top:1px;">син. на БН</div>
          <div style="font-size:8.5px; color:#525d6d; border-top:1px solid #e2e8f0; margin-top:2px; padding-top:1px;">
            факт: ${actualTot.toLocaleString('ru-RU')}
          </div>
        </div>
      `;
    }
  }

  svg.innerHTML = svgHtml;
  layer.innerHTML = htmlInputs;
}

function startBundleDrag(ev, fromLvl, startVal) {
  ev.preventDefault();
  ev.stopPropagation();
  dragBundleState = {
    fromLvl,
    startX: ev.clientX,
    startVal: Number(startVal) || 100,
    curVal: Number(startVal) || 100,
  };
}

window.addEventListener('mousemove', (ev) => {
  if (dragBundleState) {
    const dx = ev.clientX - dragBundleState.startX;
    const step = dragBundleState.fromLvl === 0 ? 25 : 10;
    const newVal = Math.max(1, dragBundleState.startVal + Math.round(dx / 2) * step);
    dragBundleState.curVal = newVal;
    const inp = document.getElementById('bundleInp_' + dragBundleState.fromLvl);
    if (inp) inp.value = newVal;
    document.getElementById('dragLiveStatus').textContent =
      dragBundleState.fromLvl === 0
        ? `Перетягивание: 1 элемент L0 → ${newVal} синапсов на БН L1`
        : `Перетягивание: 1 ФИМ L${dragBundleState.fromLvl} → ${newVal} синапсов на БН L${dragBundleState.fromLvl + 1}`;
  }
  if (singleDragState) {
    const dx = ev.clientX - singleDragState.startX;
    const step = singleDragState.slotKey === 'out' ? 10 : 1;
    const newVal = Math.max(1, singleDragState.startVal + Math.round(dx / 6) * step);
    singleDragState.curVal = newVal;
    document.getElementById(singleDragState.inputId).value = newVal;
    redrawSingleFimSvg();
  }
});

window.addEventListener('mouseup', async () => {
  if (dragBundleState) {
    const st = dragBundleState;
    dragBundleState = null;
    if (st.curVal !== st.startVal) {
      await patchBundleSynapses(st.fromLvl, st.curVal);
    }
  }
  if (singleDragState) {
    const st = singleDragState;
    singleDragState = null;
    if (st.curVal !== st.startVal && currentEdited.mode === 'edit') {
      await saveFimChanges();
    }
  }
});

// --- Interactive Single-FIM Synapse Schematic & Dragging ---
let singleDragState = null;

function startSingleWireDrag(ev, slotKey, inputId) {
  ev.preventDefault();
  ev.stopPropagation();
  const inp = document.getElementById(inputId);
  singleDragState = {
    slotKey,
    inputId,
    startX: ev.clientX,
    startVal: Number(inp.value) || 1,
    curVal: Number(inp.value) || 1,
  };
}

function onSynModeChange() {
  const mode = Number(document.getElementById('edSynMode').value);
  document.getElementById('edRowB2').style.display = mode === 3 ? '' : 'none';
  redrawSingleFimSvg();
}

function redrawSingleFimSvg() {
  const svg = document.getElementById('singleFimSvg');
  const lvl = Number(document.getElementById('edLevel').value);
  const fidText = document.getElementById('edFimId').value;
  const mode = Number(document.getElementById('edSynMode').value);
  const s1 = document.getElementById('edSrc1').value || '?';
  const s2 = document.getElementById('edSrc2').value || '?';
  const s3 = document.getElementById('edSrc3').value || '?';
  const c1 = Math.max(1, Number(document.getElementById('edCnt1').value) || 1);
  const c2 = Math.max(1, Number(document.getElementById('edCnt2').value) || 1);
  const c3 = Math.max(1, Number(document.getElementById('edCnt3').value) || 1);
  const outQ = Math.max(1, Number(document.getElementById('edOutQuota').value) || 1);

  if (currentEdited.level === 0) {
    const exp = currentEdited.explain;
    const ch = exp ? exp.text : '?';
    const b1 = exp ? exp.out_by_slot['Б1'] : 0;
    const b2 = exp ? exp.out_by_slot['Б2'] : 0;
    const b3 = exp ? exp.out_by_slot['Б3'] : 0;
    svg.innerHTML = `
      <rect x="20" y="65" width="120" height="80" fill="#f1f5f9" stroke="#334155" stroke-width="1.6"/>
      <text x="32" y="95" font-family="DejaVu Sans Mono, monospace" font-size="14" font-weight="700" fill="#1a5bb8">L0 «${esc(ch)}»</text>
      <text x="32" y="116" font-family="Tahoma, sans-serif" font-size="10.5" fill="#475569">Элемент #${exp ? exp.fim_id : 0}</text>
      <text x="32" y="133" font-family="Tahoma, sans-serif" font-size="10" fill="#475569">Такт 84 мс</text>

      <line x1="140" y1="85" x2="295" y2="45" stroke="#1a5bb8" stroke-width="2" style="cursor:ew-resize;" onmousedown="startSingleWireDrag(event, 'out', 'edOutQuota')"/>
      <line x1="140" y1="105" x2="295" y2="105" stroke="#7c3aed" stroke-width="2" style="cursor:ew-resize;" onmousedown="startSingleWireDrag(event, 'out', 'edOutQuota')"/>
      <line x1="140" y1="125" x2="295" y2="165" stroke="#d97706" stroke-width="2" style="cursor:ew-resize;" onmousedown="startSingleWireDrag(event, 'out', 'edOutQuota')"/>

      <rect x="295" y="26" width="205" height="38" fill="#e0f2fe" stroke="#0284c7"/>
      <text x="305" y="49" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="700" fill="#0369a1">на Б1 L1 (факт): ${b1} син.</text>

      <rect x="295" y="86" width="205" height="38" fill="#f3e8ff" stroke="#7c3aed"/>
      <text x="305" y="109" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="700" fill="#6d28d9">на Б2 L1 (факт): ${b2} син.</text>

      <rect x="295" y="146" width="205" height="38" fill="#fef3c7" stroke="#d97706"/>
      <text x="305" y="169" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="700" fill="#b45309">на Б3 L1 (факт): ${b3} син.</text>

      <text x="155" y="202" font-family="DejaVu Sans Mono, monospace" font-size="11" font-weight="700" fill="#167948">
        Квота: ${outQ} син. | Факт всего: ${b1+b2+b3} син. (тяните линию)
      </text>
    `;
    return;
  }

  const slots = [
    { name: 'Б1', src: s1, cnt: c1, y: 38, color: '#0284c7', bg: '#e0f2fe', inp: 'edCnt1', key: 'b1' },
    { name: 'Б2', src: s2, cnt: c2, y: 106, color: '#7c3aed', bg: '#f3e8ff', inp: 'edCnt2', key: 'b2', disabled: mode === 2 },
    { name: 'Б3', src: s3, cnt: c3, y: 174, color: '#d97706', bg: '#fef3c7', inp: 'edCnt3', key: 'b3' },
  ];

  let html = '';
  slots.forEach(sl => {
    const op = sl.disabled ? '0.35' : '1';
    const wires = sl.disabled ? 0 : Math.min(7, Math.max(1, sl.cnt));
    html += `<g opacity="${op}">
      <rect x="8" y="${sl.y - 18}" width="110" height="36" fill="#f8fafc" stroke="#64748b"/>
      <text x="14" y="${sl.y + 4}" font-family="DejaVu Sans Mono, monospace" font-size="10.5" font-weight="700" fill="#1e293b">${esc(sl.src.slice(0,13))}</text>
    `;
    for (let w = 0; w < wires; w++) {
      const off = (w - (wires - 1) / 2) * 3.5;
      html += `<line x1="118" y1="${sl.y + off}" x2="210" y2="${sl.y + off}" stroke="${sl.color}" stroke-width="1.4"/>`;
    }
    if (!sl.disabled) {
      html += `
        <rect x="120" y="${sl.y - 16}" width="88" height="32" fill="transparent" style="cursor:ew-resize;"
              onmousedown="startSingleWireDrag(event, '${sl.key}', '${sl.inp}')"/>
        <rect x="134" y="${sl.y - 10}" width="60" height="16" fill="#fffdf5" stroke="${sl.color}"/>
        <text x="139" y="${sl.y + 2}" font-family="DejaVu Sans Mono, monospace" font-size="10" font-weight="700" fill="${sl.color}"
              style="cursor:ew-resize;" onmousedown="startSingleWireDrag(event, '${sl.key}', '${sl.inp}')">${sl.cnt} син.</text>
      `;
    } else {
      html += `<text x="132" y="${sl.y + 3}" font-family="Tahoma, sans-serif" font-size="9.5" fill="#64748b">без синапса</text>`;
    }
    html += `
      <rect x="210" y="${sl.y - 18}" width="54" height="36" fill="${sl.bg}" stroke="${sl.color}" stroke-width="1.4"/>
      <text x="228" y="${sl.y + 4}" font-family="DejaVu Sans Mono, monospace" font-size="12" font-weight="700" fill="${sl.color}">${sl.name}</text>
    `;
    if (!sl.disabled) {
      html += `<line x1="264" y1="${sl.y}" x2="328" y2="106" stroke="#475569" stroke-width="1.5"/>`;
    }
    html += `</g>`;
  });

  // Head neuron Г and outgoing bundle to L+1
  const outWires = Math.min(7, Math.max(2, Math.round(Math.log10(Math.max(1, outQ)) * 2) + 2));
  for (let w = 0; w < outWires; w++) {
    const oy = 106 + (w - (outWires - 1) / 2) * 8;
    html += `<line x1="398" y1="106" x2="508" y2="${oy}" stroke="#167948" stroke-width="1.4"
                   style="cursor:ew-resize;" onmousedown="startSingleWireDrag(event, 'out', 'edOutQuota')"/>`;
  }
  html += `
    <circle cx="362" cy="106" r="34" fill="#dcfce7" stroke="#15803d" stroke-width="2"/>
    <text x="355" y="103" font-family="DejaVu Sans Mono, monospace" font-size="16" font-weight="700" fill="#14532d">Г</text>
    <text x="336" y="121" font-family="DejaVu Sans Mono, monospace" font-size="9.5" fill="#14532d">L${lvl}${esc(fidText)}</text>

    <rect x="408" y="88" width="102" height="36" fill="#fffdf5" stroke="#167948"
          style="cursor:ew-resize;" onmousedown="startSingleWireDrag(event, 'out', 'edOutQuota')"/>
    <text x="414" y="103" font-family="DejaVu Sans Mono, monospace" font-size="10" font-weight="700" fill="#167948"
          style="cursor:ew-resize;" onmousedown="startSingleWireDrag(event, 'out', 'edOutQuota')">${outQ} син. &rarr; L${lvl+1}</text>
    <text x="414" y="117" font-family="Tahoma, sans-serif" font-size="9" fill="#475569">(тяните мышкой)</text>
  `;
  svg.innerHTML = html;
}

function loadFimIntoEditor(exp) {
  if (!exp) return;
  currentEdited = { level: exp.level, fim_id: exp.fim_id, mode: 'edit', explain: exp };
  if (exp.level === 0) {
    document.getElementById('fimBoxTitle').textContent = `2. Схема синапсов элемента L0 #${exp.fim_id} «${exp.text}»`;
    document.getElementById('edFimId').value = 'L0#' + exp.fim_id;
    document.getElementById('edSlotsTable').style.display = 'none';
    document.getElementById('edOutQuota').value = exp.outgoing_synapses_per_fim || netConfig.synapses_per_element_l0;
    document.getElementById('edWin').value = netConfig.tick_ms;
    document.getElementById('btnDisableFim').disabled = true;
    document.getElementById('btnSaveFim').disabled = false;
    renderExplanation(exp);
    redrawSingleFimSvg();
    return;
  }

  document.getElementById('edSlotsTable').style.display = '';
  document.getElementById('fimBoxTitle').textContent = `2. Схема синапсов ФИМ L${exp.level} #${exp.fim_id} «${exp.text}»`;
  document.getElementById('edLevel').value = String(exp.level);
  document.getElementById('edLevel').disabled = true;
  document.getElementById('edFimId').value = '#' + exp.fim_id;
  document.getElementById('btnDisableFim').disabled = false;
  document.getElementById('btnSaveFim').disabled = false;

  const ch = exp.children || [];
  const isTwo = ch.length === 2;
  document.getElementById('edSynMode').value = isTwo ? '2' : '3';
  const fill = (idx, sEl, cEl) => {
    const item = ch[idx];
    if (!item) {
      document.getElementById(sEl).value = '';
      document.getElementById(cEl).value = 1;
      return;
    }
    document.getElementById(sEl).value = item.level === 0 ? item.text : ('#' + item.fim_id + ' (' + item.text + ')');
    document.getElementById(cEl).value = item.synapse_count || 1;
  };
  if (isTwo) {
    fill(0, 'edSrc1', 'edCnt1');
    document.getElementById('edSrc2').value = '';
    fill(1, 'edSrc3', 'edCnt3');
  } else {
    fill(0, 'edSrc1', 'edCnt1');
    fill(1, 'edSrc2', 'edCnt2');
    fill(2, 'edSrc3', 'edCnt3');
  }
  document.getElementById('edOutQuota').value = exp.outgoing_synapses_per_fim || 500;
  document.getElementById('edWin').value = exp.custom_window_ms || '';
  onSynModeChange();
  renderExplanation(exp);
}

function openCreateFim() {
  currentEdited = { level: Math.max(1, currentMemLvl), fim_id: null, mode: 'create', explain: null };
  document.getElementById('edSlotsTable').style.display = '';
  document.getElementById('fimBoxTitle').textContent = '2. Создание нового ФИМа';
  document.getElementById('edLevel').disabled = false;
  document.getElementById('edLevel').value = String(Math.max(1, currentMemLvl));
  document.getElementById('edFimId').value = 'Новый';
  document.getElementById('edSynMode').value = '3';
  document.getElementById('edSrc1').value = '';
  document.getElementById('edSrc2').value = '';
  document.getElementById('edSrc3').value = '';
  document.getElementById('edCnt1').value = 1;
  document.getElementById('edCnt2').value = 1;
  document.getElementById('edCnt3').value = 1;
  document.getElementById('edOutQuota').value = 500;
  document.getElementById('edWin').value = '';
  document.getElementById('btnSaveFim').disabled = true;
  document.getElementById('btnDisableFim').disabled = true;
  onSynModeChange();
}

function cleanSourceVal(v) {
  const s = String(v || '').trim();
  const m = s.match(/^(#\d+)\s*\(/);
  return m ? m[1] : s;
}

function collectFimPayload() {
  const lvl = Number(document.getElementById('edLevel').value);
  const mode = Number(document.getElementById('edSynMode').value);
  const s1 = cleanSourceVal(document.getElementById('edSrc1').value);
  const s2 = cleanSourceVal(document.getElementById('edSrc2').value);
  const s3 = cleanSourceVal(document.getElementById('edSrc3').value);
  const c1 = Number(document.getElementById('edCnt1').value);
  const c2 = Number(document.getElementById('edCnt2').value);
  const c3 = Number(document.getElementById('edCnt3').value);
  const winRaw = document.getElementById('edWin').value.trim();

  return {
    level: currentEdited.level === 0 ? 0 : lvl,
    fim_id: currentEdited.fim_id,
    sources: mode === 2 ? [s1, s3] : [s1, s2, s3],
    synapse_slots: mode === 2 ? [0, 2] : [0, 1, 2],
    synapse_counts: mode === 2 ? [c1, c3] : [c1, c2, c3],
    outgoing_synapses_per_fim: Number(document.getElementById('edOutQuota').value),
    custom_window_ms: winRaw ? Number(winRaw) : null,
    probe_text: document.getElementById('inputText').value,
  };
}

async function saveFimChanges() {
  if (currentEdited.fim_id === null) return;
  const payload = collectFimPayload();
  const res = await fetch('/api/fim/update', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) {
    showMsg('fimEditMsg', data.error || 'Ошибка сохранения', true);
    return;
  }
  loadFimIntoEditor(data.fim);
  if (data.config) applyConfig(data.config);
  showMsg('fimEditMsg', `Сохранено L${data.fim.level}#${data.fim.fim_id}.`);
  await sendText(false, data.fim.level, data.fim.fim_id);
  await searchMem();
}

async function createNewFim() {
  const payload = collectFimPayload();
  if (payload.level === 0) payload.level = 1;
  const res = await fetch('/api/fim/create', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) {
    showMsg('fimEditMsg', data.error || 'Ошибка создания', true);
    return;
  }
  loadFimIntoEditor(data.fim);
  if (data.config) applyConfig(data.config);
  showMsg('fimEditMsg', `Создан ФИМ L${data.fim.level}#${data.fim.fim_id} «${data.fim.text}»`);
  await switchMemLevel(data.fim.level);
  await sendText(false, data.fim.level, data.fim.fim_id);
}

async function disableCurrentFim() {
  if (currentEdited.level < 1 || currentEdited.fim_id === null) return;
  const res = await fetch('/api/fim/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ level: currentEdited.level, fim_id: currentEdited.fim_id })
  });
  const data = await res.json();
  if (res.ok) {
    showMsg('fimEditMsg', `ФИМ L${currentEdited.level}#${currentEdited.fim_id} отключён.`);
    await loadConfig();
    await searchMem();
    await sendText(false);
  }
}

async function inspectFim(lvl, fid) {
  const probe = document.getElementById('inputText').value || '';
  const res = await fetch(`/api/fim?lvl=${lvl}&id=${fid}&probe=${encodeURIComponent(probe)}`);
  const exp = await res.json();
  loadFimIntoEditor(exp);
  if (exp.simulation && lvl >= 1) loadSimulation(exp.simulation, false);
}

function renderExplanation(exp) {
  const meta = document.getElementById('edMetaText');
  const pBox = document.getElementById('edParentsList');
  const tBox = document.getElementById('edTreeList');
  if (exp.level === 0) {
    meta.innerHTML = `Элемент L0 #${exp.fim_id} «<b>${esc(exp.text)}</b>» · Исходящих синапсов в памяти L1: <b>${exp.actual_out_synapses}</b> (Б1:${exp.out_by_slot['Б1']}, Б2:${exp.out_by_slot['Б2']}, Б3:${exp.out_by_slot['Б3']})`;
    pBox.innerHTML = '<div style="font-weight:700; font-size:11px;">Куда идут синапсы на Уровень 1:</div>' +
      (exp.parents || []).map(p =>
        `<span class="tag hit" style="cursor:pointer;" onclick="inspectFim(${p.level}, ${p.fim_id})">${p.target_slot}: L1#${p.fim_id} «${esc(p.text)}» (${p.synapse_count} син.)</span>`
      ).join('');
    tBox.innerHTML = '';
    return;
  }
  meta.innerHTML = `Этап: <b>${esc(exp.training_stage || '—')}</b> · Окно W: <b>${exp.window_ms} мс</b> · Исх. синапсов от Г вверх (факт): <b>${exp.actual_out_synapses}</b>`;
  pBox.innerHTML = (exp.parents && exp.parents.length)
    ? `<div style="font-weight:700; font-size:11px;">Исходящие синапсы от Г вверх на L${exp.level + 1}:</div>` +
      exp.parents.map(p =>
        `<span class="tag hit" style="cursor:pointer;" onclick="inspectFim(${p.level}, ${p.fim_id})">${p.target_slot} в L${p.level}#${p.fim_id} «${esc(p.text)}» (${p.synapse_count} син.)</span>`
      ).join('')
    : '<span class="muted">Нет исходящих синапсов на уровень выше</span>';

  function buildTree(node, isRoot) {
    const slotTag = node.slot ? `<span class="tag">${esc(node.slot)} (${node.synapse_count || 1} син.)</span>` : '';
    let inner = `<div class="${isRoot ? 'tree-root' : 'tree-node'}">
      <div class="tree-card" onclick="inspectFim(${node.level}, ${node.fim_id})">
        ${slotTag}<b class="mono">L${node.level}#${node.fim_id}</b> <span class="mono">«${esc(node.text)}»</span>
      </div>`;
    if (node.children && node.children.length) inner += node.children.map(ch => buildTree(ch, false)).join('');
    inner += '</div>';
    return inner;
  }
  tBox.innerHTML = '<div style="font-weight:700; font-size:11px; margin-bottom:2px;">От каких элементов / ФИМ пришли сигналы:</div>' + buildTree(exp, true);
}

// --- Level 0 Matrix & Level 1..9 Search ---
async function switchMemLevel(lvl) {
  currentMemLvl = lvl;
  document.querySelectorAll('#memTabs button').forEach((b, i) => b.classList.toggle('active', i === lvl));
  if (netConfig) draw9LevelSchematic();
  await searchMem();
}

function renderL0Matrix() {
  if (!netConfig || !netConfig.l0_elements) return;
  const q = (document.getElementById('memSearch').value || '').trim().toLowerCase();
  const list = netConfig.l0_elements.filter(e =>
    !q || e.char.toLowerCase().includes(q) || String(e.node_id) === q
  );
  document.getElementById('memCountInfo').textContent = `${list.length} / 41 элементов L0`;
  document.getElementById('l0MatrixWrap').innerHTML = '<div class="l0-grid">' + list.map(e => {
    const act = currentEdited.level === 0 && currentEdited.fim_id === e.node_id ? 'active' : '';
    const chDisp = e.char === ' ' ? '[_]' : esc(e.char);
    return `<div class="l0-cell ${act}" onclick="inspectFim(0, ${e.node_id})" title="L0 #${e.node_id}: Б1=${e.b1}, Б2=${e.b2}, Б3=${e.b3}">
      <b>${chDisp}</b> #${e.node_id}<br><span style="color:#1a5bb8; font-size:10px;">${e.out_synapses} син.</span>
    </div>`;
  }).join('') + '</div>';
}

async function searchMem() {
  const l0Wrap = document.getElementById('l0MatrixWrap');
  const tWrap = document.getElementById('memTableWrap');
  if (currentMemLvl === 0) {
    l0Wrap.style.display = '';
    tWrap.style.display = 'none';
    renderL0Matrix();
    return;
  }
  l0Wrap.style.display = 'none';
  tWrap.style.display = '';
  const q = document.getElementById('memSearch').value || '';
  const res = await fetch(`/api/level?lvl=${currentMemLvl}&limit=80&q=${encodeURIComponent(q)}`);
  const data = await res.json();
  document.getElementById('memCountInfo').textContent = `${data.matched_count} / ${data.total_consolidated.toLocaleString('ru-RU')}`;
  const tbody = document.getElementById('memTableBody');
  if (!data.fims || !data.fims.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">Ничего не найдено</td></tr>';
    return;
  }
  tbody.innerHTML = data.fims.map(f => {
    const cnt = (f.synapse_counts || []).join('/');
    return `<tr class="clickable" onclick="inspectFim(${currentMemLvl}, ${f.fim_id})">
      <td class="mono">#${f.fim_id}</td>
      <td class="mono"><b>${esc(f.text)}</b></td>
      <td class="mono">${f.synapse_slots.length} (${cnt})</td>
      <td class="mono">${f.window_ms} мс</td>
      <td class="mono">${f.recognitions}</td>
    </tr>`;
  }).join('');
}

// --- Simulation & Trace ---
async function sendText(learn, focusLvl=null, focusFid=null) {
  const t0 = performance.now();
  const text = document.getElementById('inputText').value;
  const res = await fetch('/api/check', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, learn, focus_level: focusLvl, focus_fim_id: focusFid})
  });
  const data = await res.json();
  document.getElementById('latencyBadge').textContent = `${(performance.now() - t0).toFixed(1)} мс · «${data.clean_text || '—'}»`;

  const ansMain = document.getElementById('topAnswerMain');
  const ansMeta = document.getElementById('topAnswerMeta');
  if (!data.top_level) {
    ansMain.textContent = 'Нет сработавших ФИМ';
    ansMeta.innerHTML = '';
  } else {
    ansMain.textContent = `«${data.top_response_text}»`;
    ansMeta.innerHTML = `Ур. ${data.top_level}: ` + data.top_fims.map(f =>
      `<span class="tag hit" style="cursor:pointer;" onclick="inspectFim(${data.top_level}, ${f.fim_id})">L${data.top_level}#${f.fim_id} «${esc(f.text)}»</span>`
    ).join('');
  }

  const recMain = document.getElementById('recallMain');
  const recMeta = document.getElementById('recallMeta');
  if (data.recalled_level > data.top_level && data.recalled_text) {
    recMain.style.color = '#167948';
    recMain.textContent = `«${data.recalled_text}»`;
    recMeta.innerHTML = `<span class="tag hit" style="cursor:pointer;" onclick="inspectFim(${data.recalled_level}, ${data.recalled_fim_id})">L${data.recalled_level}#${data.recalled_fim_id}</span>`;
  } else {
    recMain.style.color = '#525d6d';
    recMain.textContent = 'Совпадает с верхним сработавшим уровнем';
    recMeta.innerHTML = '';
  }

  if (data.config) applyConfig(data.config);
  const exp = data.recalled_explanation || data.top_explanation;
  if (exp && focusLvl === null) loadFimIntoEditor(exp);
  if (data.simulation) loadSimulation(data.simulation, true);
}

function loadSimulation(sim, resetTime=true) {
  currentSim = sim;
  const sl = document.getElementById('timeSlider');
  sl.max = sim.duration_ms;
  sl.step = sim.tick_ms;
  if (resetTime) { stopPlay(); playheadMs = sim.duration_ms; }
  else playheadMs = Math.min(playheadMs, sim.duration_ms);
  sl.value = playheadMs;
  document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;
  renderTraceTable();
  drawRaster();
}

function renderTraceTable() {
  const tbody = document.getElementById('traceBody');
  if (!currentSim || !currentSim.fim_spikes || !currentSim.fim_spikes.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">Нет сработавших последовательностей.</td></tr>';
    return;
  }
  const q = (document.getElementById('traceSearch').value || '').trim().toLowerCase();
  const rows = currentSim.fim_spikes.filter(sp => {
    if (currentTraceFilterLvl > 0 && sp.level !== currentTraceFilterLvl) return false;
    if (q && !sp.text.toLowerCase().includes(q) && !('#' + sp.fim_id).includes(q)) return false;
    return true;
  });
  tbody.innerHTML = rows.map(sp => {
    const synHtml = (sp.sources_detail || []).map(sd => {
      const cls = sd.fired ? 'hit' : 'miss';
      const prefix = sd.child_level === 0 ? 'L0' : `L${sd.child_level}#${sd.child_id}`;
      return `<span class="tag ${cls}">${sd.neuron}:${prefix} «${esc(sd.child_text)}» (${sd.synapse_count} син.)</span>`;
    }).join(' ');
    const upHtml = (sp.parents_preview || []).length
      ? sp.parents_preview.map(p => `<span class="tag" style="cursor:pointer;" onclick="event.stopPropagation(); inspectFim(${p.level}, ${p.fim_id})">${p.target_slot}→L${p.level}#${p.fim_id}</span>`).join(' ')
      : '<span class="muted">—</span>';
    const ruleTag = sp.confidence < 0.99
      ? '<span class="tag assoc">2-из-3</span>'
      : `<span class="tag hit">${sp.synapse_slots.length}-из-${sp.synapse_slots.length}</span>`;
    return `<tr class="clickable" onclick="inspectFim(${sp.level}, ${sp.fim_id})">
      <td class="mono">${sp.time_ms}мс / Т${sp.tick_index} / И${sp.instinct_cycle}</td>
      <td class="mono"><b>L${sp.level}#${sp.fim_id}</b></td>
      <td class="mono"><b>«${esc(sp.text)}»</b></td>
      <td>${synHtml}</td>
      <td class="mono">${sp.window_ms} мс</td>
      <td>${upHtml}</td>
      <td>${ruleTag}</td>
    </tr>`;
  }).join('');
}

function resetPlayhead() {
  stopPlay(); playheadMs = 0;
  document.getElementById('timeSlider').value = 0;
  document.getElementById('clockLabel').textContent = 't = 0 мс';
  drawRaster();
}
function stepPlayhead(dir) {
  if (!currentSim) return;
  stopPlay();
  playheadMs = Math.max(0, Math.min(currentSim.duration_ms, playheadMs + dir * currentSim.tick_ms));
  document.getElementById('timeSlider').value = playheadMs;
  document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;
  drawRaster();
}
function scrubTime(val) {
  playheadMs = Number(val);
  document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;
  drawRaster();
}
function togglePlay() {
  if (!currentSim) return;
  if (isPlaying) { stopPlay(); return; }
  if (playheadMs >= currentSim.duration_ms) playheadMs = 0;
  isPlaying = true;
  document.getElementById('playBtn').textContent = '|| Пауза';
  playTimer = setInterval(() => {
    playheadMs = Math.min(currentSim.duration_ms, playheadMs + currentSim.tick_ms);
    document.getElementById('timeSlider').value = playheadMs;
    document.getElementById('clockLabel').textContent = `t = ${playheadMs} мс`;
    drawRaster();
    if (playheadMs >= currentSim.duration_ms) stopPlay();
  }, 160);
}
function stopPlay() {
  isPlaying = false;
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  document.getElementById('playBtn').textContent = '▶ Пуск';
}

function drawRaster() {
  if (!currentSim) return;
  const c = document.getElementById('rasterCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 500, H = 290;
  c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const dur = Math.max(currentSim.duration_ms, currentSim.instinct_cycle_ms);
  const padL = 38, padR = 14, padT = 14, padB = 24;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xOf = t => padL + (t / dur) * plotW;

  ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  for (let l = 0; l <= 9; l++) {
    const y = padT + ((9 - l) + 0.5) * (plotH / 10);
    ctx.strokeStyle = '#eef2f6';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#334155'; ctx.font = '10.5px "DejaVu Sans Mono", monospace';
    ctx.fillText('L' + l, 8, y + 4);
  }

  ctx.strokeStyle = '#d97706'; ctx.setLineDash([4, 3]);
  (currentSim.instinct_pulses_ms || []).forEach(tp => {
    const x = xOf(tp);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
  });
  ctx.setLineDash([]);

  ctx.strokeStyle = '#64748b'; ctx.lineWidth = 1.6;
  (currentSim.element_spikes || []).forEach(es => {
    if (es.time_ms > playheadMs) return;
    const x = xOf(es.time_ms), y = padT + 9.5 * (plotH / 10);
    ctx.beginPath(); ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8); ctx.stroke();
    ctx.fillStyle = '#334155'; ctx.font = '9.5px "DejaVu Sans Mono", monospace';
    ctx.fillText(es.char === ' ' ? '_' : es.char, x - 3, y - 10);
  });

  (currentSim.fim_spikes || []).forEach(sp => {
    if (sp.time_ms > playheadMs) return;
    const x = xOf(sp.time_ms), y = padT + ((9 - sp.level) + 0.5) * (plotH / 10);
    ctx.strokeStyle = sp.confidence < 0.99 ? '#d97706' : '#1a5bb8';
    ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9); ctx.stroke();
  });

  ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 1.3;
  ctx.beginPath(); ctx.moveTo(xOf(playheadMs), padT); ctx.lineTo(xOf(playheadMs), padT + plotH); ctx.stroke();

  ctx.fillStyle = '#334155'; ctx.font = '10px "DejaVu Sans Mono", monospace';
  ctx.fillText('0 мс', padL, H - 6);
  ctx.fillText(dur + ' мс', W - 54, H - 6);
}

window.addEventListener('resize', () => drawRaster());
loadConfig().then(() => switchMemLevel(0)).then(() => setExample('мама раму'));
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
            lvl = int(qs.get("lvl", ["0"])[0])
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
            if lvl >= 1:
                sim_input = probe_txt if probe_txt.strip() else NET.decode_fim(lvl, fid)
                exp["simulation"] = SIM.run(sim_input, target_level=lvl, target_fim_id=fid, learn=False)
            self._send_json(exp)
            return

        if parsed.path in ("/3d", "/3d/", "/snn_onion.html"):
            onion = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snn_onion.html")
            try:
                with open(onion, "rb") as fh:
                    raw = fh.read()
            except OSError:
                raw = "snn_onion.html не найден".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self._send_nocache_headers()
            self.end_headers()
            self.wfile.write(raw)
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
                if fl is not None and ff is not None and int(fl) >= 1 and result.get("clean_text"):
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
                sim = None
                if lvl >= 1:
                    probe_txt = str(req.get("probe_text") or "").strip() or exp["text"]
                    sim = SIM.run(probe_txt, target_level=lvl, target_fim_id=fid, learn=False)
                self._send_json({
                    "fim": exp,
                    "simulation": sim,
                    "summary": NET.summary(),
                    "config": NET.get_config(),
                })
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
