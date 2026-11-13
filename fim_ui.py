"""
Веб-интерфейс для проверки, тестирования и внутренней аналитики 9-уровневой ФИМ-сети.

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
    - пошаговое потоковое срабатывание;
    - совпадения по всем уровням 1..9;
    - внутреннюю аналитику (explain_fim) для верхнего узнанного ФИМа.
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

    substring_matches: Dict[int, List[Dict[str, Any]]] = {}
    for lvl in range(1, NET.num_levels + 1):
        matches = []
        for fim in NET.levels[lvl].fims:
            if not fim.consolidated:
                continue
            decoded = NET.decode_fim(lvl, fim.fim_id)
            if decoded in clean:
                matches.append({
                    "fim_id": fim.fim_id,
                    "text": decoded,
                    "recognitions": fim.recognitions,
                })
        if matches:
            matches.sort(key=lambda m: (len(m["text"]), m["recognitions"]), reverse=True)
            substring_matches[lvl] = matches[:20]

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


def get_level_fims(level: int, limit: int = 30) -> Dict[str, Any]:
    """Возвращает список консолидированных ФИМов на выбранном уровне 1..9."""
    if level < 1 or level > NET.num_levels:
        return {"level": level, "fims": []}
    cons = [f for f in NET.levels[level].fims if f.consolidated]
    cons.sort(key=lambda f: (f.recognitions, -f.fim_id), reverse=True)
    items = [
        {
            "fim_id": f.fim_id,
            "level": level,
            "text": NET.decode_fim(level, f.fim_id),
            "recognitions": f.recognitions,
            "synapse_slots": list(f.synapse_slots),
            "created_at_ms": f.created_at_ms,
            "training_context": f.training_context,
            "training_stage": f.training_stage,
        }
        for f in cons[:limit]
    ]
    return {"level": level, "total_consolidated": len(cons), "fims": items}


HTML_PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ФИМ-сеть (9 уровней) — Аналитика и проверка</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 24px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg); color: var(--text);
      line-height: 1.5;
    }
    .container { max-width: 1180px; margin: 0 auto; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    .sub { color: var(--muted); margin-bottom: 20px; font-size: 14px; }
    .card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 18px; margin-bottom: 18px;
    }
    .answer-card {
      background: linear-gradient(135deg, #0f2942 0%, #1e293b 100%);
      border: 1px solid var(--accent);
    }
    .answer-text {
      font-size: 22px; font-weight: 700; color: #f8fafc;
      margin: 6px 0 10px; word-break: break-word;
    }
    .answer-meta { font-size: 14px; color: #7dd3fc; font-weight: 600; }
    .recall-box { margin-top: 12px; padding-top: 12px; border-top: 1px solid #334155; }
    .recall-text { font-size: 16px; color: #bbf7d0; word-break: break-word; margin-top: 4px; }
    textarea {
      width: 100%; min-height: 82px; resize: vertical;
      background: #0b1220; color: var(--text);
      border: 1px solid var(--border); border-radius: 8px;
      padding: 12px; font-size: 15px; font-family: inherit;
    }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
    button {
      background: var(--accent); color: #082f49; border: none;
      border-radius: 8px; padding: 9px 14px; font-weight: 600;
      cursor: pointer; font-size: 13px;
    }
    button.secondary { background: #334155; color: var(--text); }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(215px, 1fr)); gap: 10px; }
    .lvl-box {
      background: #0b1220; border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 12px; font-size: 13px;
      cursor: pointer; transition: border-color .15s;
    }
    .lvl-box:hover, .lvl-box.active { border-color: var(--accent); }
    .lvl-title { font-weight: 700; color: var(--accent); margin-bottom: 4px; }
    .pill {
      display: inline-block; padding: 3px 9px; border-radius: 999px;
      font-size: 12px; margin: 3px 4px 3px 0; cursor: pointer;
      border: 1px solid var(--border); background: #0b1220; color: var(--text);
    }
    .pill:hover { border-color: var(--accent); }
    .pill.rec { border-color: var(--green); color: #86efac; }
    .pill.cons { border-color: var(--amber); color: #fde68a; }
    .section-title { font-size: 16px; font-weight: 600; margin: 0 0 10px; }
    .empty { color: var(--muted); font-size: 14px; }
    .tree-node {
      margin-left: 18px; padding-left: 12px; border-left: 2px solid #334155;
      margin-top: 6px; font-size: 13px;
    }
    .tree-head {
      background: #0b1220; border: 1px solid #334155; border-radius: 6px;
      padding: 6px 10px; display: inline-block; cursor: pointer;
    }
    .tree-head:hover { border-color: var(--accent); }
    .ctx-box {
      background: #0b1220; border: 1px solid #334155; border-radius: 8px;
      padding: 10px 12px; font-size: 13px; margin: 8px 0; color: #cbd5e1;
    }
    .fim-row {
      background: #0b1220; border: 1px solid #334155; border-radius: 8px;
      padding: 8px 12px; margin-bottom: 6px; cursor: pointer; font-size: 13px;
    }
    .fim-row:hover { border-color: var(--accent); }
    .nengo-toolbar {
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
      margin-bottom: 12px; background: #0b1220; border: 1px solid #334155;
      border-radius: 8px; padding: 8px 12px;
    }
    .nengo-toolbar button { padding: 6px 12px; font-size: 13px; }
    .nengo-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    }
    @media (max-width: 900px) { .nengo-grid { grid-template-columns: 1fr; } }
    .canvas-box {
      background: #070d19; border: 1px solid #334155; border-radius: 8px; padding: 10px;
    }
    .canvas-title {
      font-size: 13px; font-weight: 600; color: #7dd3fc; margin-bottom: 6px;
      display: flex; justify-content: space-between; align-items: center;
    }
    canvas { width: 100%; display: block; border-radius: 6px; background: #050912; }
    .legend { font-size: 11px; color: #94a3b8; display: flex; gap: 10px; flex-wrap: wrap; }
  </style>
</head>
<body>
<div class="container">
  <h1>ФИМ-Nengo: спайковый симулятор и анализатор 9-уровневой сети</h1>
  <div class="sub">Спайковая визуализация в стиле Nengo для Linux: генератор инстинкта 252 мс, растр спайков уровней 0–9, осциллограф мембранных потенциалов Б1/Б2/Б3/Г (порог 2.0, ассоциативное узнавание 2-из-3) и дерево памяти.</div>

  <div class="card">
    <textarea id="inputText" placeholder="Например: привет"></textarea>
    <div class="row">
      <button onclick="sendText(false)">Проверить узнавание</button>
      <button class="secondary" onclick="sendText(true)">Подать и дообучить</button>
      <button class="secondary" onclick="setExample('привет')">привет</button>
      <button class="secondary" onclick="setExample('ну привет')">ну привет</button>
      <button class="secondary" onclick="setExample('мама мыла раму')">мама мыла раму</button>
      <button class="secondary" onclick="setExample('мама раму')">мама раму (2 из 3)</button>
      <button class="secondary" onclick="setExample('дом у берега')">дом у берега</button>
      <button class="secondary" onclick="setExample('утром светило яркое солнце и пели птицы.')">утром светило яркое солнце...</button>
    </div>
  </div>

  <div class="card answer-card">
    <div class="section-title">Ответ нейросети (максимальный уровень узнавания)</div>
    <div id="topAnswer" class="empty">Введите фразу и нажмите «Проверить узнавание».</div>
  </div>

  <div class="card">
    <div class="section-title">Nengo-симулятор ФИМ (спайковый граф, растр спайков и осциллограф мембран V(t))</div>
    <div class="nengo-toolbar">
      <button id="btnPlay" onclick="simPlay()">▶ Пуск</button>
      <button class="secondary" id="btnPause" onclick="simPause()">⏸ Пауза</button>
      <button class="secondary" id="btnEnd" onclick="simEnd()">⏭ В конец</button>
      <input type="range" id="simScrub" min="0" max="504" value="0" style="flex:1;min-width:160px;" oninput="onScrub(this.value)" />
      <span id="simClock" style="font-family:monospace;font-size:13px;color:#7dd3fc;">t = 0 / 0 мс</span>
      <span id="simTargetBadge" style="font-size:12px;color:#cbd5e1;">Цель осциллографа: —</span>
    </div>
    <div class="canvas-box" style="margin-bottom:12px;">
      <div class="canvas-title">
        <span>1. Топология сети и прохождение спайков (Инстинкт 252 мс + активный ФИМ Б1/Б2/Б3 → Г)</span>
        <span class="legend">
          <span style="color:#f87171;">● Инстинкт 252мс</span>
          <span style="color:#38bdf8;">● Базовый нейрон Б</span>
          <span style="color:#4ade80;">● Полный спайк (1.0)</span>
          <span style="color:#fbbf24;">● Ассоциация 2-из-3 (0.67)</span>
        </span>
      </div>
      <canvas id="networkCanvas" width="1120" height="230"></canvas>
    </div>
    <div class="nengo-grid">
      <div class="canvas-box">
        <div class="canvas-title">
          <span>2. Растр спайков (Уровни 0–9 + импульсы инстинкта 252 мс)</span>
        </div>
        <canvas id="rasterCanvas" width="550" height="250"></canvas>
      </div>
      <div class="canvas-box">
        <div class="canvas-title">
          <span>3. Осциллограф мембранных потенциалов V(t)</span>
          <span class="legend">
            <span style="color:#60a5fa;">— V_Б1</span>
            <span style="color:#c084fc;">— V_Б2</span>
            <span style="color:#34d399;">— V_Б3</span>
            <span style="color:#fbbf24;">— V_Г</span>
            <span style="color:#f87171;">--- V_th=2.0</span>
          </span>
        </div>
        <canvas id="oscCanvas" width="550" height="250"></canvas>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="section-title">Внутренняя аналитика ФИМ (происхождение при обучении и дерево переходов)</div>
    <div id="analyticsBox" class="empty">Нажмите на любой ФИМ ниже (или проверьте фразу), чтобы увидеть, из каких дочерних ФИМ и какой последовательности обучения он образовался.</div>
  </div>

  <div class="card">
    <div class="section-title">1. Потоковое срабатывание по уровням (кликните на ФИМ для аналитики)</div>
    <div id="streamOut" class="empty">Введите текст и нажмите «Проверить узнавание».</div>
  </div>

  <div class="card">
    <div class="section-title">2. Совпадения в памяти по уровням (независимо от сдвига)</div>
    <div id="matchOut" class="empty">Здесь появятся все известные ФИМы на уровнях 1–9, найденные во фразе.</div>
  </div>

  <div class="card">
    <div class="section-title">Состояние памяти (уровни 1–9 — кликните на уровень, чтобы прочитать его ФИМы)</div>
    <div id="summaryGrid" class="grid"></div>
    <div id="levelBrowser" style="margin-top:14px;"></div>
  </div>
</div>

<script>
let activeLevel = null;

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setExample(txt) {
  document.getElementById('inputText').value = txt;
  sendText(false);
}

function renderSummary(summary) {
  const grid = document.getElementById('summaryGrid');
  grid.innerHTML = '';
  for (let lvl = 1; lvl <= 9; lvl++) {
    const s = summary[lvl];
    const div = document.createElement('div');
    div.className = 'lvl-box' + (activeLevel === lvl ? ' active' : '');
    div.onclick = () => inspectLevel(lvl);
    div.innerHTML = `<div class="lvl-title">Уровень ${lvl} (${s.window_ms} мс)</div>
      <div>ФИМ: <b>${s.consolidated_fims}</b> / 5000</div>
      <div>Узнаваний: <b>${s.total_recognitions}</b></div>`;
    grid.appendChild(div);
  }
}

async function loadSummary() {
  const res = await fetch('/api/summary');
  const data = await res.json();
  renderSummary(data);
}

async function inspectLevel(lvl) {
  activeLevel = lvl;
  loadSummary();
  const res = await fetch('/api/level?lvl=' + lvl);
  const data = await res.json();
  const box = document.getElementById('levelBrowser');
  if (!data.fims.length) {
    box.innerHTML = `<div class="empty">На уровне ${lvl} пока нет консолидированных ФИМов.</div>`;
    return;
  }
  let html = `<div style="margin-bottom:8px;font-weight:600;color:#7dd3fc;">Уровень ${lvl}: показано ${data.fims.length} из ${data.total_consolidated} ФИМ (кликните по ФИМу для разбора)</div>`;
  for (const f of data.fims) {
    const preview = f.text.length > 260 ? f.text.slice(0, 260) + '...' : f.text;
    const stageBadge = f.training_stage ? ` | ${esc(f.training_stage)}` : '';
    html += `<div class="fim-row" onclick="inspectFim(${lvl}, ${f.fim_id})">
      <b>#${f.fim_id}</b> (узнан ${f.recognitions}x, слоты ${JSON.stringify(f.synapse_slots)}, t=${f.created_at_ms}мс${stageBadge}):
      <div style="color:#f8fafc;margin-top:3px;">«${esc(preview)}»</div>
    </div>`;
  }
  box.innerHTML = html;
}

function renderTree(node) {
  const slotPrefix = node.slot ? `<b>${esc(node.slot)}</b> ← ` : '';
  const slotsStr = node.synapse_slots && node.synapse_slots.length ? ` | слоты ${JSON.stringify(node.synapse_slots)}` : '';
  const clickAttr = node.level > 0 ? `onclick="event.stopPropagation(); inspectFim(${node.level}, ${node.fim_id})"` : '';
  let html = `<div class="tree-node">
    <div class="tree-head" ${clickAttr}>
      ${slotPrefix}<b>Ур.${node.level} #${node.fim_id}</b> «${esc(node.text)}»
      <span style="color:#94a3b8;font-size:12px;">(узнан ${node.recognitions}x${slotsStr})</span>
    </div>`;
  if (node.children && node.children.length) {
    for (const ch of node.children) {
      html += renderTree(ch);
    }
  }
  html += `</div>`;
  return html;
}

function renderExplanation(exp) {
  if (!exp) return;
  const box = document.getElementById('analyticsBox');
  let parentsHtml = '<span class="empty">Пока нет исходящих связей на уровень выше.</span>';
  if (exp.parents && exp.parents.length) {
    parentsHtml = exp.parents.map(p =>
      `<span class="pill rec" onclick="inspectFim(${p.level}, ${p.fim_id})">Ур.${p.level} #${p.fim_id} «${esc(p.text.slice(0, 60))}» (${p.recognitions}x)</span>`
    ).join('');
  }
  box.innerHTML = `
    <div style="font-size:15px;font-weight:700;color:#7dd3fc;">
      Уровень ${exp.level}, ФИМ #${exp.fim_id} — «${esc(exp.text)}»
    </div>
    <div class="ctx-box">
      <div><b>Этап обучения:</b> ${esc(exp.training_stage || '—')} | <b>Момент консолидации:</b> ${exp.created_at_ms} мс | <b>Слоты синапсов:</b> ${JSON.stringify(exp.synapse_slots)} | <b>Узнаваний:</b> ${exp.recognitions}</div>
      <div style="margin-top:4px;"><b>Контекст обучения, в котором записался этот ФИМ:</b> «${esc(exp.training_context || '—')}»</div>
    </div>
    <div style="margin:8px 0;"><b>Куда идёт спайк дальше (родительские ФИМы на уровне ${exp.level + 1}):</b><br/>${parentsHtml}</div>
    <div style="margin-top:8px;"><b>От каких ФИМов снизу он собрался (дерево синапсов Б1, Б2, Б3):</b>
      ${renderTree(exp)}
    </div>
  `;
}

async function inspectFim(lvl, fid) {
  const res = await fetch(`/api/fim?lvl=${lvl}&id=${fid}`);
  const exp = await res.json();
  renderExplanation(exp);
  if (exp.simulation) {
    loadSimulation(exp.simulation, false);
  }
  document.getElementById('analyticsBox').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

// ===== Nengo-стиль симулятор и визуализатор =====
let currentSim = null;
let playheadMs = 0;
let simPlaying = false;
let simRafId = null;
let lastFrameTs = null;

function loadSimulation(sim, autoPlay = true) {
  currentSim = sim;
  if (!sim) {
    playheadMs = 0;
    document.getElementById('simClock').textContent = 't = 0 / 0 мс';
    document.getElementById('simTargetBadge').textContent = 'Цель осциллографа: —';
    drawNengoAll();
    return;
  }
  const scrub = document.getElementById('simScrub');
  scrub.max = sim.duration_ms;
  const tgt = sim.target_fim;
  document.getElementById('simTargetBadge').textContent = tgt
    ? `Цель осциллографа: L${tgt.level} #${tgt.fim_id} «${tgt.text.slice(0, 32)}» (окно ${tgt.window_ms} мс)`
    : 'Цель осциллографа: —';

  if (autoPlay) {
    playheadMs = 0;
    simPlay();
  } else {
    simPause();
    playheadMs = sim.duration_ms;
    scrub.value = playheadMs;
    document.getElementById('simClock').textContent = `t = ${Math.round(playheadMs)} / ${sim.duration_ms} мс`;
    drawNengoAll();
  }
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
  document.getElementById('simScrub').value = playheadMs;
  document.getElementById('simClock').textContent = `t = ${currentSim.duration_ms} / ${currentSim.duration_ms} мс`;
  drawNengoAll();
}

function onScrub(val) {
  if (!currentSim) return;
  simPause();
  playheadMs = Number(val);
  document.getElementById('simClock').textContent = `t = ${Math.round(playheadMs)} / ${currentSim.duration_ms} мс`;
  drawNengoAll();
}

function stepPlayback(ts) {
  if (!simPlaying || !currentSim) return;
  if (lastFrameTs === null) lastFrameTs = ts;
  const elapsedMs = ts - lastFrameTs;
  lastFrameTs = ts;
  // Проигрываем всю симуляцию плавно (~1.6 сек реального времени на 1000 мс модели или быстрее для длинных текстов)
  const speed = Math.max(0.55, currentSim.duration_ms / 1600);
  playheadMs = Math.min(currentSim.duration_ms, playheadMs + elapsedMs * speed);
  document.getElementById('simScrub').value = Math.round(playheadMs);
  document.getElementById('simClock').textContent = `t = ${Math.round(playheadMs)} / ${currentSim.duration_ms} мс`;
  drawNengoAll();
  if (playheadMs < currentSim.duration_ms) {
    simRafId = requestAnimationFrame(stepPlayback);
  } else {
    simPlaying = false;
  }
}

function drawNengoAll() {
  drawNetworkGraph();
  drawRasterPlot();
  drawOscilloscope();
}

function drawNetworkGraph() {
  const canvas = document.getElementById('networkCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (!currentSim) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px system-ui';
    ctx.fillText('Подайте фразу или выберите ФИМ, чтобы увидеть спайковую топологию в стиле Nengo.', 24, H / 2);
    return;
  }

  // Проверяем близость к импульсу инстинкта 252 мс
  let instinctPulse = false;
  for (const tp of currentSim.instinct_pulses_ms) {
    if (Math.abs(playheadMs - tp) <= 45 && playheadMs >= tp - 10) instinctPulse = true;
  }

  // Рисуем 10 колонок: Ур.0 .. Ур.9, а сверху справа блок целевого ФИМ (Б1,Б2,Б3 -> Г)
  const tgt = currentSim.target_fim;
  const maxActiveLvl = Math.max(
    tgt ? tgt.level : 1,
    ...currentSim.fim_spikes.map(s => s.level),
    3
  );
  const cols = Math.min(9, Math.max(4, maxActiveLvl));
  const leftPad = 95, rightPad = 310;
  const colSpan = (W - leftPad - rightPad) / Math.max(1, cols);

  // Генератор инстинкта (слева вверху)
  ctx.save();
  ctx.strokeStyle = instinctPulse ? '#f87171' : '#7f1d1d';
  ctx.lineWidth = instinctPulse ? 3 : 1.5;
  ctx.fillStyle = instinctPulse ? '#450a0a' : '#111827';
  ctx.beginPath();
  ctx.roundRect(14, 16, 130, 44, 8);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = instinctPulse ? '#fca5a5' : '#cbd5e1';
  ctx.font = 'bold 11px system-ui';
  ctx.fillText('Инстинкт 252 мс', 26, 35);
  ctx.font = '10px monospace';
  ctx.fillText(instinctPulse ? 'PULSE ACTIVE' : 'Осциллятор', 26, 50);
  ctx.restore();

  // Собираем спайки до playheadMs по уровням
  const levelPos = {};
  for (let lvl = 0; lvl <= cols; lvl++) {
    const x = leftPad + lvl * colSpan;
    const y = 135;
    levelPos[lvl] = {x, y};
  }

  // Линии связей между уровнями (снизу вверх)
  for (let lvl = 0; lvl < cols; lvl++) {
    const p1 = levelPos[lvl], p2 = levelPos[lvl + 1];
    const hasSpikeUp = currentSim.fim_spikes.some(s => s.level === lvl + 1 && s.time_ms <= playheadMs);
    const partialUp = currentSim.fim_spikes.some(s => s.level === lvl + 1 && s.time_ms <= playheadMs && s.confidence < 0.99);
    ctx.strokeStyle = hasSpikeUp ? (partialUp ? '#fbbf24' : '#4ade80') : '#1e293b';
    ctx.lineWidth = hasSpikeUp ? 2.6 : 1.2;
    ctx.beginPath();
    ctx.moveTo(p1.x + 28, p1.y);
    ctx.lineTo(p2.x - 28, p2.y);
    ctx.stroke();

    // Пунктир от Инстинкта к уровням (тактирование окна)
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = instinctPulse ? 'rgba(248,113,113,0.7)' : 'rgba(100,116,139,0.22)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(80, 60);
    ctx.lineTo(p2.x, p2.y - 26);
    ctx.stroke();
    ctx.restore();
  }

  // Узлы уровней 0..cols
  for (let lvl = 0; lvl <= cols; lvl++) {
    const {x, y} = levelPos[lvl];
    let recentSpike = null;
    if (lvl === 0) {
      for (const es of currentSim.element_spikes) {
        if (es.time_ms <= playheadMs) recentSpike = es;
      }
    } else {
      for (const fs of currentSim.fim_spikes) {
        if (fs.level === lvl && fs.time_ms <= playheadMs) recentSpike = fs;
      }
    }
    const isFresh = recentSpike && (playheadMs - recentSpike.time_ms <= 95);
    const isPartial = recentSpike && recentSpike.confidence !== undefined && recentSpike.confidence < 0.99;

    ctx.fillStyle = isFresh ? (isPartial ? '#78350f' : '#064e3b') : '#0f172a';
    ctx.strokeStyle = recentSpike ? (isPartial ? '#fbbf24' : '#4ade80') : '#334155';
    ctx.lineWidth = isFresh ? 3 : 1.5;
    ctx.beginPath();
    ctx.roundRect(x - 32, y - 26, 64, 52, 10);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 11px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText(lvl === 0 ? 'Ур. 0' : `Ур. ${lvl}`, x, y - 8);

    ctx.fillStyle = '#f8fafc';
    ctx.font = '11px monospace';
    let lbl = '—';
    if (lvl === 0 && recentSpike) lbl = `'${recentSpike.char}'`;
    else if (recentSpike) lbl = recentSpike.text.slice(0, 8);
    ctx.fillText(lbl, x, y + 10);
    if (isPartial) {
      ctx.fillStyle = '#fbbf24';
      ctx.font = '9px monospace';
      ctx.fillText('2/3', x, y + 22);
    }
    ctx.textAlign = 'left';
  }

  // Справа: схема целевого ансамбля ФИМ (Б1, Б2, Б3 -> Г)
  const panelX = W - 285, panelY = 16;
  ctx.fillStyle = '#0f172a';
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.roundRect(panelX, panelY, 270, H - 32, 10);
  ctx.fill(); ctx.stroke();

  if (tgt) {
    ctx.fillStyle = '#7dd3fc';
    ctx.font = 'bold 12px system-ui';
    ctx.fillText(`Ансамбль L${tgt.level} #${tgt.fim_id}: «${tgt.text.slice(0, 18)}»`, panelX + 12, panelY + 20);

    const traces = currentSim.voltage_traces;
    let stepIdx = Math.min(traces.t_grid.length - 1, Math.floor(playheadMs / currentSim.dt_ms));
    if (stepIdx < 0) stepIdx = 0;

    const slotsMap = {};
    for (const s of tgt.sources) slotsMap[s.slot] = s;
    const bVals = [traces.v_b1[stepIdx] || 0, traces.v_b2[stepIdx] || 0, traces.v_b3[stepIdx] || 0];
    const headVal = traces.v_head[stepIdx] || 0;
    const headFired = traces.spike_times_head.some(t => t <= playheadMs);
    const gx = panelX + 215, gy = panelY + 115;

    for (let s = 0; s < 3; s++) {
      const bx = panelX + 46, by = panelY + 55 + s * 58;
      const src = slotsMap[s];
      const active = bVals[s] > 0.15;

      // Синапс к Г
      ctx.strokeStyle = src ? (active ? '#38bdf8' : '#334155') : '#1e293b';
      ctx.lineWidth = active ? 2.5 : 1.2;
      ctx.beginPath();
      ctx.moveTo(bx + 20, by);
      ctx.lineTo(gx - 22, gy);
      ctx.stroke();

      ctx.fillStyle = src ? (active ? '#0c4a6e' : '#111827') : '#090d16';
      ctx.strokeStyle = src ? (active ? '#38bdf8' : '#475569') : '#1e293b';
      ctx.beginPath();
      ctx.arc(bx, by, 18, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = src ? '#f8fafc' : '#475569';
      ctx.font = 'bold 11px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(`Б${s + 1}`, bx, by + 4);
      ctx.textAlign = 'left';
      if (src) {
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '10px monospace';
        ctx.fillText(src.text.slice(0, 12), bx + 24, by - 4);
      }
    }

    // Головной нейрон Г
    ctx.fillStyle = headFired ? '#064e3b' : (headVal >= 1.5 ? '#451a03' : '#111827');
    ctx.strokeStyle = headFired ? '#4ade80' : '#fbbf24';
    ctx.lineWidth = headFired ? 3 : 2;
    ctx.beginPath();
    ctx.arc(gx, gy, 24, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 13px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Г', gx, gy + 4);
    ctx.font = '10px monospace';
    ctx.fillStyle = '#fbbf24';
    ctx.fillText(`V=${headVal.toFixed(2)}`, gx, gy + 38);
    ctx.textAlign = 'left';
  }
}

function drawRasterPlot() {
  const canvas = document.getElementById('rasterCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!currentSim) return;

  const padL = 78, padR = 14, padT = 14, padB = 24;
  const pw = W - padL - padR, ph = H - padT - padB;
  const dur = Math.max(1, currentSim.duration_ms);
  const rows = 11; // 0 = Инстинкт, 1 = L0, 2..10 = L1..L9
  const rowH = ph / rows;

  const xOf = t => padL + (t / dur) * pw;

  // Сетка 252 мс (такт инстинкта)
  for (const tp of currentSim.instinct_pulses_ms) {
    const x = xOf(tp);
    ctx.strokeStyle = 'rgba(248, 113, 113, 0.25)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, H - padB);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  const labels = ['Инстинкт', 'Ур.0 (сим)', 'Ур.1', 'Ур.2', 'Ур.3', 'Ур.4', 'Ур.5', 'Ур.6', 'Ур.7', 'Ур.8', 'Ур.9'];
  ctx.font = '10px monospace';
  for (let r = 0; r < rows; r++) {
    const yMid = padT + (rows - 1 - r + 0.5) * rowH;
    ctx.strokeStyle = '#111827';
    ctx.beginPath();
    ctx.moveTo(padL, yMid + rowH * 0.48);
    ctx.lineTo(W - padR, yMid + rowH * 0.48);
    ctx.stroke();
    ctx.fillStyle = r === 0 ? '#f87171' : '#94a3b8';
    ctx.fillText(labels[r], 6, yMid + 3);
  }

  // Импульсы инстинкта
  for (const tp of currentSim.instinct_pulses_ms) {
    if (tp > playheadMs) continue;
    const x = xOf(tp), yMid = padT + (rows - 1 + 0.5) * rowH;
    ctx.strokeStyle = '#f87171';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(x, yMid - rowH * 0.35);
    ctx.lineTo(x, yMid + rowH * 0.35);
    ctx.stroke();
  }

  // Элементные спайки Ур.0
  for (const es of currentSim.element_spikes) {
    if (es.time_ms > playheadMs) continue;
    const x = xOf(es.time_ms), yMid = padT + (rows - 2 + 0.5) * rowH;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(x, yMid - rowH * 0.32);
    ctx.lineTo(x, yMid + rowH * 0.32);
    ctx.stroke();
  }

  // Спайки ФИМов Ур.1..9
  for (const fs of currentSim.fim_spikes) {
    if (fs.time_ms > playheadMs) continue;
    const r = fs.level + 1;
    if (r >= rows) continue;
    const x = xOf(fs.time_ms), yMid = padT + (rows - 1 - r + 0.5) * rowH;
    const isPartial = fs.confidence < 0.99;
    ctx.strokeStyle = isPartial ? '#fbbf24' : '#4ade80';
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(x, yMid - rowH * 0.36);
    ctx.lineTo(x, yMid + rowH * 0.36);
    ctx.stroke();
    if (fs.level >= 2) {
      ctx.fillStyle = isPartial ? '#fde68a' : '#bbf7d0';
      ctx.font = '9px monospace';
      ctx.fillText(fs.text.slice(0, 10), Math.min(W - 72, x + 3), yMid + 3);
    }
  }

  // Бегунок времени
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
  ctx.fillText(`${dur} мс`, W - 54, H - 6);
}

function drawOscilloscope() {
  const canvas = document.getElementById('oscCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!currentSim || !currentSim.voltage_traces) return;

  const tr = currentSim.voltage_traces;
  const padL = 42, padR = 14, padT = 16, padB = 24;
  const pw = W - padL - padR, ph = H - padT - padB;
  const dur = Math.max(1, currentSim.duration_ms);
  const vMax = 3.5;

  const xOf = t => padL + (t / dur) * pw;
  const yOf = v => padT + ph - (Math.min(vMax, Math.max(0, v)) / vMax) * ph;

  // Горизонтальная сетка потенциалов
  ctx.font = '10px monospace';
  for (const val of [0, 1.0, 2.0, 3.0]) {
    const y = yOf(val);
    ctx.strokeStyle = val === 2.0 ? '#ef4444' : '#1e293b';
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

  // Линии инстинкта 252 мс
  for (const tp of currentSim.instinct_pulses_ms) {
    const x = xOf(tp);
    ctx.strokeStyle = 'rgba(248, 113, 113, 0.2)';
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

  // Бегунок
  const px = xOf(playheadMs);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(px, padT);
  ctx.lineTo(px, H - padB);
  ctx.stroke();

  ctx.fillStyle = '#94a3b8';
  ctx.fillText('0 мс', padL, H - 6);
  ctx.fillText(`${dur} мс`, W - 54, H - 6);
}

async function sendText(learn) {
  const text = document.getElementById('inputText').value;
  const res = await fetch('/api/check', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, learn})
  });
  const data = await res.json();

  const ansEl = document.getElementById('topAnswer');
  if (!data.top_level) {
    ansEl.innerHTML = `<span class="empty">На этом вводе пока нет узнанных ФИМов. Нажмите «Подать и дообучить», чтобы запомнить фразу.</span>`;
  } else {
    const fimsDesc = data.top_fims.map(f => {
      const confBadge = (f.confidence && f.confidence < 0.99) ? ` [2-из-3: ${f.confidence}]` : '';
      return `<span class="pill rec" onclick="inspectFim(${data.top_level}, ${f.fim_id})">#${f.fim_id} «${esc(f.text)}»${confBadge}</span>`;
    }).join(' ');
    let recallHtml = '';
    if (data.recalled_level > data.top_level && data.recalled_text) {
      recallHtml = `
        <div class="recall-box">
          <div class="answer-meta" style="color:#86efac;">
            Продолжение / сюжет на максимальном связанном уровне
            (<span style="cursor:pointer;text-decoration:underline;" onclick="inspectFim(${data.recalled_level}, ${data.recalled_fim_id})">Уровень ${data.recalled_level}, ФИМ #${data.recalled_fim_id}</span>):
          </div>
          <div class="recall-text">«${esc(data.recalled_text)}»</div>
        </div>
      `;
    }
    ansEl.innerHTML = `
      <div class="answer-meta">Узнано из входа (Уровень ${data.top_level}): ${fimsDesc}</div>
      <div class="answer-text">«${esc(data.top_response_text)}»</div>
      ${recallHtml}
    `;
  }

  if (data.recalled_explanation) {
    renderExplanation(data.recalled_explanation);
  } else if (data.top_explanation) {
    renderExplanation(data.top_explanation);
  }

  if (data.simulation) {
    loadSimulation(data.simulation, true);
  }

  const streamEl = document.getElementById('streamOut');
  if (!data.stream_events.length) {
    streamEl.innerHTML = `<span class="empty">Поток: <code>${esc(data.clean_text || '—')}</code>. Потоковых срабатываний не найдено.</span>`;
  } else {
    const byLvl = {};
    for (const ev of data.stream_events) {
      byLvl[ev.level] = byLvl[ev.level] || [];
      byLvl[ev.level].push(ev);
    }
    let html = `<div style="margin-bottom:8px;color:#94a3b8;font-size:13px;">Входной поток: <code>${esc(data.clean_text)}</code></div>`;
    for (const lvl of Object.keys(byLvl).sort((a,b)=>a-b)) {
      html += `<div style="margin-bottom:6px;"><b>Уровень ${lvl}:</b> `;
      for (const ev of byLvl[lvl]) {
        const isPartial = ev.confidence && ev.confidence < 0.99;
        const cls = isPartial ? 'cons' : (ev.type === 'recognition' ? 'rec' : 'cons');
        const label = ev.type === 'recognition'
          ? (isPartial ? `узнан 2-из-3 (${ev.confidence})` : 'узнан')
          : 'записан';
        html += `<span class="pill ${cls}" onclick="inspectFim(${ev.level}, ${ev.fim_id})">#${ev.fim_id} «${esc(ev.text)}» (${label})</span>`;
      }
      html += `</div>`;
    }
    streamEl.innerHTML = html;
  }

  const matchEl = document.getElementById('matchOut');
  const lvls = Object.keys(data.substring_matches).sort((a,b)=>b-a);
  if (!lvls.length) {
    matchEl.innerHTML = `<span class="empty">Совпадений с уже записанными ФИМами не найдено.</span>`;
  } else {
    let html = '';
    for (const lvl of lvls) {
      html += `<div style="margin-bottom:8px;"><b>Уровень ${lvl}:</b> `;
      for (const m of data.substring_matches[lvl]) {
        html += `<span class="pill rec" onclick="inspectFim(${lvl}, ${m.fim_id})">#${m.fim_id} «${esc(m.text)}» (${m.recognitions}x)</span>`;
      }
      html += `</div>`;
    }
    matchEl.innerHTML = html;
  }

  renderSummary(data.summary);
}

loadSummary();
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
            self._send_json(get_level_fims(lvl))
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
