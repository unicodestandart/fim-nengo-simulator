#!/usr/bin/env python3
"""Спайковый симулятор архитектуры ФИМ в стиле Nengo (FIMNengoSimulator)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from fim_algorithm import (
    INSTINCT_CYCLE_MS,
    TICK_MS,
    InstinctNetwork,
    SignalEvent,
    level_window_ms,
    normalize_text,
)


class FIMNengoSimulator:
    """Симулятор спайковой динамики 9-уровневой сети ФИМ с осциллографом мембранных потенциалов."""

    V_THRESHOLD = 2.0
    DT_MS = 12  # Делитель 84 мс (7 * 12) и 252 мс (21 * 12)

    def __init__(self, network: InstinctNetwork) -> None:
        self.net = network

    def _choose_target_fim(
        self,
        fim_spikes: List[Dict[str, Any]],
        target_level: Optional[int],
        target_fim_id: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        if (
            target_level is not None
            and target_fim_id is not None
            and target_level in self.net.levels
            and target_fim_id in self.net.levels[target_level].fims
        ):
            return (target_level, target_fim_id)

        if fim_spikes:
            best = max(
                fim_spikes,
                key=lambda s: (s["level"], s["confidence"], -s["time_ms"]),
            )
            return (int(best["level"]), int(best["fim_id"]))

        # Если ничего не сработало, берём самый верхний существующий ФИМ для демонстрации
        for lvl in range(self.net.num_levels, 0, -1):
            fims = self.net.levels[lvl].fims
            if fims:
                first_id = next(iter(fims))
                return (lvl, first_id)
        return None

    def _build_target_meta(self, lvl: int, fid: int) -> Dict[str, Any]:
        fim = self.net.levels[lvl].fims[fid]
        sources_meta: List[Dict[str, Any]] = []
        for idx, (slot, src_id) in enumerate(zip(fim.synapse_slots, fim.sources)):
            if lvl == 1:
                src_text = self.net.id_to_char.get(src_id, "?")
            else:
                src_text = self.net.decode_fim(lvl - 1, src_id)
            sources_meta.append(
                {
                    "slot": slot,
                    "neuron": f"Б{slot + 1}",
                    "level": lvl - 1,
                    "node_id": src_id,
                    "text": src_text,
                    "source_index": idx,
                }
            )
        return {
            "level": lvl,
            "fim_id": fid,
            "text": self.net.decode_fim(lvl, fid),
            "synapse_slots": list(fim.synapse_slots),
            "window_ms": level_window_ms(lvl),
            "sources": sources_meta,
        }

    def run(
        self,
        text: str,
        target_level: Optional[int] = None,
        target_fim_id: Optional[int] = None,
        learn: bool = False,
    ) -> Dict[str, Any]:
        """Запускает спайковую симуляцию потока текста и возвращает спайки, импульсы инстинкта и V(t)."""
        clean = normalize_text(text)
        element_spikes: List[Dict[str, Any]] = []
        t_cur = 0
        for ch in clean:
            t_cur += TICK_MS
            element_spikes.append(
                {
                    "time_ms": t_cur,
                    "char": ch,
                    "node_id": self.net.char_to_id[ch],
                }
            )

        if learn:
            raw_events = self.net.feed_stream(clean)
        else:
            raw_events = self.net.probe_stream(clean)

        fim_spikes: List[Dict[str, Any]] = []
        for ev in raw_events:
            if ev.level < 1:
                continue
            if ev.event_type not in ("recognition", "consolidation"):
                continue
            fim = self.net.levels[ev.level].fims[ev.node_id]
            active_set = set(ev.active_slots) if ev.active_slots else set(fim.synapse_slots)
            sources_detail: List[Dict[str, Any]] = []
            for slot, src_id in zip(fim.synapse_slots, fim.sources):
                child_text = (
                    self.net.id_to_char.get(src_id, "?")
                    if ev.level == 1
                    else self.net.decode_fim(ev.level - 1, src_id)
                )
                sources_detail.append(
                    {
                        "slot": slot,
                        "neuron": f"Б{slot + 1}",
                        "child_level": ev.level - 1,
                        "child_id": src_id,
                        "child_text": child_text,
                        "fired": slot in active_set,
                    }
                )
            fim_spikes.append(
                {
                    "level": ev.level,
                    "fim_id": ev.node_id,
                    "text": self.net.decode_fim(ev.level, ev.node_id),
                    "time_ms": ev.time_ms,
                    "event_type": ev.event_type,
                    "confidence": round(float(ev.confidence), 2),
                    "active_slots": list(ev.active_slots) if ev.active_slots else list(fim.synapse_slots),
                    "synapse_slots": list(fim.synapse_slots),
                    "window_ms": level_window_ms(ev.level),
                    "parents_count": len(self.net.parent_fims.get((ev.level, ev.node_id), ())),
                    "recognitions": fim.recognitions,
                    "training_stage": fim.training_stage,
                    "sources_detail": sources_detail,
                }
            )

        fim_spikes.sort(key=lambda item: (item["time_ms"], item["level"], item["fim_id"]))

        elem_by_t = {es["time_ms"]: es for es in element_spikes}
        fim_by_t: Dict[int, List[Dict[str, Any]]] = {}
        for sp in fim_spikes:
            fim_by_t.setdefault(sp["time_ms"], []).append(
                {
                    "level": sp["level"],
                    "fim_id": sp["fim_id"],
                    "text": sp["text"],
                    "confidence": sp["confidence"],
                    "event_type": sp["event_type"],
                }
            )
        all_times = sorted(set(elem_by_t.keys()) | set(fim_by_t.keys()))
        tick_timeline = [
            {
                "time_ms": t,
                "tick_index": (t // TICK_MS) if (t % TICK_MS == 0) else None,
                "instinct_pulse": (t % INSTINCT_CYCLE_MS == 0),
                "char": elem_by_t[t]["char"] if t in elem_by_t else None,
                "node_id": elem_by_t[t]["node_id"] if t in elem_by_t else None,
                "events": fim_by_t.get(t, []),
            }
            for t in all_times
        ]

        max_ev_t = max([s["time_ms"] for s in fim_spikes] + [t_cur, TICK_MS])
        duration_ms = ((max_ev_t + INSTINCT_CYCLE_MS - 1) // INSTINCT_CYCLE_MS) * INSTINCT_CYCLE_MS
        if duration_ms < INSTINCT_CYCLE_MS:
            duration_ms = INSTINCT_CYCLE_MS

        instinct_pulses_ms = list(range(INSTINCT_CYCLE_MS, duration_ms + 1, INSTINCT_CYCLE_MS))

        target_pair = self._choose_target_fim(fim_spikes, target_level, target_fim_id)
        target_meta = self._build_target_meta(*target_pair) if target_pair else None

        voltage_traces = self._simulate_membrane(
            duration_ms=duration_ms,
            instinct_pulses_ms=instinct_pulses_ms,
            element_spikes=element_spikes,
            fim_spikes=fim_spikes,
            target_pair=target_pair,
            target_meta=target_meta,
        )

        highest_recognition: Optional[Dict[str, Any]] = None
        if fim_spikes:
            highest_recognition = max(
                fim_spikes,
                key=lambda s: (s["level"], s["confidence"], s["time_ms"]),
            )

        return {
            "clean_text": clean,
            "dt_ms": self.DT_MS,
            "tick_ms": TICK_MS,
            "instinct_cycle_ms": INSTINCT_CYCLE_MS,
            "duration_ms": duration_ms,
            "instinct_pulses_ms": instinct_pulses_ms,
            "element_spikes": element_spikes,
            "fim_spikes": fim_spikes,
            "tick_timeline": tick_timeline,
            "highest_recognition": highest_recognition,
            "target_fim": target_meta,
            "voltage_traces": voltage_traces,
        }

    def _simulate_membrane(
        self,
        duration_ms: int,
        instinct_pulses_ms: List[int],
        element_spikes: List[Dict[str, Any]],
        fim_spikes: List[Dict[str, Any]],
        target_pair: Optional[Tuple[int, int]],
        target_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        dt = self.DT_MS
        t_grid = list(range(0, duration_ms + 1, dt))
        n_steps = len(t_grid)

        v_b1 = [0.0] * n_steps
        v_b2 = [0.0] * n_steps
        v_b3 = [0.0] * n_steps
        v_head = [0.0] * n_steps
        v_instinct = [0.0] * n_steps

        pulse_set = set(instinct_pulses_ms)
        spike_times_b1: List[int] = []
        spike_times_b2: List[int] = []
        spike_times_b3: List[int] = []
        spike_times_head: List[int] = []

        if target_pair is None or target_meta is None:
            for idx, t in enumerate(t_grid):
                if t in pulse_set:
                    v_instinct[idx] = 1.0
            return {
                "t_grid": t_grid,
                "v_b1": v_b1,
                "v_b2": v_b2,
                "v_b3": v_b3,
                "v_head": v_head,
                "v_instinct": v_instinct,
                "v_threshold": self.V_THRESHOLD,
                "spike_times_b1": spike_times_b1,
                "spike_times_b2": spike_times_b2,
                "spike_times_b3": spike_times_b3,
                "spike_times_head": spike_times_head,
            }

        target_lvl, target_fid = target_pair
        fim = self.net.levels[target_lvl].fims[target_fid]
        win_ms = float(level_window_ms(target_lvl))
        # Медленная утечка внутри окна W_L, чтобы 2-3 входных спайка уверенно накапливались
        decay_base = math.exp(-dt / max(win_ms * 1.4, float(dt)))

        # Сопоставляем входящие спайки уровня target_lvl - 1 по слотам (0, 1, 2)
        src_to_slots: Dict[int, List[int]] = {}
        for slot, src_id in zip(fim.synapse_slots, fim.sources):
            src_to_slots.setdefault(src_id, []).append(slot)

        incoming_by_step: Dict[int, List[int]] = {}
        if target_lvl == 1:
            for sp in element_spikes:
                slots = src_to_slots.get(sp["node_id"])
                if slots:
                    step_t = min(duration_ms, ((sp["time_ms"] + dt // 2) // dt) * dt)
                    incoming_by_step.setdefault(step_t, []).extend(slots)
        else:
            for sp in fim_spikes:
                if sp["level"] == target_lvl - 1:
                    slots = src_to_slots.get(sp["fim_id"])
                    if slots:
                        step_t = min(duration_ms, ((sp["time_ms"] + dt // 2) // dt) * dt)
                        incoming_by_step.setdefault(step_t, []).extend(slots)

        head_fire_steps: set[int] = set()
        for sp in fim_spikes:
            if sp["level"] == target_lvl and sp["fim_id"] == target_fid:
                step_t = min(duration_ms, ((sp["time_ms"] + dt // 2) // dt) * dt)
                head_fire_steps.add(step_t)

        b = [0.0, 0.0, 0.0]
        last_spike_t = [-10**9, -10**9, -10**9]
        cooldown_until = -1

        for idx, t in enumerate(t_grid):
            if t in pulse_set:
                v_instinct[idx] = 1.0
            elif idx > 0:
                v_instinct[idx] = round(v_instinct[idx - 1] * 0.45, 3)

            # Утечка мембран Б1, Б2, Б3 и сброс при выходе за окно W_L
            for s in range(3):
                if t - last_spike_t[s] > win_ms:
                    b[s] *= 0.65
                else:
                    b[s] *= decay_base
                if b[s] < 0.01:
                    b[s] = 0.0

            # Входящие спайки на Б1 / Б2 / Б3
            slots_now = incoming_by_step.get(t, ())
            for slot in slots_now:
                b[slot] = 1.05
                last_spike_t[slot] = t
                if slot == 0:
                    spike_times_b1.append(t)
                elif slot == 1:
                    spike_times_b2.append(t)
                else:
                    spike_times_b3.append(t)

            # Суммарный потенциал головного нейрона Г (LIF интегратор)
            v_sum = b[0] + b[1] + b[2]
            fired_now = False
            if t in head_fire_steps:
                v_sum = max(v_sum, 2.15 if len(fim.sources) == 2 else (3.1 if v_sum >= 2.8 else 2.25))
                fired_now = True
            elif t > cooldown_until and v_sum >= self.V_THRESHOLD:
                # Ассоциативное срабатывание порога, если оба импульса попали в окно в порядке
                fired_now = True

            if fired_now:
                v_head[idx] = round(max(v_sum, 2.2), 3)
                spike_times_head.append(t)
                cooldown_until = t + dt
                # Записываем текущие Б до рефрактерного сброса на следующем шаге
                v_b1[idx] = round(b[0], 3)
                v_b2[idx] = round(b[1], 3)
                v_b3[idx] = round(b[2], 3)
                b = [0.0, 0.0, 0.0]
            else:
                v_b1[idx] = round(b[0], 3)
                v_b2[idx] = round(b[1], 3)
                v_b3[idx] = round(b[2], 3)
                v_head[idx] = round(v_sum, 3)

        return {
            "t_grid": t_grid,
            "v_b1": v_b1,
            "v_b2": v_b2,
            "v_b3": v_b3,
            "v_head": v_head,
            "v_instinct": v_instinct,
            "v_threshold": self.V_THRESHOLD,
            "spike_times_b1": spike_times_b1,
            "spike_times_b2": spike_times_b2,
            "spike_times_b3": spike_times_b3,
            "spike_times_head": spike_times_head,
        }


def _run_cli_verification() -> None:
    from train_fim import train_default_network

    print("=== Загрузка и обучение сети ребёнка (0 предзаданных ФИМов) ===")
    net = train_default_network()
    sim = FIMNengoSimulator(net)

    tests = [
        "привет",
        "мама мыла раму",
        "дом у берега",
        "ну привет",
        "мама мыла",
        "мама раму",
    ]

    print("\n=== Проверка спайкового симулятора FIMNengoSimulator ===")
    for phrase in tests:
        res = sim.run(phrase)
        top = res["highest_recognition"]
        by_lvl: Dict[int, List[str]] = {}
        for sp in res["fim_spikes"]:
            label = f"{sp['text']!r} (conf={sp['confidence']})"
            by_lvl.setdefault(sp["level"], []).append(label)

        print(f"\nВход: {phrase!r} | duration={res['duration_ms']} мс | импульсов инстинкта={len(res['instinct_pulses_ms'])}")
        for lvl in sorted(by_lvl):
            print(f"  Уровень {lvl}: {', '.join(by_lvl[lvl])}")
        if top:
            print(
                f"  Макс. узнавание: L{top['level']} #{top['fim_id']} {top['text']!r} "
                f"(conf={top['confidence']}, t={top['time_ms']} мс)"
            )
        else:
            print("  Макс. узнавание: нет")

        traces = res["voltage_traces"]
        max_head = max(traces["v_head"]) if traces["v_head"] else 0.0
        target_label = repr(res["target_fim"]["text"]) if res["target_fim"] else "None"
        print(
            f"  Осциллограф (target={target_label}): "
            f"шагов={len(traces['t_grid'])}, max V_Г={max_head:.2f}, V_th={traces['v_threshold']}, "
            f"спайков Г={len(traces['spike_times_head'])}"
        )


if __name__ == "__main__":
    _run_cli_verification()
