"""
Иерархическая временная модель памяти на основе ФИМ (9 уровней + уровень 0).

Все ФИМы на уровнях 1..9 изначально ПУСТЫЕ (никаких предзаданных/вшитых слогов).
Система учится с чистого листа:
- Уровень 0: 41 элемент (33 буквы русского алфавита + " .,!?;:-"), такт 84 мс.
- Уровни 1..9: единый класс LevelPool (до 5000 ФИМ на уровень).
- Правила едины для всех уровней:
  - Минимум 2 сигнала снизу, максимум 3 (3 базовых нейрона B1, B2, B3 + головной H).
  - Окна: Уровень 1 = 252 мс, Уровень L (L >= 2) = 252 * 2^L мс.
  - На уровне 2 действует физический предел <= 6 символов на один ФИМ.
  - И при консолидации нового ФИМа, и при узнавании известного головной нейрон
    сразу посылает сигнал наверх.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


TICK_MS = 84
INSTINCT_CYCLE_MS = 252  # 3 * 84 мс
LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
PUNCTUATION = " .,!?;:-"
RUSSIAN_ALPHABET = LETTERS + PUNCTUATION  # 41 элемент уровня 0


def normalize_text(text: str) -> str:
    """Приводит текст к нижнему регистру, оставляя элементы 0-го уровня и схлопывая пробелы."""
    filtered = [ch for ch in text.lower() if ch in RUSSIAN_ALPHABET]
    out: List[str] = []
    for ch in filtered:
        if ch == " " and out and out[-1] == " ":
            continue
        out.append(ch)
    return "".join(out)


def level_window_ms(level: int) -> int:
    """Возвращает окно прослушивания (мс) для уровня 1..9."""
    if level <= 1:
        return INSTINCT_CYCLE_MS
    return INSTINCT_CYCLE_MS * (2 ** level)


@dataclass
class FIM:
    """Функциональный информационный модуль: 3 базовых нейрона + 1 головной."""

    fim_id: int
    level: int
    synapse_slots: Tuple[int, ...]  # (0, 2) для 2-синапсных или (0, 1, 2) для 3-синапсных
    sources: Tuple[int, ...] = ()
    consolidated: bool = False
    recognitions: int = 0
    created_at_ms: int = 0
    training_context: str = ""
    training_stage: str = ""


@dataclass
class SignalEvent:
    """Сигнал от уровня level в момент time_ms."""

    level: int
    node_id: int
    time_ms: int
    event_type: str  # "element" | "consolidation" | "recognition"
    confidence: float = 1.0
    active_slots: Tuple[int, ...] = ()


class LevelPool:
    """
    Единый пул ФИМ для любого уровня (1..9).
    Изначально полностью пустой: заполняется только в процессе обучения.
    """

    def __init__(
        self,
        level: int,
        get_decoded_text: Callable[[int, int], str],
        max_fims: int = 5000,
    ) -> None:
        self.level = level
        self.max_fims = max_fims
        self.window_ms = level_window_ms(level)
        self.get_decoded_text = get_decoded_text
        self.fims: List[FIM] = []
        self.pattern_index: Dict[Tuple[int, ...], int] = {}
        self.buffer: List[Tuple[int, int]] = []  # максимум 3 сигнала (time_ms, source_id)
        self.window_start_ms: Optional[int] = None

    def _emit_seq(
        self,
        seq: Tuple[int, ...],
        time_ms: int,
        context_str: str = "",
        stage_str: str = "",
    ) -> Optional[SignalEvent]:
        if seq in self.pattern_index:
            fim_id = self.pattern_index[seq]
            self.fims[fim_id].recognitions += 1
            return SignalEvent(
                level=self.level,
                node_id=fim_id,
                time_ms=time_ms,
                event_type="recognition",
            )

        if len(self.fims) >= self.max_fims:
            return None

        fim_id = len(self.fims)
        slots = (0, 2) if len(seq) == 2 else (0, 1, 2)
        fim = FIM(
            fim_id=fim_id,
            level=self.level,
            synapse_slots=slots,
            sources=seq,
            consolidated=True,
            created_at_ms=time_ms,
            training_context=context_str,
            training_stage=stage_str,
        )
        self.fims.append(fim)
        self.pattern_index[seq] = fim_id
        return SignalEvent(
            level=self.level,
            node_id=fim_id,
            time_ms=time_ms,
            event_type="consolidation",
        )

    def receive(
        self,
        source_id: int,
        time_ms: int,
        context_str: str = "",
        stage_str: str = "",
    ) -> List[SignalEvent]:
        events: List[SignalEvent] = []
        if self.window_start_ms is None:
            self.window_start_ms = time_ms
        elif self.buffer and (time_ms - self.buffer[-1][0] > self.window_ms):
            last_t = self.buffer[-1][0]
            events.extend(
                self.flush(
                    last_t + self.window_ms,
                    context_str=context_str,
                    stage_str=stage_str,
                )
            )
            self.window_start_ms = time_ms

        self.buffer.append((time_ms, source_id))
        if len(self.buffer) < 3:
            return events

        seq3 = (self.buffer[0][1], self.buffer[1][1], self.buffer[2][1])
        seq2 = (self.buffer[0][1], self.buffer[1][1])

        # Случай 1: 3-сигнальный паттерн уже выучен -> узнавание
        if seq3 in self.pattern_index:
            ev = self._emit_seq(seq3, self.buffer[2][0], context_str=context_str, stage_str=stage_str)
            self.buffer.clear()
            self.window_start_ms = None
            if ev is not None:
                events.append(ev)
            return events

        # Случай 2: 2-сигнальный паттерн уже выучен -> узнавание, 3-й остаётся в буфере
        if seq2 in self.pattern_index:
            ev = self._emit_seq(seq2, self.buffer[1][0], context_str=context_str, stage_str=stage_str)
            self.buffer = [self.buffer[2]]
            self.window_start_ms = self.buffer[0][0]
            if ev is not None:
                events.append(ev)
            return events

        # Случай 3: новая консолидация
        if self.level == 2:
            len3 = sum(len(self.get_decoded_text(1, s)) for s in seq3)
            if len3 > 6:
                ev = self._emit_seq(seq2, self.buffer[1][0], context_str=context_str, stage_str=stage_str)
                self.buffer = [self.buffer[2]]
                self.window_start_ms = self.buffer[0][0]
                if ev is not None:
                    events.append(ev)
                return events

        ev = self._emit_seq(seq3, self.buffer[2][0], context_str=context_str, stage_str=stage_str)
        self.buffer.clear()
        self.window_start_ms = None
        if ev is not None:
            events.append(ev)
        return events

    def flush(
        self,
        time_ms: int,
        context_str: str = "",
        stage_str: str = "",
    ) -> List[SignalEvent]:
        if len(self.buffer) < 2:
            self.buffer.clear()
            self.window_start_ms = None
            return []

        seq = tuple(nid for _, nid in self.buffer[:3])
        self.buffer.clear()
        self.window_start_ms = None
        ev = self._emit_seq(seq, time_ms, context_str=context_str, stage_str=stage_str)
        return [ev] if ev is not None else []


class InstinctNetwork:
    """9-уровневая сеть ФИМ с уровнем 0 (41 элемент), пустыми пулами ФИМ и аналитикой."""

    def __init__(self, num_levels: int = 9, max_fims_per_level: int = 5000) -> None:
        self.num_levels = num_levels
        self.char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(RUSSIAN_ALPHABET)}
        self.id_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(RUSSIAN_ALPHABET)}
        self.levels: Dict[int, LevelPool] = {
            lvl: LevelPool(lvl, self.decode_fim, max_fims=max_fims_per_level)
            for lvl in range(1, num_levels + 1)
        }
        self.current_time_ms = 0
        self.events_log: List[SignalEvent] = []
        self.parent_fims: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        # Параллельный синаптический индекс: level -> src_id -> [(fim_id, slot_index), ...]
        self.incoming_synapses: Dict[int, Dict[int, List[Tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def _register_parent_links(self, level: int, fim_id: int) -> None:
        fim = self.levels[level].fims[fim_id]
        for slot_idx, src_id in enumerate(fim.sources):
            key = (level - 1, src_id)
            target = (level, fim_id)
            if target not in self.parent_fims[key]:
                self.parent_fims[key].append(target)
            entry = (fim_id, slot_idx)
            if entry not in self.incoming_synapses[level][src_id]:
                self.incoming_synapses[level][src_id].append(entry)

    def _propagate_up(
        self,
        ev: SignalEvent,
        out: List[SignalEvent],
        context_str: str,
        stage_str: str,
    ) -> None:
        if ev.event_type == "consolidation":
            self._register_parent_links(ev.level, ev.node_id)
        out.append(ev)
        next_lvl = ev.level + 1
        if next_lvl > self.num_levels:
            return
        pool = self.levels[next_lvl]
        for up_ev in pool.receive(
            ev.node_id,
            ev.time_ms,
            context_str=context_str,
            stage_str=stage_str,
        ):
            self._propagate_up(up_ev, out, context_str, stage_str)

    def flush_all(self, context_str: str = "", stage_str: str = "Поток") -> List[SignalEvent]:
        out: List[SignalEvent] = []
        for lvl in range(1, self.num_levels + 1):
            for ev in self.levels[lvl].flush(
                self.current_time_ms,
                context_str=context_str,
                stage_str=stage_str,
            ):
                self._propagate_up(ev, out, context_str, stage_str)
        self.events_log.extend(out)
        return out

    def feed_stream(
        self,
        text: str,
        flush_end: bool = True,
        stage: str = "Поток",
    ) -> List[SignalEvent]:
        """Подаёт текст с пробелами и пунктуацией (каждые 84 мс по одному элементу)."""
        out: List[SignalEvent] = []
        clean = normalize_text(text)
        lvl1 = self.levels[1]
        for idx, ch in enumerate(clean):
            self.current_time_ms += TICK_MS
            ctx_start = max(0, idx - 55)
            ctx = clean[ctx_start : idx + 1]
            for ev in lvl1.receive(
                self.char_to_id[ch],
                self.current_time_ms,
                context_str=ctx,
                stage_str=stage,
            ):
                self._propagate_up(ev, out, ctx, stage)

        if flush_end:
            tail_ctx = clean[-60:] if clean else ""
            out.extend(self.flush_all(context_str=tail_ctx, stage_str=stage))
        else:
            self.events_log.extend(out)
        return out

    def probe_stream(self, text: str) -> List[SignalEvent]:
        """Прогоняет текст без изменения памяти (read-only) с полным и 2-из-3 спайковым узнаванием."""
        clean = normalize_text(text)
        if not clean:
            return []

        out: List[SignalEvent] = []
        buffers: Dict[int, List[Tuple[int, int]]] = {lvl: [] for lvl in range(1, self.num_levels + 1)}
        window_starts: Dict[int, Optional[int]] = {lvl: None for lvl in range(1, self.num_levels + 1)}
        unmatched: Dict[int, List[Tuple[int, int]]] = {lvl: [] for lvl in range(1, self.num_levels + 1)}

        def lookup_consolidated(lvl: int, seq: Tuple[int, ...]) -> Optional[int]:
            pool = self.levels[lvl]
            if seq in pool.pattern_index:
                fid = pool.pattern_index[seq]
                if pool.fims[fid].consolidated:
                    return fid
            return None

        def run_associative_2of3(lvl: int) -> None:
            """Ассоциативное спайковое узнавание 2-из-3 для уровней >= 2 (V >= V_th = 2.0)."""
            if lvl < 2 or lvl > self.num_levels:
                return
            pool = self.levels[lvl]
            spikes = unmatched[lvl]
            if len(spikes) < 2:
                return

            used_indices: set[int] = set()
            # Группируем спайки по кандидатным 3-синаптическим ФИМам через incoming_synapses
            while True:
                hits_by_fim: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
                for idx, (t_sp, src_id) in enumerate(spikes):
                    if idx in used_indices:
                        continue
                    for fid, slot_idx in self.incoming_synapses[lvl].get(src_id, ()):
                        fim = pool.fims[fid] if 0 <= fid < len(pool.fims) else None
                        if fim is None or not fim.consolidated or len(fim.sources) != 3:
                            continue
                        hits_by_fim[fid].append((idx, slot_idx, t_sp))

                best_match: Optional[Tuple[int, int, int, int, int, int, int]] = None
                # (score, -t_end, fid, i1, i2, slot1, slot2, t_end)
                for fid, hits in hits_by_fim.items():
                    if len(hits) < 2:
                        continue
                    fim = pool.fims[fid]
                    for a_pos in range(len(hits)):
                        i1, s1, t1 = hits[a_pos]
                        for b_pos in range(a_pos + 1, len(hits)):
                            i2, s2, t2 = hits[b_pos]
                            if s1 >= s2:
                                continue
                            if (t2 - t1) > pool.window_ms:
                                continue
                            # Выбираем более ранний / чаще узнаваемый ФИМ
                            score = (fim.recognitions, -t2, -fid)
                            cand = (score[0], score[1], score[2], fid, i1, i2, s1, s2, t2)
                            if best_match is None or cand[:3] > best_match[:3]:
                                best_match = cand

                if best_match is None:
                    break

                _, _, _, fid, i1, i2, s1, s2, t_ev = best_match
                used_indices.add(i1)
                used_indices.add(i2)
                fim = pool.fims[fid]
                active = (fim.synapse_slots[s1], fim.synapse_slots[s2])
                out.append(
                    SignalEvent(
                        level=lvl,
                        node_id=fid,
                        time_ms=t_ev,
                        event_type="recognition",
                        confidence=0.67,
                        active_slots=active,
                    )
                )
                probe_level(lvl + 1, fid, t_ev)

            if used_indices:
                unmatched[lvl] = [item for idx, item in enumerate(spikes) if idx not in used_indices]

        def probe_level(lvl: int, src_id: int, time_ms: int) -> None:
            if lvl > self.num_levels:
                return
            pool = self.levels[lvl]
            buf = buffers[lvl]
            if window_starts[lvl] is None:
                window_starts[lvl] = time_ms
            elif buf and (time_ms - buf[-1][0] > pool.window_ms):
                last_t = buf[-1][0]
                flush_level(lvl, last_t + pool.window_ms)
                window_starts[lvl] = time_ms

            buf.append((time_ms, src_id))
            if len(buf) < 3:
                return

            seq3 = (buf[0][1], buf[1][1], buf[2][1])
            seq2 = (buf[0][1], buf[1][1])

            fid3 = lookup_consolidated(lvl, seq3)
            if fid3 is not None:
                t_ev = buf[2][0]
                slots = pool.fims[fid3].synapse_slots
                buf.clear()
                window_starts[lvl] = None
                out.append(
                    SignalEvent(
                        level=lvl,
                        node_id=fid3,
                        time_ms=t_ev,
                        event_type="recognition",
                        confidence=1.0,
                        active_slots=slots,
                    )
                )
                probe_level(lvl + 1, fid3, t_ev)
                return

            fid2 = lookup_consolidated(lvl, seq2)
            if fid2 is not None:
                t_ev = buf[1][0]
                slots = pool.fims[fid2].synapse_slots
                del buf[:2]
                window_starts[lvl] = buf[0][0]
                out.append(
                    SignalEvent(
                        level=lvl,
                        node_id=fid2,
                        time_ms=t_ev,
                        event_type="recognition",
                        confidence=1.0,
                        active_slots=slots,
                    )
                )
                probe_level(lvl + 1, fid2, t_ev)
                return

            # Если оба не узнаны — сдвигаем окно, сохраняя выбывший спайк для 2-из-3 ассоциации
            dropped = buf.pop(0)
            if lvl >= 2:
                unmatched[lvl].append(dropped)
            window_starts[lvl] = buf[0][0]

        def flush_level(lvl: int, time_ms: int) -> None:
            if lvl > self.num_levels:
                return
            pool = self.levels[lvl]
            buf = buffers[lvl]
            if len(buf) >= 2:
                seq = tuple(nid for _, nid in buf[:3])
                fid = lookup_consolidated(lvl, seq)
                if fid is not None:
                    slots = pool.fims[fid].synapse_slots
                    buf.clear()
                    window_starts[lvl] = None
                    out.append(
                        SignalEvent(
                            level=lvl,
                            node_id=fid,
                            time_ms=time_ms,
                            event_type="recognition",
                            confidence=1.0,
                            active_slots=slots,
                        )
                    )
                    probe_level(lvl + 1, fid, time_ms)
                    run_associative_2of3(lvl)
                    return

            if buf:
                if lvl >= 2:
                    unmatched[lvl].extend(buf)
                buf.clear()
            window_starts[lvl] = None
            run_associative_2of3(lvl)

        t_cur = 0
        for ch in clean:
            t_cur += TICK_MS
            probe_level(1, self.char_to_id[ch], t_cur)

        for lvl in range(1, self.num_levels + 1):
            flush_level(lvl, t_cur)

        return out

    def recall_highest_from(self, level: int, fim_id: int) -> Tuple[int, int, str]:
        """Поднимается по исходящим синапсам до самого высокого консолидированного ФИМа."""
        cur_lvl, cur_fid = level, fim_id
        while cur_lvl < self.num_levels:
            candidates = [
                (p_lvl, p_fid)
                for (p_lvl, p_fid) in self.parent_fims.get((cur_lvl, cur_fid), [])
                if self.levels[p_lvl].fims[p_fid].consolidated
            ]
            if not candidates:
                break

            def score(item: Tuple[int, int]) -> Tuple[int, int, int]:
                p_lvl, p_fid = item
                pfim = self.levels[p_lvl].fims[p_fid]
                starts_at_b1 = 1 if (pfim.sources and pfim.sources[0] == cur_fid) else 0
                return (starts_at_b1, pfim.recognitions, -p_fid)

            cur_lvl, cur_fid = max(candidates, key=score)

        return cur_lvl, cur_fid, self.decode_fim(cur_lvl, cur_fid)

    def explain_fim(self, level: int, fim_id: int, max_depth: int = 4) -> Dict[str, Any]:
        """Возвращает подробную аналитику ФИМа: этап и контекст обучения, родителей и дерево дочерних ФИМ."""
        if level == 0:
            return {
                "level": 0,
                "fim_id": fim_id,
                "text": self.id_to_char.get(fim_id, "?"),
                "synapse_slots": [],
                "recognitions": 0,
                "created_at_ms": 0,
                "training_context": "",
                "training_stage": "Уровень 0 (элемент)",
                "parents": [],
                "children": [],
            }

        fim = self.levels[level].fims[fim_id]
        parents = [
            {
                "level": p_lvl,
                "fim_id": p_fid,
                "text": self.decode_fim(p_lvl, p_fid),
                "recognitions": self.levels[p_lvl].fims[p_fid].recognitions,
            }
            for (p_lvl, p_fid) in self.parent_fims.get((level, fim_id), [])
            if self.levels[p_lvl].fims[p_fid].consolidated
        ]

        children: List[Dict[str, Any]] = []
        if max_depth > 0:
            slot_names = ["Б1", "Б3"] if fim.synapse_slots == (0, 2) else ["Б1", "Б2", "Б3"]
            for idx, src_id in enumerate(fim.sources):
                slot_label = slot_names[idx] if idx < len(slot_names) else f"Б{idx + 1}"
                child_info = self.explain_fim(level - 1, src_id, max_depth=max_depth - 1)
                child_info["slot"] = slot_label
                children.append(child_info)

        return {
            "level": level,
            "fim_id": fim.fim_id,
            "text": self.decode_fim(level, fim.fim_id),
            "synapse_slots": list(fim.synapse_slots),
            "recognitions": fim.recognitions,
            "created_at_ms": fim.created_at_ms,
            "training_context": fim.training_context,
            "training_stage": fim.training_stage,
            "parents": parents[:20],
            "children": children,
        }

    def decode_fim(self, level: int, fim_id: int) -> str:
        """Разворачивает ФИМ любого уровня обратно в читаемый текст с пробелами и пунктуацией."""
        if level == 0:
            return self.id_to_char.get(fim_id, "?")
        fim = self.levels[level].fims[fim_id]
        return "".join(self.decode_fim(level - 1, src) for src in fim.sources)

    def summary(self) -> Dict[int, Dict[str, int]]:
        stats: Dict[int, Dict[str, int]] = {}
        for lvl, pool in self.levels.items():
            cons_fims = [f for f in pool.fims if f.consolidated]
            stats[lvl] = {
                "window_ms": pool.window_ms,
                "consolidated_fims": len(cons_fims),
                "total_recognitions": sum(f.recognitions for f in cons_fims),
            }
        return stats


def run_demo() -> None:
    net = InstinctNetwork(num_levels=9, max_fims_per_level=5000)
    print("Пустых ФИМов при создании:", {l: len(p.fims) for l, p in net.levels.items()})

    # Учим как ребёнка: сначала слоги, потом слова, потом фразы
    for syl in ["при", "вет", "дом", " у ", "ма ", "ла ", "ма", "мы", "ра", "му", "бе", "ре", "га"]:
        net.feed_stream(syl, flush_end=True, stage="Этап 1: Слоги")
    for w in ["привет", "мама ", "мыла ", "раму", "дом у ", "берега"]:
        net.feed_stream(w, flush_end=True, stage="Этап 2: Слова")
    for p in ["мама мыла раму", "дом у берега"]:
        evs = net.feed_stream(p, flush_end=True, stage="Этап 3: Фразы")
        l1 = [net.decode_fim(1, e.node_id) for e in evs if e.level == 1]
        l2 = [net.decode_fim(2, e.node_id) for e in evs if e.level == 2]
        l3 = [net.decode_fim(3, e.node_id) for e in evs if e.level == 3]
        print(f"Фраза {p!r:16} -> Ур.1: {l1} | Ур.2: {l2} | Ур.3: {l3}")


if __name__ == "__main__":
    run_demo()
