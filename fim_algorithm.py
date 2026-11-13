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
    """Функциональный информационный модуль: 3 базовых нейрона (Б1, Б2, Б3) + 1 головной (Г)."""

    fim_id: int
    level: int
    synapse_slots: Tuple[int, ...]  # (0, 2) для 2-синапсных или (0, 1, 2) для 3-синапсных
    sources: Tuple[int, ...] = ()
    consolidated: bool = False
    recognitions: int = 0
    created_at_ms: int = 0
    training_context: str = ""
    training_stage: str = ""
    decoded_text: str = ""
    synapse_counts: Tuple[int, ...] = ()  # Кол-во синапсов на каждый вход Б1..Б3 (пусто = по 1 шт.)
    outgoing_synapses_per_fim: Optional[int] = None  # Кол-во синапсов от Г этого ФИМ вверх на БН L+1
    custom_window_ms: Optional[int] = None

    def get_counts(self) -> Tuple[int, ...]:
        n = len(self.sources)
        if len(self.synapse_counts) == n:
            return self.synapse_counts
        return (1,) * n

    def effective_window_ms(self, default_win: int) -> int:
        return int(self.custom_window_ms) if (self.custom_window_ms and self.custom_window_ms > 0) else int(default_win)


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
        max_fims: int = 1_000_000,
    ) -> None:
        self.level = level
        self.max_fims = max_fims
        self.window_ms = level_window_ms(level)
        self.min_synapses = 2
        self.max_synapses = 3
        self.default_synapse_count = 1
        self.synapses_per_fim_out = 500  # Квота исходящих синапсов от 1 ФИМ (Г) на БН уровня L+1
        self.synapses_to_b1 = 1
        self.synapses_to_b2 = 1
        self.synapses_to_b3 = 1
        self.level2_max_chars = 6
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
        decoded = "".join(self.get_decoded_text(self.level - 1, s) for s in seq)
        if self.synapses_to_b1 != 1 or self.synapses_to_b2 != 1 or self.synapses_to_b3 != 1:
            counts = (self.synapses_to_b1, self.synapses_to_b3) if len(seq) == 2 else (
                self.synapses_to_b1, self.synapses_to_b2, self.synapses_to_b3
            )
        elif self.default_synapse_count != 1:
            counts = (self.default_synapse_count,) * len(seq)
        else:
            counts = ()
        fim = FIM(
            fim_id=fim_id,
            level=self.level,
            synapse_slots=slots,
            sources=seq,
            consolidated=True,
            created_at_ms=time_ms,
            training_context=context_str,
            training_stage=stage_str,
            decoded_text=decoded,
            synapse_counts=counts,
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
        max_syn = 2 if self.max_synapses <= 2 else 3
        if max_syn == 2:
            if len(self.buffer) < 2:
                return events
            seq2 = (self.buffer[0][1], self.buffer[1][1])
            span2 = self.buffer[1][0] - self.buffer[0][0]
            too_wide2 = (span2 > self.window_ms) or (
                self.level == 2 and sum(len(self.get_decoded_text(1, s)) for s in seq2) > self.level2_max_chars
            )
            if too_wide2 and seq2 not in self.pattern_index:
                self.buffer = [self.buffer[1]]
                self.window_start_ms = self.buffer[0][0]
                return events
            ev = self._emit_seq(seq2, self.buffer[1][0], context_str=context_str, stage_str=stage_str)
            self.buffer.clear()
            self.window_start_ms = None
            if ev is not None:
                events.append(ev)
            return events

        if len(self.buffer) < 3:
            return events

        seq3 = (self.buffer[0][1], self.buffer[1][1], self.buffer[2][1])
        seq2 = (self.buffer[0][1], self.buffer[1][1])
        span3 = self.buffer[2][0] - self.buffer[0][0]

        # Случай 1: 3-сигнальный паттерн уже выучен и умещается в окно уровня W_L -> узнавание
        if span3 <= self.window_ms and seq3 in self.pattern_index:
            ev = self._emit_seq(seq3, self.buffer[2][0], context_str=context_str, stage_str=stage_str)
            self.buffer.clear()
            self.window_start_ms = None
            if ev is not None:
                events.append(ev)
            return events

        # Случай 2: 2-сигнальный паттерн уже выучен -> узнавание, 3-й остаётся в буфере
        if self.min_synapses <= 2 and seq2 in self.pattern_index:
            ev = self._emit_seq(seq2, self.buffer[1][0], context_str=context_str, stage_str=stage_str)
            self.buffer = [self.buffer[2]]
            self.window_start_ms = self.buffer[0][0]
            if ev is not None:
                events.append(ev)
            return events

        # Случай 3: новая консолидация (соблюдаем физическое окно W_L и лимит символов на Ур. 2)
        too_wide = (span3 > self.window_ms) or (
            self.level == 2 and sum(len(self.get_decoded_text(1, s)) for s in seq3) > self.level2_max_chars
        )
        if too_wide:
            if self.min_synapses <= 2:
                ev = self._emit_seq(seq2, self.buffer[1][0], context_str=context_str, stage_str=stage_str)
                if ev is not None:
                    events.append(ev)
            self.buffer = [self.buffer[2]]
            self.window_start_ms = self.buffer[0][0]
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
        min_req = max(2, self.min_synapses)
        if len(self.buffer) < min_req:
            self.buffer.clear()
            self.window_start_ms = None
            return []

        max_take = 2 if self.max_synapses <= 2 else 3
        seq = tuple(nid for _, nid in self.buffer[:max_take])
        self.buffer.clear()
        self.window_start_ms = None
        if self.level == 2 and sum(len(self.get_decoded_text(1, s)) for s in seq) > self.level2_max_chars:
            if len(seq) == 3 and min_req <= 2:
                seq = seq[:2]
            else:
                return []
        ev = self._emit_seq(seq, time_ms, context_str=context_str, stage_str=stage_str)
        return [ev] if ev is not None else []


class InstinctNetwork:
    """9-уровневая сеть ФИМ с уровнем 0 (41 элемент), пулом 1 млн/уровень и аналитикой."""

    def __init__(self, num_levels: int = 9, max_fims_per_level: int = 1_000_000) -> None:
        self.num_levels = num_levels
        self.tick_ms: int = TICK_MS  # Такт подачи элементов L0 и длительность импульса БН (84 мс)
        self.instinct_cycle_ms: int = INSTINCT_CYCLE_MS  # Цикл нейрона инстинкта (252 мс = 3 * 84 мс)
        self.allow_associative_2of3: bool = True  # Узнавание 3-синапсного ФИМ по 2 из 3 сигналов в порядке
        self.level2_max_chars: int = 6  # Физический предел Уровня 2 <= 6 символов
        self.synapses_per_element_l0: int = 1000  # Кол-во синапсов от 1 элемента L0 на Б1..Б3 Ур. 1
        self.element_out_quotas: Dict[int, int] = {}  # Индивидуальная квота синапсов для элементов L0

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
        if len(self.events_log) > 2000:
            del self.events_log[:-1000]
        return out

    def feed_stream(
        self,
        text: str,
        flush_end: bool = True,
        stage: str = "Поток",
    ) -> List[SignalEvent]:
        """Подаёт текст с пробелами и пунктуацией (каждые tick_ms мс по одному элементу)."""
        out: List[SignalEvent] = []
        clean = normalize_text(text)
        lvl1 = self.levels[1]
        for idx, ch in enumerate(clean):
            self.current_time_ms += self.tick_ms
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
        if len(self.events_log) > 2000:
            del self.events_log[:-1000]
        return out

    def probe_stream(self, text: str) -> List[SignalEvent]:
        """Прогоняет текст без изменения памяти (read-only) с полным (3/3, 2/2) и 2-из-3 узнаванием."""
        clean = normalize_text(text)
        if not clean:
            return []

        out: List[SignalEvent] = []
        buffers: Dict[int, List[Tuple[int, int]]] = {lvl: [] for lvl in range(1, self.num_levels + 1)}
        window_starts: Dict[int, Optional[int]] = {lvl: None for lvl in range(1, self.num_levels + 1)}
        unmatched: Dict[int, List[Tuple[int, int]]] = {lvl: [] for lvl in range(1, self.num_levels + 1)}

        def lookup_consolidated(lvl: int, seq: Tuple[int, ...], span_ms: int = 0) -> Optional[int]:
            pool = self.levels[lvl]
            if seq in pool.pattern_index:
                fid = pool.pattern_index[seq]
                fim = pool.fims[fid]
                if fim.consolidated and span_ms <= fim.effective_window_ms(pool.window_ms):
                    return fid
            return None

        def run_associative_2of3(lvl: int) -> None:
            """Узнавание 3-синапсного ФИМ по 2 из 3 сигналов снизу внутри окна W_L (уровни >= 2)."""
            if not self.allow_associative_2of3 or lvl < 2 or lvl > self.num_levels:
                return
            pool = self.levels[lvl]
            spikes = unmatched[lvl]
            if len(spikes) < 2:
                return

            used_indices: set[int] = set()
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

                best_match: Optional[Tuple[int, int, int, int, int, int, int, int, int]] = None
                for fid, hits in hits_by_fim.items():
                    if len(hits) < 2:
                        continue
                    fim = pool.fims[fid]
                    eff_win = fim.effective_window_ms(pool.window_ms)
                    for a_pos in range(len(hits)):
                        i1, s1, t1 = hits[a_pos]
                        for b_pos in range(a_pos + 1, len(hits)):
                            i2, s2, t2 = hits[b_pos]
                            if s1 >= s2:
                                continue
                            if (t2 - t1) > eff_win:
                                continue
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
            span3 = buf[2][0] - buf[0][0]
            span2 = buf[1][0] - buf[0][0]

            fid3 = lookup_consolidated(lvl, seq3, span3)
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

            fid2 = lookup_consolidated(lvl, seq2, span2)
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
                span = buf[len(seq) - 1][0] - buf[0][0]
                fid = lookup_consolidated(lvl, seq, span)
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
            t_cur += self.tick_ms
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

    def _parent_links_with_slots(self, level: int, node_id: int) -> List[Dict[str, Any]]:
        """Возвращает родительские ФИМы на уровне level+1 с точным указанием слота Б1/Б2/Б3 и кол-ва синапсов."""
        if level >= self.num_levels:
            return []
        next_pool = self.levels[level + 1]
        out: List[Dict[str, Any]] = []
        for (p_lvl, p_fid) in self.parent_fims.get((level, node_id), []):
            if p_lvl != level + 1:
                continue
            pfim = next_pool.fims[p_fid]
            if not pfim.consolidated:
                continue
            counts = pfim.get_counts()
            for idx, src in enumerate(pfim.sources):
                if src == node_id:
                    slot_num = pfim.synapse_slots[idx] if idx < len(pfim.synapse_slots) else idx
                    out.append(
                        {
                            "level": p_lvl,
                            "fim_id": p_fid,
                            "text": self.decode_fim(p_lvl, p_fid),
                            "recognitions": pfim.recognitions,
                            "target_slot": f"Б{slot_num + 1}",
                            "synapse_count": counts[idx] if idx < len(counts) else 1,
                        }
                    )
        return out

    def explain_fim(self, level: int, fim_id: int, max_depth: int = 4) -> Dict[str, Any]:
        """Возвращает аналитику ФИМа или элемента L0 строго по ТЗ: синапсы Б1..Б3, Г, окно и связи."""
        if level == 0:
            parents = self._parent_links_with_slots(0, fim_id)
            actual_out = sum(p["synapse_count"] for p in parents)
            b1_out = sum(p["synapse_count"] for p in parents if p["target_slot"] == "Б1")
            b2_out = sum(p["synapse_count"] for p in parents if p["target_slot"] == "Б2")
            b3_out = sum(p["synapse_count"] for p in parents if p["target_slot"] == "Б3")
            quota = self.element_out_quotas.get(fim_id, self.synapses_per_element_l0)
            return {
                "level": 0,
                "fim_id": fim_id,
                "text": self.id_to_char.get(fim_id, "?"),
                "synapse_slots": [],
                "synapse_counts": [],
                "window_ms": self.tick_ms,
                "custom_window_ms": None,
                "consolidated": True,
                "sources": [],
                "recognitions": 0,
                "created_at_ms": 0,
                "training_context": "",
                "training_stage": "Уровень 0 (элемент)",
                "outgoing_synapses_per_fim": quota,
                "actual_out_synapses": actual_out,
                "out_by_slot": {"Б1": b1_out, "Б2": b2_out, "Б3": b3_out},
                "parents": parents[:40],
                "children": [],
            }

        fim = self.levels[level].fims[fim_id]
        pool = self.levels[level]
        counts = fim.get_counts()
        parents = self._parent_links_with_slots(level, fim_id)
        actual_out = sum(p["synapse_count"] for p in parents)
        b1_out = sum(p["synapse_count"] for p in parents if p["target_slot"] == "Б1")
        b2_out = sum(p["synapse_count"] for p in parents if p["target_slot"] == "Б2")
        b3_out = sum(p["synapse_count"] for p in parents if p["target_slot"] == "Б3")

        children: List[Dict[str, Any]] = []
        if max_depth > 0:
            slot_names = ["Б1", "Б3"] if fim.synapse_slots == (0, 2) else ["Б1", "Б2", "Б3"]
            for idx, src_id in enumerate(fim.sources):
                slot_label = slot_names[idx] if idx < len(slot_names) else f"Б{idx + 1}"
                child_info = self.explain_fim(level - 1, src_id, max_depth=max_depth - 1)
                child_info["slot"] = slot_label
                child_info["slot_index"] = fim.synapse_slots[idx] if idx < len(fim.synapse_slots) else idx
                child_info["synapse_count"] = counts[idx] if idx < len(counts) else 1
                children.append(child_info)

        return {
            "level": level,
            "fim_id": fim.fim_id,
            "text": self.decode_fim(level, fim.fim_id),
            "synapse_slots": list(fim.synapse_slots),
            "synapse_counts": list(counts),
            "window_ms": fim.effective_window_ms(pool.window_ms),
            "custom_window_ms": fim.custom_window_ms,
            "consolidated": fim.consolidated,
            "sources": list(fim.sources),
            "recognitions": fim.recognitions,
            "created_at_ms": fim.created_at_ms,
            "training_context": fim.training_context,
            "training_stage": fim.training_stage,
            "outgoing_synapses_per_fim": (
                fim.outgoing_synapses_per_fim
                if fim.outgoing_synapses_per_fim is not None
                else pool.synapses_per_fim_out
            ),
            "actual_out_synapses": actual_out,
            "out_by_slot": {"Б1": b1_out, "Б2": b2_out, "Б3": b3_out},
            "parents": parents[:30],
            "children": children,
        }

    def _clear_decoded_from(self, start_level: int) -> None:
        for lvl in range(max(1, start_level), self.num_levels + 1):
            for f in self.levels[lvl].fims:
                f.decoded_text = ""

    def _rebuild_level_indexes(self, level: int) -> None:
        """Перестраивает pattern_index, incoming_synapses и parent_fims для уровня level."""
        if level < 1 or level > self.num_levels:
            return
        pool = self.levels[level]
        pool.pattern_index.clear()
        self.incoming_synapses[level].clear()
        for key in list(self.parent_fims.keys()):
            lst = [item for item in self.parent_fims[key] if item[0] != level]
            if lst:
                self.parent_fims[key] = lst
            else:
                del self.parent_fims[key]

        for f in pool.fims:
            if not f.consolidated:
                continue
            if f.sources and f.sources not in pool.pattern_index:
                pool.pattern_index[f.sources] = f.fim_id
            for slot_idx, src_id in enumerate(f.sources):
                key = (level - 1, src_id)
                target = (level, f.fim_id)
                if target not in self.parent_fims[key]:
                    self.parent_fims[key].append(target)
                entry = (f.fim_id, slot_idx)
                if entry not in self.incoming_synapses[level][src_id]:
                    self.incoming_synapses[level][src_id].append(entry)

        self._clear_decoded_from(level)

    def _resolve_source_spec(self, child_level: int, spec: Any) -> int:
        """Преобразует ID (int / '#12') или текст дочернего уровня в node_id на уровне child_level."""
        if isinstance(spec, int):
            if child_level == 0:
                if spec in self.id_to_char:
                    return spec
                raise ValueError(f"Неизвестный элемент L0 #{spec}")
            pool = self.levels[child_level]
            if 0 <= spec < len(pool.fims):
                return spec
            raise ValueError(f"Не найден ФИМ L{child_level} #{spec}")

        s = str(spec)
        s_strip = s.strip()
        if s_strip.startswith("#") and s_strip[1:].isdigit():
            return self._resolve_source_spec(child_level, int(s_strip[1:]))
        if s_strip.isdigit() and (child_level > 0 or int(s_strip) >= 41):
            return self._resolve_source_spec(child_level, int(s_strip))

        if child_level == 0:
            ch = normalize_text(s)[:1]
            if not ch:
                ch = " "
            if ch not in self.char_to_id:
                raise ValueError(f"Символ {s!r} вне алфавита L0")
            return self.char_to_id[ch]

        pool = self.levels[child_level]
        target = normalize_text(s)
        if not target:
            raise ValueError(f"Пустое значение источника для L{child_level}")
        for f in pool.fims:
            if f.consolidated and (f.decoded_text or self.decode_fim(child_level, f.fim_id)) == target:
                return f.fim_id

        if child_level == 1 and 2 <= len(target) <= 3:
            seq = tuple(self.char_to_id[c] for c in target)
            ev = pool._emit_seq(seq, self.current_time_ms, context_str=target, stage_str="Ручная настройка")
            if ev is not None:
                self._register_parent_links(1, ev.node_id)
                return ev.node_id
        raise ValueError(f"На уровне L{child_level} не найден ФИМ со значением {target!r}")

    def _compute_synapse_metrics(self) -> Dict[str, Any]:
        """Подсчитывает фактические синапсы между всеми уровнями L0 -> L1 -> ... -> L9."""
        per_lvl: Dict[int, Dict[str, int]] = {}
        for lvl in range(1, self.num_levels + 1):
            pool = self.levels[lvl]
            in_b1 = 0
            in_b2 = 0
            in_b3 = 0
            two_syn = 0
            three_syn = 0
            cons_cnt = 0
            tot_rec = 0
            for f in pool.fims:
                if not f.consolidated:
                    continue
                cons_cnt += 1
                tot_rec += f.recognitions
                if len(f.sources) == 2:
                    two_syn += 1
                else:
                    three_syn += 1
                sc = f.synapse_counts
                slots = f.synapse_slots
                for i, sl in enumerate(slots):
                    cnt = sc[i] if len(sc) == len(slots) else 1
                    if sl == 0:
                        in_b1 += cnt
                    elif sl == 1:
                        in_b2 += cnt
                    else:
                        in_b3 += cnt
            per_lvl[lvl] = {
                "consolidated_fims": cons_cnt,
                "two_synapse_fims": two_syn,
                "three_synapse_fims": three_syn,
                "total_recognitions": tot_rec,
                "actual_in_b1": in_b1,
                "actual_in_b2": in_b2,
                "actual_in_b3": in_b3,
                "actual_in_total": in_b1 + in_b2 + in_b3,
            }

        l0_elements: List[Dict[str, Any]] = []
        l1_pool = self.levels[1]
        l0_total_out = 0
        for nid in range(len(RUSSIAN_ALPHABET)):
            b1 = b2 = b3 = 0
            for fid, slot_idx in self.incoming_synapses[1].get(nid, ()):
                f = l1_pool.fims[fid]
                if not f.consolidated:
                    continue
                c = f.synapse_counts[slot_idx] if len(f.synapse_counts) == len(f.sources) else 1
                sl = f.synapse_slots[slot_idx] if slot_idx < len(f.synapse_slots) else slot_idx
                if sl == 0:
                    b1 += c
                elif sl == 1:
                    b2 += c
                else:
                    b3 += c
            tot = b1 + b2 + b3
            l0_total_out += tot
            l0_elements.append(
                {
                    "node_id": nid,
                    "char": self.id_to_char[nid],
                    "out_synapses": tot,
                    "b1": b1,
                    "b2": b2,
                    "b3": b3,
                    "quota": self.element_out_quotas.get(nid, self.synapses_per_element_l0),
                }
            )
        return {"per_lvl": per_lvl, "l0_elements": l0_elements, "l0_total_out": l0_total_out}

    def get_config(self) -> Dict[str, Any]:
        """Возвращает строго ТЗ-параметры сети, элементов L0 и уровней L1..L9."""
        metrics = self._compute_synapse_metrics()
        per_lvl = metrics["per_lvl"]
        levels_cfg: Dict[int, Dict[str, Any]] = {}
        for lvl, pool in self.levels.items():
            m = per_lvl[lvl]
            out_tot = per_lvl[lvl + 1]["actual_in_total"] if lvl < self.num_levels else 0
            cons = m["consolidated_fims"]
            levels_cfg[lvl] = {
                "level": lvl,
                "window_ms": pool.window_ms,
                "min_synapses": pool.min_synapses,
                "max_synapses": pool.max_synapses,
                "synapses_to_b1": pool.synapses_to_b1,
                "synapses_to_b2": pool.synapses_to_b2,
                "synapses_to_b3": pool.synapses_to_b3,
                "synapses_per_fim_out": pool.synapses_per_fim_out,
                "level2_max_chars": pool.level2_max_chars if lvl == 2 else None,
                "max_fims": pool.max_fims,
                "consolidated_fims": cons,
                "two_synapse_fims": m["two_synapse_fims"],
                "three_synapse_fims": m["three_synapse_fims"],
                "total_recognitions": m["total_recognitions"],
                "actual_in_b1": m["actual_in_b1"],
                "actual_in_b2": m["actual_in_b2"],
                "actual_in_b3": m["actual_in_b3"],
                "actual_in_total": m["actual_in_total"],
                "actual_out_total": out_tot,
                "avg_out_per_fim": round(out_tot / cons, 2) if cons else 0.0,
            }
        return {
            "num_levels": self.num_levels,
            "tick_ms": self.tick_ms,
            "instinct_cycle_ms": self.instinct_cycle_ms,
            "allow_associative_2of3": self.allow_associative_2of3,
            "level2_max_chars": self.level2_max_chars,
            "synapses_per_element_l0": self.synapses_per_element_l0,
            "l0_count": len(RUSSIAN_ALPHABET),
            "l0_total_out": metrics["l0_total_out"],
            "l0_elements": metrics["l0_elements"],
            "levels": levels_cfg,
        }

    def update_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Обновляет строго ТЗ-параметры сети, квоты синапсов L0->L1 и уровней L1..L9."""
        if "tick_ms" in cfg:
            self.tick_ms = max(12, min(1000, int(cfg["tick_ms"])))
        if "instinct_cycle_ms" in cfg:
            self.instinct_cycle_ms = max(self.tick_ms, min(5000, int(cfg["instinct_cycle_ms"])))
        if "allow_associative_2of3" in cfg:
            self.allow_associative_2of3 = bool(cfg["allow_associative_2of3"])
        if "level2_max_chars" in cfg:
            self.level2_max_chars = max(2, min(18, int(cfg["level2_max_chars"])))
            self.levels[2].level2_max_chars = self.level2_max_chars
        if "synapses_per_element_l0" in cfg and cfg["synapses_per_element_l0"] is not None:
            self.synapses_per_element_l0 = max(1, min(1_000_000, int(cfg["synapses_per_element_l0"])))
        if "element_quota" in cfg and isinstance(cfg["element_quota"], dict):
            eq = cfg["element_quota"]
            nid = int(eq.get("node_id", 0))
            if nid in self.id_to_char:
                self.element_out_quotas[nid] = max(1, min(1_000_000, int(eq.get("quota", self.synapses_per_element_l0))))

        if "min_synapses" in cfg:
            ms = 2 if int(cfg["min_synapses"]) <= 2 else 3
            for pool in self.levels.values():
                pool.min_synapses = ms
        if "max_synapses" in cfg:
            mx = 2 if int(cfg["max_synapses"]) <= 2 else 3
            for pool in self.levels.values():
                pool.max_synapses = max(pool.min_synapses, mx)

        apply_syn_counts = bool(cfg.get("apply_synapse_counts_to_existing", False))
        levels_patch = cfg.get("levels") or {}
        for k, lcfg in levels_patch.items():
            lvl = int(k)
            if lvl not in self.levels or not isinstance(lcfg, dict):
                continue
            pool = self.levels[lvl]
            if "window_ms" in lcfg and lcfg["window_ms"] is not None:
                pool.window_ms = max(24, min(2_000_000, int(lcfg["window_ms"])))
            if "min_synapses" in lcfg:
                pool.min_synapses = 2 if int(lcfg["min_synapses"]) <= 2 else 3
            if "max_synapses" in lcfg:
                pool.max_synapses = max(pool.min_synapses, 2 if int(lcfg["max_synapses"]) <= 2 else 3)
            if "synapses_per_fim_out" in lcfg and lcfg["synapses_per_fim_out"] is not None:
                pool.synapses_per_fim_out = max(1, min(1_000_000, int(lcfg["synapses_per_fim_out"])))
            if "synapses_to_b1" in lcfg and lcfg["synapses_to_b1"] is not None:
                pool.synapses_to_b1 = max(1, min(1000, int(lcfg["synapses_to_b1"])))
            if "synapses_to_b2" in lcfg and lcfg["synapses_to_b2"] is not None:
                pool.synapses_to_b2 = max(1, min(1000, int(lcfg["synapses_to_b2"])))
            if "synapses_to_b3" in lcfg and lcfg["synapses_to_b3"] is not None:
                pool.synapses_to_b3 = max(1, min(1000, int(lcfg["synapses_to_b3"])))
            if "max_fims" in lcfg and lcfg["max_fims"] is not None:
                pool.max_fims = max(len(pool.fims), min(5_000_000, int(lcfg["max_fims"])))
            if apply_syn_counts and any(x in lcfg for x in ("synapses_to_b1", "synapses_to_b2", "synapses_to_b3")):
                b1, b2, b3 = pool.synapses_to_b1, pool.synapses_to_b2, pool.synapses_to_b3
                for f in pool.fims:
                    if len(f.sources) == 2:
                        f.synapse_counts = (b1, b3)
                    else:
                        f.synapse_counts = (b1, b2, b3)
        return self.get_config()

    def update_fim(self, level: int, fim_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Редактирует конкретный ФИМ (слоты Б1..Б3, источники, кол-во синапсов на Б1..Б3 и от Г, окно)."""
        if level == 0:
            if fim_id not in self.id_to_char:
                raise ValueError(f"Элемент L0 #{fim_id} не найден")
            if "outgoing_synapses_per_fim" in data and data["outgoing_synapses_per_fim"] is not None:
                self.element_out_quotas[fim_id] = max(1, min(1_000_000, int(data["outgoing_synapses_per_fim"])))
            return self.explain_fim(0, fim_id)

        if level not in self.levels:
            raise ValueError(f"Уровень {level} не существует")
        pool = self.levels[level]
        if fim_id < 0 or fim_id >= len(pool.fims):
            raise ValueError(f"ФИМ L{level} #{fim_id} не найден")
        fim = pool.fims[fim_id]

        rebuild = False
        if "sources" in data and isinstance(data["sources"], list) and 2 <= len(data["sources"]) <= 3:
            new_sources = tuple(self._resolve_source_spec(level - 1, s) for s in data["sources"])
            if new_sources != fim.sources:
                fim.sources = new_sources
                fim.synapse_slots = (0, 2) if len(new_sources) == 2 else (0, 1, 2)
                rebuild = True

        n_src = len(fim.sources)
        if "synapse_slots" in data and isinstance(data["synapse_slots"], list) and len(data["synapse_slots"]) == n_src:
            fim.synapse_slots = tuple(int(s) for s in data["synapse_slots"])

        if "synapse_counts" in data and isinstance(data["synapse_counts"], list):
            raw_c = [max(1, min(10000, int(c))) for c in data["synapse_counts"][:n_src]]
            while len(raw_c) < n_src:
                raw_c.append(1)
            fim.synapse_counts = tuple(raw_c)
        elif rebuild:
            fim.synapse_counts = (1,) * n_src

        if "outgoing_synapses_per_fim" in data:
            og = data["outgoing_synapses_per_fim"]
            fim.outgoing_synapses_per_fim = int(og) if (og is not None and str(og).strip() != "" and int(og) > 0) else None

        if "custom_window_ms" in data:
            cw = data["custom_window_ms"]
            fim.custom_window_ms = int(cw) if (cw is not None and str(cw).strip() != "" and int(cw) > 0) else None

        if "recognitions" in data and data["recognitions"] is not None:
            fim.recognitions = max(0, int(data["recognitions"]))

        if "consolidated" in data:
            new_cons = bool(data["consolidated"])
            if new_cons != fim.consolidated:
                fim.consolidated = new_cons
                rebuild = True

        if rebuild:
            self._rebuild_level_indexes(level)
        else:
            fim.decoded_text = ""
        return self.explain_fim(level, fim_id, max_depth=5)

    def create_fim(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Создаёт новый вручную настроенный ФИМ на уровне level."""
        if level not in self.levels:
            raise ValueError(f"Уровень {level} не существует")
        pool = self.levels[level]
        if len(pool.fims) >= pool.max_fims:
            raise ValueError(f"Достигнут лимит пула L{level} ({pool.max_fims})")

        raw_sources = data.get("sources") or []
        if not (2 <= len(raw_sources) <= 3):
            raise ValueError("Нужно задать 2 или 3 источника (Б1..Б3)")
        sources = tuple(self._resolve_source_spec(level - 1, s) for s in raw_sources)
        n_src = len(sources)
        slots = tuple(int(x) for x in data["synapse_slots"][:n_src]) if (
            isinstance(data.get("synapse_slots"), list) and len(data["synapse_slots"]) == n_src
        ) else ((0, 2) if n_src == 2 else (0, 1, 2))

        counts = tuple(max(1, min(10000, int(c))) for c in (data.get("synapse_counts") or [1] * n_src)[:n_src])
        og = data.get("outgoing_synapses_per_fim")
        out_quota = int(og) if (og is not None and str(og).strip() != "" and int(og) > 0) else None
        cw = data.get("custom_window_ms")
        custom_win = int(cw) if (cw is not None and str(cw).strip() != "" and int(cw) > 0) else None

        fim_id = len(pool.fims)
        fim = FIM(
            fim_id=fim_id,
            level=level,
            synapse_slots=slots,
            sources=sources,
            consolidated=True,
            recognitions=max(0, int(data.get("recognitions", 1))),
            created_at_ms=self.current_time_ms,
            training_context=str(data.get("training_context", "Ручное создание")),
            training_stage=str(data.get("training_stage", "Инженерный конструктор")),
            synapse_counts=counts,
            outgoing_synapses_per_fim=out_quota,
            custom_window_ms=custom_win,
        )
        fim.decoded_text = "".join(self.decode_fim(level - 1, s) for s in sources)
        pool.fims.append(fim)
        if sources not in pool.pattern_index:
            pool.pattern_index[sources] = fim_id
        self._register_parent_links(level, fim_id)
        return self.explain_fim(level, fim_id, max_depth=5)

    def delete_fim(self, level: int, fim_id: int) -> Dict[str, Any]:
        """Отключает (деконсолидирует) ФИМ на уровне level и перестраивает индексы."""
        if level not in self.levels:
            raise ValueError(f"Уровень {level} не существует")
        pool = self.levels[level]
        if fim_id < 0 or fim_id >= len(pool.fims):
            raise ValueError(f"ФИМ L{level} #{fim_id} не найден")
        pool.fims[fim_id].consolidated = False
        self._rebuild_level_indexes(level)
        return {"level": level, "fim_id": fim_id, "consolidated": False}

    def decode_fim(self, level: int, fim_id: int) -> str:
        """Разворачивает ФИМ любого уровня обратно в читаемый текст с пробелами и пунктуацией."""
        if level == 0:
            return self.id_to_char.get(fim_id, "?")
        fim = self.levels[level].fims[fim_id]
        if fim.decoded_text:
            return fim.decoded_text
        fim.decoded_text = "".join(self.decode_fim(level - 1, src) for src in fim.sources)
        return fim.decoded_text

    def summary(self) -> Dict[int, Dict[str, Any]]:
        metrics = self._compute_synapse_metrics()
        per_lvl = metrics["per_lvl"]
        stats: Dict[int, Dict[str, Any]] = {}
        for lvl, pool in self.levels.items():
            m = per_lvl[lvl]
            out_tot = per_lvl[lvl + 1]["actual_in_total"] if lvl < self.num_levels else 0
            stats[lvl] = {
                "window_ms": pool.window_ms,
                "max_fims": pool.max_fims,
                "allocated_fims": len(pool.fims),
                "consolidated_fims": m["consolidated_fims"],
                "two_synapse_fims": m["two_synapse_fims"],
                "three_synapse_fims": m["three_synapse_fims"],
                "total_recognitions": m["total_recognitions"],
                "actual_in_b1": m["actual_in_b1"],
                "actual_in_b2": m["actual_in_b2"],
                "actual_in_b3": m["actual_in_b3"],
                "actual_in_total": m["actual_in_total"],
                "actual_out_total": out_tot,
                "synapses_per_fim_out": pool.synapses_per_fim_out,
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
