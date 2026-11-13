"""
Поэтапное обучение 9-уровневой ФИМ-сети «как маленького ребёнка» с нуля:
все ФИМы стартуют пустыми, и на каждом из 9 этапов формируется свой уровень:
  Этап 1 -> Уровень 1: слоги и короткие сочетания (2–3 симв.)
  Этап 2 -> Уровень 2: целые слова (<= 6 симв.)
  Этап 3 -> Уровень 3: короткие фразы (2–3 слова)
  Этап 4 -> Уровень 4: законченные предложения (2–3 фразы)
  Этап 5 -> Уровень 5: сцены (2–3 предложения)
  Этап 6 -> Уровень 6: эпизоды (2–3 сцены)
  Этап 7 -> Уровень 7: главы (2–3 эпизода)
  Этап 8 -> Уровень 8: части историй (2–3 главы)
  Этап 9 -> Уровень 9: целые сюжеты (2–3 части)
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Tuple
from fim_algorithm import InstinctNetwork, LETTERS


VOWELS = set("аеёиоуыэюя")
CONSONANTS = [ch for ch in LETTERS if ch not in VOWELS]

# Этап 1а: 3-символьные слоги, слоги с пробелом и предлоги (строго ДО 2-символьных!)
STAGE1_SYLLABLES_3 = [
    "при", "вет", "дом", "лес", "кот", "сад", "мир", "сон",
    " у ", " и ", " с ", " в ", " о ", " а ",
    "ма ", "па ", "ба ", "да ", "ла ", "на ", "ка ", "са ",
    "ва ", "бо ", "ре ", "то ", "ды ", "ли ", "но ", "хо ",
    "ло ", "ра ", "ду ", "ня ", "бе ", "ат ", "ук ",
]

# Этап 1б: 2-символьные слоги и знаки препинания с пробелами
STAGE1_SYLLABLES_2 = (
    [
        "ма", "мы", "ла", "ра", "му", "па", "ба", "де", "да",
        "пи", "пе", "ли", "лу", "на", "во", "не", "бо", "мо",
        "ре", "ле", "то", "зи", "ро", "са", "ка", "со", "ва",
        "бе", "га", "иг", "по", "сю", "ту", "до", "ти", "хо",
        "ми", "ло", "яс", "но", "бы", "од", "те", "ме", "ня",
        "ку", "ок", "чи", "та", "л ", "си", "гу", "ля", "ви",
        "лю", "би", "ва", "ри", "хо", "ди", "ры", "бу", "ду",
        "вс", "ё ", "оч", "ен", "ь ", "к ", "кн", "иг", "у.",
        "ки", "у ", "бр", "вн", "го", "ры", "се", "ст",
        ", ", ". ", "! ", "? ",
    ]
    + [c + v for c in CONSONANTS for v in VOWELS]
    + [v + c for v in VOWELS for c in CONSONANTS]
)

# Этап 2: Отдельные слова (строго <= 6 символов; 3-сложные со знаками идут ДО 2-сложных!)
STAGE2_WORDS = [
    # 3-сложные (6 символов) со знаками и пробелами
    "дела? ", ", как ",
    "раму. ", "окно. ", "кашу. ", "рыбу. ", "воду. ", "море. ",
    "лето. ", "дома. ", "тихо. ", "мило. ", "рано. ", "реки. ", "леса. ",
    "саду. ", "поля. ", "небе. ", "горы. ", "ясно. ", "туда. ",
    "раму, ", "окно, ", "дома, ", "саду, ", "реки, ",
    "читал ", "писал ", "сидел ", "гулял ", "ловил ", "любил ",
    "носил ", "варил ", "ходил ", "видел ", "очень ", "берега", "книгу.",
    # 2-сложные слова (4–6 символов)
    "привет", "мама", "раму", "окно", "дела?",
    "мама ", "папа ", "баба ", "деда ", "брат ", "внук ",
    "дом у ", "лес у ", "сад у ", "кот и ", "мир и ", "сон и ",
    "утром ", "днём ", "тихо ", "мило ", "ясно ", "рано ", "было ",
    "дома ", "сюда ", "туда ", "все ", "всё ",
    "мыла ", "пила ", "пели ",
]

# Этап 3: Законченные короткие фразы (из 2–3 слов Этапа 2)
STAGE3_PHRASES = [
    "привет, как дела? ",
    "мама мыла раму",
    "мама мыла раму. ",
    "дом у берега",
    "всё очень мило. ",
    "всё очень тихо. ",
    "всё очень ясно. ",
    "мама мыла окно. ",
    "мама варил кашу. ",
    "папа читал книгу.",
    "папа гулял рано. ",
    "папа любил море. ",
    "папа видел горы. ",
    "баба варил кашу. ",
    "баба пели дома. ",
    "деда ловил рыбу. ",
    "деда сидел дома. ",
    "брат ходил туда. ",
    "брат носил воду. ",
    "внук гулял рано. ",
    "внук писал дома. ",
    "дом у реки. ",
    "лес у поля. ",
    "сад у дома. ",
    "кот и море. ",
    "мир и лето. ",
    "сон и небо. ",
    "утром было тихо. ",
    "утром было ясно. ",
    "днём было мило. ",
    "днём было рано. ",
]

# Этап 4: Законченные предложения (из 2–3 фраз Этапа 3)
STAGE4_SENTENCES = [
    "привет, как дела? всё очень мило. ",
    "привет, как дела? всё очень тихо. ",
    "привет, как дела? утром было ясно. ",
    "мама мыла раму. папа гулял рано. ",
    "мама мыла окно. баба варил кашу. ",
    "дом у реки. утром было тихо. ",
    "лес у поля. днём было ясно. ",
    "сад у дома. мир и лето. ",
    "деда ловил рыбу. внук гулял рано. ",
    "брат носил воду. баба пели дома. ",
    "папа любил море. папа видел горы. ",
    "утром было тихо. кот и море. ",
    "днём было мило. сон и небо. ",
    "деда сидел дома. брат ходил туда. ",
    "внук писал дома. папа видел горы. ",
    "мама мыла раму. всё очень тихо. ",
]


def _compose_level_units(prev_units: List[str], count: int, seed: int) -> List[str]:
    """Собирает составные блоки следующего уровня из 2–3 выученных блоков предыдущего уровня."""
    rng = random.Random(seed)
    units: List[str] = []
    if len(prev_units) >= 3:
        units.append(prev_units[0] + prev_units[1])
        units.append(prev_units[0] + prev_units[1] + prev_units[2])
        units.append(prev_units[3] + prev_units[4] if len(prev_units) > 4 else prev_units[1] + prev_units[2])

    while len(units) < count:
        k = 2 if len(units) % 3 == 0 else 3
        chosen = rng.sample(prev_units, k=min(k, len(prev_units)))
        units.append("".join(chosen))
    return units


def _load_corpus_file(path: str) -> Tuple[List[str], List[str]]:
    words: List[str] = []
    sentences: List[str] = []
    if not os.path.exists(path):
        return words, sentences
    mode = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "[WORDS]":
                mode = "words"
                continue
            if line == "[SENTENCES]":
                mode = "sentences"
                continue
            if mode == "words":
                words.append(line)
            elif mode == "sentences":
                sentences.append(line)
    return words, sentences


def _train_russian_corpus(net: InstinctNetwork, corpus_path: str) -> None:
    """Дообучает сеть на большом русском корпусе (частые слова + предложения RuCoLA по главам)."""
    words, sentences = _load_corpus_file(corpus_path)
    for w in words:
        piece = w + " "
        net.feed_stream(piece, flush_end=True, stage="Корпус СЧР: частые слова")
        net.feed_stream(piece, flush_end=True, stage="Корпус СЧР: частые слова")

    chapter_size = 32
    for idx in range(0, len(sentences), chapter_size):
        chapter = " ".join(sentences[idx : idx + chapter_size])
        stage_name = f"Корпус RuCoLA: глава #{idx // chapter_size + 1}"
        for _ in range(9):
            net.feed_stream(chapter, flush_end=True, stage=stage_name)


def train_default_network(include_corpus: bool = True) -> InstinctNetwork:
    """
    Обучает абсолютно пустую сеть (ёмкость 1 000 000 ФИМ на уровень) с нуля:
    сначала 9 этапов взросления ребёнка, затем большой русский корпус слов и текстов.
    """
    net = InstinctNetwork(num_levels=9, max_fims_per_level=1_000_000)

    # Этап 1: Слоги (Уровень 1)
    for syl in STAGE1_SYLLABLES_3:
        net.feed_stream(syl, flush_end=True, stage="Этап 1: Слоги")
    for syl in STAGE1_SYLLABLES_2:
        net.feed_stream(syl, flush_end=True, stage="Этап 1: Слоги")

    # Этап 2: Слова (Уровень 2)
    for word in STAGE2_WORDS:
        net.feed_stream(word, flush_end=True, stage="Этап 2: Слова")
        net.feed_stream(word, flush_end=True, stage="Этап 2: Слова")

    # Этап 3: Короткие фразы (Уровень 3)
    for phrase in STAGE3_PHRASES:
        net.feed_stream(phrase, flush_end=True, stage="Этап 3: Фразы")
        net.feed_stream(phrase, flush_end=True, stage="Этап 3: Фразы")

    # Этап 4: Законченные предложения (Уровень 4)
    for sent in STAGE4_SENTENCES:
        net.feed_stream(sent, flush_end=True, stage="Этап 4: Предложения")
        net.feed_stream(sent, flush_end=True, stage="Этап 4: Предложения")

    # Этапы 5–9: Сцены -> Эпизоды -> Главы -> Части -> Сюжеты
    stage5_scenes = _compose_level_units(STAGE4_SENTENCES, count=24, seed=105)
    stage6_episodes = _compose_level_units(stage5_scenes, count=24, seed=106)
    stage7_chapters = _compose_level_units(stage6_episodes, count=20, seed=107)
    stage8_parts = _compose_level_units(stage7_chapters, count=18, seed=108)
    stage9_stories = _compose_level_units(stage8_parts, count=16, seed=109)

    stages = [
        ("Этап 5: Сцены", stage5_scenes),
        ("Этап 6: Эпизоды", stage6_episodes),
        ("Этап 7: Главы", stage7_chapters),
        ("Этап 8: Части историй", stage8_parts),
        ("Этап 9: Целые сюжеты", stage9_stories),
    ]
    for stage_name, units in stages:
        for u in units:
            net.feed_stream(u, flush_end=True, stage=stage_name)
            net.feed_stream(u, flush_end=True, stage=stage_name)

    if include_corpus:
        corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_ru.txt")
        _train_russian_corpus(net, corpus_path)

    return net


def main() -> None:
    empty_check = InstinctNetwork(num_levels=9, max_fims_per_level=1_000_000)
    print("Проверка пустой сети до обучения:", {l: len(p.fims) for l, p in empty_check.levels.items()})

    t0 = time.perf_counter()
    net = train_default_network()
    dt = time.perf_counter() - t0

    max_lvl2_len = max(
        (len(net.decode_fim(2, f.fim_id)) for f in net.levels[2].fims if f.consolidated),
        default=0,
    )

    total_cons = sum(st["consolidated_fims"] for st in net.summary().values())
    print(f"Обучение (детский этап + большой русский корпус) завершено за {dt:.2f} сек.")
    print(f"Макс. длина ФИМ на уровне 2: {max_lvl2_len} симв. (лимит <= 6) | Всего ФИМ: {total_cons}")
    print("\n=== СТАТИСТИКА ПО ВСЕМ 9 УРОВНЯМ (1 000 000 ФИМ/уровень) ===")
    for lvl, st in net.summary().items():
        print(
            f"Уровень {lvl} (окно {st['window_ms']:6d} мс): "
            f"консолидировано ФИМ = {st['consolidated_fims']:6d} / {st['max_fims']} | "
            f"узнаваний = {st['total_recognitions']:8d}"
        )

    print("\n=== ПРИМЕРЫ ФИМОВ ПО УРОВНЯМ (2..6) ===")
    for lvl in range(2, 7):
        fims = [f for f in net.levels[lvl].fims if f.consolidated][:5]
        samples = [f"{net.decode_fim(lvl, f.fim_id)!r}" for f in fims]
        print(f"Уровень {lvl}: " + ", ".join(samples))

    print("\n=== ПРИМЕР СЮЖЕТОВ НА УРОВНЕ 9 ===")
    l9_fims = [f for f in net.levels[9].fims if f.consolidated]
    if l9_fims:
        sample = net.decode_fim(9, l9_fims[0].fim_id)
        print(f"Уровень 9 (ФИМ #0, длина {len(sample)} симв.): {sample[:220]!r}...")


if __name__ == "__main__":
    main()
