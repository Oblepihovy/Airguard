"""Загрузчик правил из JSON. Сами значения — в rules_v1.json рядом.

Все имена из JSON становятся атрибутами этого модуля, поэтому остальной код не меняется:
    from rules.rules_v1 import *          ->  PLANE_SEPARATION_H, ...
    from rules import rules_v1 as R       ->  R.PLANE_SEPARATION_H, ...

Другой файл правил без правки кода: AIRGUARD_RULES=путь/к/rules.json python src/run.py
"""

import json
import os
from pathlib import Path

RULES_PATH = Path(os.environ.get("AIRGUARD_RULES", Path(__file__).with_name("rules_v1.json")))


def _load(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = {}
    for group, values in raw.items():
        if group.startswith("_"):
            continue  # комментарии
        for name, value in values.items():
            if name.startswith("_"):
                continue
            if name in flat:
                raise ValueError(f"{path}: имя {name} встречается дважды")
            flat[name] = tuple(value) if isinstance(value, list) else value
    return flat


_RULES = _load(RULES_PATH)
globals().update(_RULES)
__all__ = list(_RULES)