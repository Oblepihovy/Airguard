"""Полный прогон одной командой: симуляция -> траектории в CSV -> график -> checker.

    python src/run.py                 # SearchControls, результаты в out/
    python src/run.py --straight      # все летят прямо (шаг «1»: нарушения ожидаемы)
    python src/run.py --gif           # дополнительно анимация поля опасности (~1 мин)

Код возврата 0, если checker не нашёл нарушений, иначе 1.
"""

import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checker import check_system
from controls import SearchControls
from rules import rules_v1 as R
from system import StraightControls, load_system



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--straight", action="store_true", help="без управления, все летят прямо")
    ap.add_argument("--gif", action="store_true", help="сохранить анимацию поля")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "straight" if a.straight else "search"
    controls = StraightControls if a.straight else SearchControls

    system = load_system(str(ROOT / "scenario.json"), str(ROOT / "flights.csv"), controls)
    system.run(R.SIM_DURATION_S)

    system.save_trajectory(str(out / f"trajectories_{name}.csv"))
    report = check_system(system)
    print(report.summary())
    for v in report.violations[:20]:
        print(f"  {v.rule:<12} {v.flight:>6} / {v.other:<8} t={v.t:6.0f} с  {v.value:9.1f} (норма {v.limit:g})")

    try:
        from visualize import animate, plot_tracks
    except ImportError:
        print("matplotlib не установлен — графики пропущены")
    else:
        plot_tracks(system, out / f"tracks_{name}.png")
        if a.gif:
            animate(system, out / f"field_{name}.gif")
    print(f"-> {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())