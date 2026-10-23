from pathlib import Path

import pytest

from system import StraightControls, load_system

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def straight_run():
    """Полный прогон 1200 с на данных задачи, все летят прямо (общий для всех тестов)."""
    system = load_system(str(ROOT / "scenario.json"), str(ROOT / "flights.csv"), StraightControls)
    system.run(1200.0)
    return system