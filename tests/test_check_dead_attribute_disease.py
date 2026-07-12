from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.check_dead_attribute_disease as scanner


def test_flags_getattr_on_never_assigned_self_attribute(tmp_path: Path) -> None:
    src = tmp_path / "bad.py"
    src.write_text(
        "class Widget:\n"
        "    def forward(self):\n"
        "        state = getattr(self, 'inference_state', None)\n"
        "        return state\n",
        encoding="utf-8",
    )
    findings = scanner.scan_file(src)
    assert len(findings) == 1
    assert findings[0].attr == "inference_state"
    assert findings[0].call == "getattr"


def test_does_not_flag_getattr_when_attribute_is_assigned_in_init(tmp_path: Path) -> None:
    src = tmp_path / "good.py"
    src.write_text(
        "class Widget:\n"
        "    def __init__(self):\n"
        "        self.inference_state = None\n"
        "    def forward(self):\n"
        "        state = getattr(self, 'inference_state', None)\n"
        "        return state\n",
        encoding="utf-8",
    )
    findings = scanner.scan_file(src)
    assert findings == []


def test_does_not_flag_hasattr_when_attribute_set_via_setattr(tmp_path: Path) -> None:
    src = tmp_path / "dynamic.py"
    src.write_text(
        "class Widget:\n"
        "    def configure(self, **kwargs):\n"
        "        for k, v in kwargs.items():\n"
        "            setattr(self, 'router_state', v)\n"
        "    def forward(self):\n"
        "        if hasattr(self, 'router_state'):\n"
        "            return self.router_state\n"
        "        return None\n",
        encoding="utf-8",
    )
    findings = scanner.scan_file(src)
    assert findings == []


def test_ignores_getattr_on_non_self_targets(tmp_path: Path) -> None:
    src = tmp_path / "other_obj.py"
    src.write_text(
        "class Widget:\n"
        "    def forward(self, cfg):\n"
        "        return getattr(cfg, 'learning_rate', 3e-4)\n",
        encoding="utf-8",
    )
    findings = scanner.scan_file(src)
    assert findings == []
