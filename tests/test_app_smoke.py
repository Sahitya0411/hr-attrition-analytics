"""End-to-end smoke test: the whole Streamlit script runs without exceptions."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_without_exceptions() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6
    assert at.metric[0].label == "Employees"


def test_app_survives_narrow_filter() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    at.sidebar.multiselect(key="f_depts").set_value(["Human Resources"]).run()
    assert not at.exception, [e.value for e in at.exception]


def test_user_sql_box_runs_and_blocks_writes() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    next(b for b in at.button if b.label == "Run query").click().run()
    assert at.success and "rows returned" in at.success[0].value

    at.text_area[0].set_value("DELETE FROM employees").run()
    next(b for b in at.button if b.label == "Run query").click().run()
    assert any("blocked" in e.value for e in at.error)
