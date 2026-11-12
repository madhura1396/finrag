"""scripts/check_profiles.py: its output contract (ADR 0003, decision 7; .claude/CLAUDE.md Data safety)."""

import os
import re
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from app.statement import CHECKS, SECTIONS
from scripts import check_profiles
from tests.fixtures.make_wells_fargo import make_statement, standard_spec

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^(PASS|FAIL) (\S+)(?: (.+))?$")


def run(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL",)}
    return subprocess.run([sys.executable, "-m", "scripts.check_profiles", *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


@pytest.fixture
def statements(tmp_path):
    good = tmp_path / "good.pdf"
    make_statement(good, standard_spec())
    bad_spec = standard_spec()
    bad_spec.rows[4] = replace(bad_spec.rows[4], raw_amount="8.4O")
    bad = tmp_path / "bad.pdf"
    make_statement(bad, bad_spec)
    return tmp_path, good, bad


def test_output_is_only_pass_fail_and_check_names(statements):
    folder, good, bad = statements
    out = run(folder)
    assert out.returncode == 1 and out.stderr == ""
    lines = out.stdout.splitlines()
    assert lines == ["FAIL bad.pdf " + lines[0].split(" ", 2)[2], "PASS good.pdf"]
    for line in lines:
        m = LINE.match(line)
        assert m
        for token in (m[3] or "").split():
            check, _, section = token.partition(":")
            assert check in CHECKS and (not section or section in SECTIONS)


def test_a_broken_statement_leaks_nothing_from_the_document(statements):
    _, _, bad = statements
    out = run(bad)
    text = out.stdout + out.stderr
    for leak in ("8.4O", "8.40", "CORNER", "CAFE", "RIVERTON", "09/12", "2026", "250.00", "$"):
        assert leak not in text


def test_all_passing_exits_zero(statements):
    _, good, _ = statements
    out = run(good)
    assert (out.returncode, out.stdout, out.stderr) == (0, "PASS good.pdf\n", "")


def test_numbered_hides_file_names(statements):
    folder, _, _ = statements
    out = run(folder, "--numbered")
    assert [l.split()[1] for l in out.stdout.splitlines()] == ["#1", "#2"]
    assert "good" not in out.stdout and "bad" not in out.stdout


def test_unexpected_errors_print_only_the_check_name(monkeypatch, tmp_path, capsys):
    import app.reader

    def explode(path, profile):
        raise RuntimeError("secret statement text 123.45")
    monkeypatch.setattr(app.reader, "read_statement", explode)
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4")
    assert check_profiles.main([str(tmp_path / "x.pdf")]) == 1
    captured = capsys.readouterr()
    assert captured.out == "FAIL x.pdf unexpected_error\n" and captured.err == ""


def test_the_script_never_imports_the_database_models(statements):
    _, good, _ = statements
    code = ("import sys; from scripts import check_profiles; check_profiles.main([sys.argv[1]]); "
            "print('MODELS' if 'app.models' in sys.modules else 'CLEAN')")
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    out = subprocess.run([sys.executable, "-c", code, str(good)], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "CLEAN"


@pytest.mark.parametrize("args", [(), ("{folder}", "--numberd"), ("{empty}",)])
def test_usage_errors_exit_two_and_print_no_file_names(statements, tmp_path_factory, args):
    folder, _, _ = statements
    empty = tmp_path_factory.mktemp("empty")
    out = run(*(a.format(folder=folder, empty=empty) for a in args))
    assert out.returncode == 2 and "PASS" not in out.stdout
    assert "good" not in out.stdout and "bad" not in out.stdout
