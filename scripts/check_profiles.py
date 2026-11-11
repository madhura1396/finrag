"""Check the layout profile against your real statements (ADR 0003, decision 7).

    python -m scripts.check_profiles <pdf or directory> [...] [--numbered]

Prints exactly one line per file and nothing else, on stdout or stderr:
    PASS <name>
    FAIL <name> <check>[:<section>] ...
where every word after the name is a fixed check name from app.statement.CHECKS,
optionally with a section key. No amounts, dates, merchants, labels, mismatch
values or tracebacks are ever printed. An unexpected error prints
`FAIL <name> unexpected_error`. With --numbered, <name> is #1, #2, ... instead
of the file name, for when file names contain dates or account digits.

Reads PDFs only: no database, no model, no writes. Exits 1 if any file fails.
Imports only app.profile, app.reader and app.statement, never app.models,
which would load .env and build a database engine.
"""

import sys
import warnings
from pathlib import Path


def _quiet_mupdf(fitz) -> None:
    for switch in ("mupdf_display_errors", "mupdf_display_warnings"):
        if hasattr(fitz.TOOLS, switch):
            getattr(fitz.TOOLS, switch)(False)


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() == ".pdf"))
        else:
            files.append(p)
    return files


def check_file(path: Path, profile) -> list[str]:
    """Failing check tokens for one file; empty means it reconciles."""
    from app.reader import read_statement
    from app.statement import StatementRejected, validate

    try:
        data = read_statement(path, profile)
        mismatches = validate(data)
    except StatementRejected as e:
        mismatches = e.mismatches
    except Exception:
        return ["unexpected_error"]
    tokens = []
    for m in mismatches:
        token = m.check + (f":{m.section}" if m.section else "")
        if token not in tokens:
            tokens.append(token)
    return tokens


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    usage = "usage: python -m scripts.check_profiles <pdf or directory> [...] [--numbered]"
    numbered = "--numbered" in argv
    unknown = [a for a in argv if a.startswith("-") and a != "--numbered"]
    paths = [a for a in argv if not a.startswith("-")]
    if not paths or unknown:
        print(usage)                          # never treat a mistyped flag as a file name
        return 2

    warnings.simplefilter("ignore")
    import fitz
    _quiet_mupdf(fitz)
    from app.profile import ProfileError, load_single
    try:
        profile = load_single()
    except ProfileError:
        print("FAIL profile")                 # the profile file is malformed; nothing from it is printed
        return 2
    files = collect(paths)
    if not files:
        print("no PDF files found")
        return 2

    failed = False
    for i, path in enumerate(files, start=1):
        name = f"#{i}" if numbered else path.name
        tokens = check_file(path, profile)
        if tokens:
            failed = True
            print(f"FAIL {name} " + " ".join(tokens))
        else:
            print(f"PASS {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
