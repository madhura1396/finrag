"""Behaviour that needs separate real connections: the data_version lock, and
the reset script end to end. These commit for real on finrag_test, so they
leave it seeded and empty of statements, as the schema fixture does."""

import threading
import time

import pytest
from sqlalchemy.orm import Session

from app import categorization as cz

pytestmark = pytest.mark.integration


def test_apply_waits_for_the_lock_and_then_sees_the_newer_version(engine):
    payload = {"kind": "create_category", "params": {"name": "Lock Test Category"}}
    with Session(bind=engine) as reader:
        token = cz.preview(reader, payload)["token"]

    holder_has_lock = threading.Event()

    def hold_lock_then_bump():
        with Session(bind=engine) as holder:
            cz.lock_version(holder)
            holder_has_lock.set()
            time.sleep(0.5)
            cz.bump_version(holder)          # a concurrent writer (e.g. an upload) commits
            holder.commit()

    thread = threading.Thread(target=hold_lock_then_bump)
    thread.start()
    assert holder_has_lock.wait(5)

    started = time.monotonic()
    with Session(bind=engine) as applier, pytest.raises(cz.StalePreview):
        cz.apply(applier, payload, token)
    waited = time.monotonic() - started
    thread.join()

    assert waited >= 0.3, "apply did not wait for the lock holder"


def test_reset_script_rebuilds_finrag_test_and_refuses_bad_input(test_url, capsys):
    from scripts import reset_db

    url = test_url.render_as_string(hide_password=False)
    assert reset_db.main(["--database-url", url, "--env", "test", "--confirm-name", "wrong", "--yes"]) == 1
    assert reset_db.main(["--database-url", url, "--env", "production",
                          "--confirm-name", "finrag_test", "--yes"]) == 1      # marker is 'test'
    assert reset_db.main(["--database-url", url, "--env", "test", "--confirm-name", "finrag_test", "--yes"]) == 0

    output = capsys.readouterr()
    assert test_url.password is None or test_url.password not in output.out + output.err

    with Session(bind=engine_for(test_url)) as session:
        assert cz.current_version(session) == 0
        assert len(cz.category_store.current_categories(session)) == 10


def engine_for(url):
    from sqlalchemy import create_engine
    return create_engine(url)
