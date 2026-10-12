"""Python log sink contract (needs the built bindings)."""

import threading

import pytest

import ivit

pytestmark = pytest.mark.skipif(
    not ivit.is_cpp_available(), reason="C++ bindings are not built"
)


@pytest.fixture(autouse=True)
def restore_logging():
    yield
    ivit.set_log_callback(None)
    ivit.set_log_level("info")


def test_callback_receives_level_and_message():
    seen = []
    ivit.set_log_callback(lambda level, message: seen.append((level, message)))
    ivit.set_log_level("info")
    assert (1, "Log level set to: info") in seen


def test_level_filter_applies_to_callback():
    seen = []
    ivit.set_log_callback(lambda level, message: seen.append(message))
    ivit.set_log_level("error")
    ivit.set_log_level("warning")  # announcement is INFO, filtered at warning
    assert not any("Log level set to" in message for message in seen)


def test_callback_exceptions_do_not_propagate():
    def broken(level, message):
        raise RuntimeError("sink failure")

    ivit.set_log_callback(broken)
    ivit.set_log_level("info")  # must not raise


def test_non_callable_is_rejected_and_none_restores_stderr(capfd):
    with pytest.raises(TypeError):
        ivit.set_log_callback(42)
    ivit.set_log_callback(None)
    ivit.set_log_level("info")
    captured = capfd.readouterr()
    assert "Log level set to: info" in captured.err
    assert captured.out == ""


def test_callback_can_be_called_while_other_threads_log():
    seen = []
    lock = threading.Lock()

    def sink(level, message):
        with lock:
            seen.append(message)

    ivit.set_log_callback(sink)
    threads = [threading.Thread(target=ivit.set_log_level, args=("info",)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == 8
