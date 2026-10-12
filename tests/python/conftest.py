"""Keep TensorRT engines built by the tests out of the user's ~/.cache/ivit/engines."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_engine_cache(tmp_path_factory):
    try:
        import ivit
    except ImportError:
        yield
        return
    if not ivit.is_cpp_available():
        yield
        return
    ivit.set_cache_dir(str(tmp_path_factory.mktemp("engine-cache")))
    yield
    ivit.set_cache_dir("")
