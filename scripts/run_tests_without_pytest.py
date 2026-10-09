"""Minimal pytest-compatible runner, used ONLY when the real pytest package
cannot be installed (offline / egress-restricted build environments).

It discovers tests/test_*.py, runs every top-level `test_*` function and
supports the subset of pytest features the suite uses: plain `assert`,
the `tmp_path` and `monkeypatch` fixtures, `pytest.raises` and
`pytest.approx` (via a tiny shim module injected as `pytest`).
On a normal machine run the real thing:  python -m pytest -q
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import shutil
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match:
            import re
            assert re.search(self.match, str(ev)), f"{ev!r} !~ {self.match}"
        return True


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.e, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        return math.isclose(other, self.e, rel_tol=self.rel, abs_tol=self.abs)

    def __repr__(self):
        return f"approx({self.e})"


def _install_pytest_shim():
    if importlib.util.find_spec("pytest") is not None:
        return
    shim = types.ModuleType("pytest")
    shim.raises = _Raises
    shim.approx = _Approx

    class _Mark:
        def __getattr__(self, _name):
            return lambda *a, **k: (a[0] if a and callable(a[0]) and not k else (lambda f: f))
    shim.mark = _Mark()
    shim.fixture = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["pytest"] = shim


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value, raising=True):
        old = getattr(target, name)
        self._undo.append(lambda: setattr(target, name, old))
        setattr(target, name, value)

    def setenv(self, key, value):
        import os
        old = os.environ.get(key)
        self._undo.append(lambda: os.environ.__setitem__(key, old) if old is not None else os.environ.pop(key, None))
        os.environ[key] = str(value)

    def delenv(self, key, raising=True):
        import os
        old = os.environ.pop(key, None)
        if old is not None:
            self._undo.append(lambda: os.environ.__setitem__(key, old))

    def undo(self):
        while self._undo:
            self._undo.pop()()


def main(root: str) -> int:
    root_p = Path(root).resolve()
    sys.path.insert(0, str(root_p))
    _install_pytest_shim()
    files = sorted((root_p / "tests").glob("test_*.py"))
    passed = failed = 0
    t0 = time.time()
    for f in files:
        spec = importlib.util.spec_from_file_location(f"tests.{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # like pytest: lets pickle find classes defined in test modules
        try:
            spec.loader.exec_module(mod)
        except Exception:
            failed += 1
            print(f"ERROR collecting {f.name}")
            traceback.print_exc()
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            kwargs, mp, tmp = {}, None, None
            params = inspect.signature(fn).parameters
            if "tmp_path" in params:
                tmp = Path(tempfile.mkdtemp())
                kwargs["tmp_path"] = tmp
            if "monkeypatch" in params:
                mp = MonkeyPatch()
                kwargs["monkeypatch"] = mp
            try:
                fn(**kwargs)
                passed += 1
                print(f"PASSED {f.name}::{name}")
            except Exception:
                failed += 1
                print(f"FAILED {f.name}::{name}")
                traceback.print_exc()
            finally:
                if mp:
                    mp.undo()
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{passed} passed, {failed} failed in {time.time() - t0:.2f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
