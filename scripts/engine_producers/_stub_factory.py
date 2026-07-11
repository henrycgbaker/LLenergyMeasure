"""Factory helpers for engine-producer dispatcher shims.

Each engine-producer stub at ``scripts/engine_producers/<engine>_<producer>.py``
forwards module-level attribute access to a per-version archive resolved from
the SSOT's ``library.current_version``. Without this factory, every stub
repeats the same ~70 lines of sys.path defence, late-import boilerplate,
resolver, PEP 562 ``__getattr__``, and typed forwarder. The factory collapses
each stub to ~5 lines that pick (engine, producer) and bind the right
``main`` or ``discover`` shape.

Resolution is cached process-lifetime via ``functools.cache`` on
``_resolve_producer_cached``. Tests that need to swap the SSOT mid-process
call ``_resolve_producer_cached.cache_clear()`` in a fixture.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine_versions._dispatcher import ProducerKind


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    script_dir_resolved = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if p != "" and Path(p).resolve() != script_dir_resolved]


@functools.cache
def _resolve_producer_cached(engine: str, producer: ProducerKind) -> ModuleType:
    from engine_versions._dispatcher import load_producer
    from scripts.engine_producers._current import load_current

    current = load_current(engine)
    library = current.get("library")
    if not isinstance(library, dict) or "current_version" not in library:
        raise ValueError(
            f"engine_versions/{engine}/current.yaml is missing "
            "library.current_version; cannot resolve archived "
            f"{engine} {producer} machinery."
        )
    return load_producer(
        engine=engine,
        version=str(library["current_version"]),
        producer=producer,
    )


def _make_getattr(resolver: Callable[[], ModuleType], module_name: str) -> Callable[[str], Any]:
    def __getattr__(name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        machinery = resolver()
        try:
            return getattr(machinery, name)
        except AttributeError as exc:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from exc

    return __getattr__


def make_static_stub(
    engine: str,
) -> tuple[
    Callable[[], ModuleType],
    Callable[[str], Any],
    Callable[..., int],
]:
    """Factory for a static-invariant-miner shim. Returns (resolver, __getattr__, main)."""
    _ensure_project_root_on_path()

    def resolver() -> ModuleType:
        return _resolve_producer_cached(engine, "static_invariant_miner")

    module_name = f"scripts.engine_producers.{engine}_static_invariant_miner"
    getattr_fn = _make_getattr(resolver, module_name)

    def main(argv: list[str] | None = None) -> int:
        return int(resolver().main(argv))

    return resolver, getattr_fn, main


def make_dynamic_stub(
    engine: str,
) -> tuple[
    Callable[[], ModuleType],
    Callable[[str], Any],
    Callable[..., int],
]:
    """Factory for a dynamic-invariant-miner shim. Returns (resolver, __getattr__, main)."""
    _ensure_project_root_on_path()

    def resolver() -> ModuleType:
        return _resolve_producer_cached(engine, "dynamic_invariant_miner")

    module_name = f"scripts.engine_producers.{engine}_dynamic_invariant_miner"
    getattr_fn = _make_getattr(resolver, module_name)

    def main(argv: list[str] | None = None) -> int:
        return int(resolver().main(argv))

    return resolver, getattr_fn, main


def make_schema_stub(
    engine: str,
) -> tuple[
    Callable[[], ModuleType],
    Callable[[str], Any],
    Callable[[Path, str | None], dict[str, Any]],
]:
    """Factory for a schema-introspector shim. Returns (resolver, __getattr__, discover)."""
    _ensure_project_root_on_path()

    def resolver() -> ModuleType:
        return _resolve_producer_cached(engine, "schema_introspector")

    module_name = f"scripts.engine_producers.{engine}_schema_introspector"
    getattr_fn = _make_getattr(resolver, module_name)

    def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
        return resolver().discover(repo_root, image_ref)  # type: ignore[no-any-return]

    return resolver, getattr_fn, discover
