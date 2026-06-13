"""Per-symbol source windowing for the Stage-1 diagnose prompt.

The diagnose prompt embeds the NEW pin's source so the proposer can read what
happened to each carried rule. Feeding cited files WHOLE overflows the model
context (transformers ``configuration_utils.py`` / ``quantization_config.py`` are
~100K chars each, well past a 16K window); the model then anchors on the biggest
file and invents rule_ids instead of triaging the residual. The R7 trial got
clean quality by embedding small per-SYMBOL windows (the cited class def plus the
specific field/validator span) rather than whole files.

This module formalises that windowing. It reuses the deterministic AST walker
(:func:`scripts.engine_producers._source_walker.iter_config_classes`) to locate a
class span, then narrows WITHIN the class to the cited field(s) and any member
(method / validator) whose body references them, plus a little context. The file
selector stays the diff-scoping-by-file-path the caller already does; this only
narrows within those files.

A char budget caps the assembled context (default comfortably under the model
window, leaving room for the instruction + residual entries). When windows would
exceed the budget the cited symbols are prioritised and what was dropped is
REPORTED, never silently truncated (the design's no-silent-truncation principle).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


def _iter_config_classes(module: ast.Module) -> list[ast.ClassDef]:
    """Reuse the deterministic AST class walker (lazy-imported).

    ``scripts/`` is on ``sys.path`` only when running from the repo root (the
    diagnose maintainer venue); it is NOT part of the installed package. The
    import is deferred to call time so merely importing this module - which the
    public ``api`` package does at init - never pulls in ``scripts`` and so never
    breaks the installed CLI.
    """
    from scripts.engine_producers._source_walker import iter_config_classes

    return iter_config_classes(module)


# Lines of context kept either side of a located span before rendering.
_CONTEXT_LINES = 3

# Lines kept either side of a field reference INSIDE a long member body. A large
# __init__ / validate can be 100+ lines; the proposer only needs the region that
# touches the cited field, not the whole method (the R7 keyword-window approach).
_REF_CONTEXT_LINES = 12

# Max body lines a single 1b surface window covers below its class header. A 1b
# request has no cited field, so it windows a BOUNDED leading slice of each class
# (declarations the proposer scans for missed rules) rather than the whole body -
# a class with hundreds of fields must not blow the budget or starve the rest.
_SURFACE_BODY_LINES = 40

# Default assembled-source budget. The live model runs num_ctx=16384; ~12K chars
# of source leaves headroom for the instruction template + the residual entries.
DEFAULT_BUDGET_CHARS = 12_000


@dataclass(frozen=True)
class SymbolRequest:
    """One symbol to window: a class and the field(s) cited for it.

    ``class_name`` is the bare class name (the last dotted component of a rule's
    ``native_type``, e.g. ``GenerationConfig``). ``fields`` are the cited field
    names (rule ``kwargs`` / ``match`` keys, e.g. ``cache_implementation``); empty
    means "window the whole class surface" (used by 1b gap-diagnose).
    """

    file: str
    class_name: str
    fields: tuple[str, ...] = ()


@dataclass
class WindowReport:
    """What windowing assembled and what it could not fit / locate.

    - ``source`` - the assembled, line-numbered, budget-capped prompt source.
    - ``truncated`` - human-readable notes for symbols dropped to stay under
      budget (never silent).
    - ``missing`` - notes for cited symbols that no longer exist in the new pin
      (class gone, or field gone from the class); flagged, not crashed.
    """

    source: str
    truncated: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.source)


@dataclass
class _Window:
    """A located, not-yet-rendered span within one file."""

    file: str
    label: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    cited: bool  # a window for an explicitly cited symbol (budget priority)


def _number(lines: list[str], start: int, end: int) -> str:
    """Render a 1-based inclusive line slice with 1-based line numbers."""
    lo = max(1, start)
    hi = min(len(lines), end)
    return "\n".join(f"{i + 1:5d}: {lines[i]}" for i in range(lo - 1, hi))


def _find_class(classes: Sequence[ast.ClassDef], name: str) -> ast.ClassDef | None:
    for cls in classes:
        if cls.name == name:
            return cls
    return None


def _reference_lines(node: ast.AST, field_name: str) -> list[int]:
    """Lines within ``node`` that name ``field_name`` (attribute / name / string).

    Catches ``self.cache_implementation``, a bare ``cache_implementation``
    reference, and a string mention inside a message template - the three places
    a validator body touches the field it constrains. Returns the sorted set of
    1-based line numbers so the caller can window AROUND the references rather
    than over the whole (possibly 100-line) member body.
    """
    lines: set[int] = set()
    for sub in ast.walk(node):
        hit = (
            (isinstance(sub, ast.Attribute) and sub.attr == field_name)
            or (isinstance(sub, ast.Name) and sub.id == field_name)
            or (
                isinstance(sub, ast.Constant)
                and isinstance(sub.value, str)
                and field_name in sub.value
            )
        )
        if hit and hasattr(sub, "lineno"):
            lines.add(sub.lineno)
    return sorted(lines)


def _windows_for_class(
    cls: ast.ClassDef,
    *,
    file: str,
    fields: tuple[str, ...],
) -> tuple[list[_Window], list[str]]:
    """Bounded windows for a class: its header plus the cited or surface spans.

    Returns ``(windows, missing_notes)``. With cited fields the window is the
    class header line plus, per field, the field annotation span and any member
    (method / validator) whose body references the field. With NO cited fields
    (1b surface mode) :func:`_surface_windows` returns one bounded leading slice
    of the class. Either way the windows stay bounded, never the whole (possibly
    100K-char) class body that would blow the budget and starve the rest.
    """
    header_end = cls.body[0].lineno - 1 if cls.body else cls.lineno
    header = _Window(
        file=file,
        label=f"class {cls.name}",
        start=cls.lineno,
        end=max(cls.lineno, header_end),
        cited=bool(fields),
    )

    if not fields:
        return _surface_windows(cls, file=file, header=header), []

    windows: list[_Window] = [header]
    missing: list[str] = []
    for fname in fields:
        located = False
        for stmt in cls.body:
            # The field's own annotation / assignment span.
            if fname in _assign_targets(stmt):
                windows.append(
                    _span(file, f"{cls.name}.{fname}", stmt.lineno, stmt.end_lineno or stmt.lineno)
                )
                located = True
            # A method / validator body: window a BOUNDED region around the first
            # field reference, not the whole (possibly 100-line) method. A long
            # __init__ that touches the field once must not blow the budget.
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                refs = _reference_lines(stmt, fname)
                if refs:
                    method_end = stmt.end_lineno or stmt.lineno
                    lo = max(stmt.lineno, refs[0] - _REF_CONTEXT_LINES)
                    hi = min(method_end, refs[0] + _REF_CONTEXT_LINES)
                    windows.append(
                        _span(file, f"{cls.name}.{stmt.name} (references {fname})", lo, hi)
                    )
                    located = True
        if not located:
            missing.append(f"{file}: field '{fname}' not found in class {cls.name}")
    return windows, missing


def _surface_windows(cls: ast.ClassDef, *, file: str, header: _Window) -> list[_Window]:
    """One bounded surface window for a fieldless (1b) request.

    The class header plus a LEADING slice of the body (up to
    :data:`_SURFACE_BODY_LINES` lines) - the declarations a 1b proposer scans for
    missed rules - never the whole (possibly hundreds-of-fields) class body. A
    fixed line cap keeps every class's surface window small so a huge class can
    neither blow the budget nor starve the rest. Marked uncited so cited-symbol
    windows keep budget priority over the surface.
    """
    body_end = cls.end_lineno or header.end
    end = min(body_end, header.start + _SURFACE_BODY_LINES)
    return [
        _Window(
            file=file,
            label=f"class {cls.name}",
            start=header.start,
            end=max(header.end, end),
            cited=False,
        )
    ]


def _span(file: str, label: str, start: int, end: int) -> _Window:
    """A cited window over a line span."""
    return _Window(file=file, label=label, start=start, end=end, cited=True)


def _assign_targets(stmt: ast.stmt) -> set[str]:
    """Names assigned / annotated by a class-body statement."""
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    if isinstance(stmt, ast.Assign):
        return {t.id for t in stmt.targets if isinstance(t, ast.Name)}
    return set()


def _expand(window: _Window, line_count: int) -> _Window:
    """Pad a window with a few context lines, clamped to the file bounds."""
    return _Window(
        file=window.file,
        label=window.label,
        start=max(1, window.start - _CONTEXT_LINES),
        end=min(line_count, window.end + _CONTEXT_LINES),
        cited=window.cited,
    )


def windowed_source(
    *,
    source_root: Path,
    requests: Sequence[SymbolRequest],
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> WindowReport:
    """Assemble per-symbol source windows under a char budget.

    For each :class:`SymbolRequest`, parse its file, locate the class via the
    shared AST walker, and extract bounded windows (class header + cited field /
    member spans, or the whole class for a fieldless surface request). Windows
    are rendered with 1-based line numbers so the model can cite real lines.

    Cited-symbol windows are admitted first; if the budget is exhausted the
    remaining windows are dropped and noted in :attr:`WindowReport.truncated`.
    Cited classes / fields that no longer exist in the new pin are noted in
    :attr:`WindowReport.missing` (empty window, flagged - never a crash).
    """
    # Group requests by file so each file is parsed once.
    by_file: dict[str, list[SymbolRequest]] = {}
    order: list[str] = []
    for req in requests:
        if req.file not in by_file:
            by_file[req.file] = []
            order.append(req.file)
        by_file[req.file].append(req)

    all_windows: list[_Window] = []
    file_lines: dict[str, list[str]] = {}
    missing: list[str] = []
    located_classes: set[str] = set()
    not_in_file: list[str] = []

    for rel in order:
        path = source_root / rel
        if not path.is_file():
            missing.append(f"{rel}: file not found under {source_root}")
            continue
        body = path.read_text()
        if not body.strip():
            missing.append(f"{rel}: file is empty")
            continue
        lines = body.splitlines()
        file_lines[rel] = lines
        try:
            module = ast.parse(body)
        except SyntaxError as exc:
            missing.append(f"{rel}: could not parse source ({exc})")
            continue
        classes = _iter_config_classes(module)
        for req in by_file[rel]:
            cls = _find_class(classes, req.class_name)
            if cls is None:
                # A class absent from THIS file may simply live in another cited
                # file (callers pass the file UNION across residuals). Defer the
                # missing-note until we know it was found nowhere.
                not_in_file.append(req.class_name)
                continue
            located_classes.add(req.class_name)
            windows, miss = _windows_for_class(cls, file=rel, fields=req.fields)
            missing.extend(miss)
            all_windows.extend(_expand(w, len(lines)) for w in windows)

    # A class genuinely vanished from the new pin only if it was found in NONE of
    # the requested files (not merely "absent from one of several cited files").
    for name in dict.fromkeys(not_in_file):
        if name not in located_classes:
            missing.append(f"class {name} not found in new pin")

    source, truncated = _render_under_budget(all_windows, file_lines, budget_chars)
    return WindowReport(source=source, truncated=truncated, missing=missing)


def _render_under_budget(
    windows: Sequence[_Window],
    file_lines: dict[str, list[str]],
    budget_chars: int,
) -> tuple[str, list[str]]:
    """Render windows (cited first) up to the budget; report what was dropped.

    Overlapping windows within a file are merged so a field span that falls
    inside a referencing method is not duplicated. Cited windows are admitted
    before fieldless surface windows so a tight budget keeps the load-bearing
    spans. Both groups round-robin by (file, class) so one large class cannot
    starve the rest: each class contributes its header before any class's later
    windows are considered.
    """
    merged = _merge_windows(windows)
    cited = _round_robin(merged, predicate=lambda w: w.cited)
    surface = _round_robin(merged, predicate=lambda w: not w.cited)

    rendered: list[tuple[_Window, str]] = []
    truncated: list[str] = []
    used = 0
    for window in [*cited, *surface]:
        lines = file_lines.get(window.file, [])
        block = (
            f"### FILE: {window.file} -- {window.label} "
            f"(lines {window.start}-{window.end})\n{_number(lines, window.start, window.end)}"
        )
        cost = len(block) + 2  # joiner
        if used + cost > budget_chars and rendered:
            truncated.append(
                f"{window.file}: {window.label} (lines {window.start}-{window.end}) "
                f"dropped to stay under {budget_chars}-char budget"
            )
            continue
        rendered.append((window, block))
        used += cost

    rendered.sort(key=lambda rb: (rb[0].file, rb[0].start))
    return "\n\n".join(block for _, block in rendered), truncated


def _round_robin(
    windows: Sequence[_Window], *, predicate: Callable[[_Window], bool]
) -> list[_Window]:
    """Interleave matching windows so each (file, class) contributes one per pass.

    Preserves source order within a group. With classes A and B this yields
    A0, B0, A1, B1, ... so a tight budget keeps the first window (the header) of
    every class before spending the rest on any one class - a huge class cannot
    starve the others. Grouping by class (not just file) matters in 1b mode where
    one file holds many config classes.
    """
    groups: dict[tuple[str, str], list[_Window]] = {}
    order: list[tuple[str, str]] = []
    for window in sorted(windows, key=lambda w: (w.file, w.start)):
        if not predicate(window):
            continue
        key = (window.file, _window_class(window))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(window)

    out: list[_Window] = []
    idx = 0
    while any(idx < len(groups[k]) for k in order):
        for k in order:
            if idx < len(groups[k]):
                out.append(groups[k][idx])
        idx += 1
    return out


def _window_class(window: _Window) -> str:
    """Class name owning a window, parsed from its label.

    Labels are ``class X`` / ``X.field`` / ``X.method (references f)``; a merged
    window's label concatenates constituents with ``; `` and the FIRST is the
    owner. Returns the class token so round-robin can group windows by class.
    """
    head = window.label.split(";", 1)[0].strip()
    if head.startswith("class "):
        return head[len("class ") :].strip()
    return head.split(".", 1)[0].split(" ", 1)[0]


def _merge_windows(windows: Sequence[_Window]) -> list[_Window]:
    """Merge overlapping / adjacent windows within the same file.

    A merged span is ``cited`` if any constituent was cited (so a method body
    that subsumes a field annotation keeps budget priority), and its label lists
    the merged constituents.
    """
    out: list[_Window] = []
    for window in sorted(windows, key=lambda w: (w.file, w.start)):
        if out and out[-1].file == window.file and window.start <= out[-1].end + 1:
            prev = out[-1]
            labels = prev.label if window.label in prev.label else f"{prev.label}; {window.label}"
            out[-1] = _Window(
                file=prev.file,
                label=labels,
                start=prev.start,
                end=max(prev.end, window.end),
                cited=prev.cited or window.cited,
            )
        else:
            out.append(window)
    return out
