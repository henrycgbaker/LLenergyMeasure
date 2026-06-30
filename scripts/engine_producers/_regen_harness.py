"""Shared write-or-check CLI scaffold for the two regen scripts.

``regen_engine_corpus.py`` (SSOT outputs -> src/ shadow sync) and
``regen_engine_configs.py`` (typed config.py codegen) share the same CLI shape:
a mutually-exclusive ``--check`` (default) / ``--write`` mode group, a repeatable
``--engine`` filter, a per-engine loop that accumulates drift + changed-file
reports, and an epilogue that either reports what ``--write`` wrote or prints the
drift report and exits non-zero.

The two scripts diverge only in their per-engine sync logic and in their epilogue
WORDING (the corpus write-report prints a per-file ``wrote:`` list with an
"already in sync" fallback; the configs write-report prints a per-engine line),
so :func:`run` parametrises those as callbacks and keeps each script's
``sync_engine`` local.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

# A per-engine sync callback: given the engine name and the parsed args, sync (or
# check) that engine and return ``(drift, changed)`` - the drift reports
# (non-empty fails --check) and the files (re)written under --write.
SyncEngine = Callable[[str, argparse.Namespace], tuple[list[str], list[str]]]

# A write-mode reporter: given every (re)written file across all engines, print
# the script-specific "what was written" report. Only called under --write.
ReportWritten = Callable[[list[str]], None]


def run(
    *,
    argv: Sequence[str] | None,
    description: str,
    engines: Sequence[str],
    sync_engine: SyncEngine,
    report_written: ReportWritten,
    check_footer: str,
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
) -> int:
    """Drive one regen script's ``--check`` / ``--write`` CLI.

    ``engines`` is the full registry to fan out over by default (passed by the
    caller so a monkeypatched ``ENGINES`` flows through). ``sync_engine`` does the
    per-engine work; ``report_written`` prints the script's write-mode report;
    ``check_footer`` is appended to stderr after the per-file drift reports under
    ``--check``. ``configure_parser`` may add script-specific options (e.g. the
    corpus script's ``--only``) before parsing.
    """
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify SSOT/shadow parity; exit 1 with a per-file drift report. Default mode.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate (mutates the working tree) and report what changed.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=tuple(engines),
        help="Restrict to one or more engines (repeatable). Default: all engines.",
    )
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)

    selected = tuple(args.engine) if args.engine else tuple(engines)
    all_drift: list[str] = []
    all_changed: list[str] = []
    for engine in selected:
        drift, changed = sync_engine(engine, args)
        all_drift.extend(drift)
        all_changed.extend(changed)

    if args.write:
        report_written(all_changed)
        return 0

    if all_drift:
        for entry in all_drift:
            print(entry, file=sys.stderr)
        print(check_footer, file=sys.stderr)
        return 1
    return 0
