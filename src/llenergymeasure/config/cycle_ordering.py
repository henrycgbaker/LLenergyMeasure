"""Execution-sequence ordering across repeated study cycles.

Turns a validated experiment list into the ordered run sequence for a given
number of cycles (:func:`apply_cycles`) and reports where the larger cycle-gap
pauses fall in that sequence (:func:`cycle_boundary_indices`). Distinct from
grid expansion, which produces the experiment list this module orders.
"""

from __future__ import annotations

import random

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.utils.compat import StrEnum


class ExperimentOrder(StrEnum):
    SEQUENTIAL = "sequential"
    INTERLEAVE = "interleave"
    SHUFFLE = "shuffle"
    REVERSE = "reverse"
    LATIN_SQUARE = "latin_square"


def apply_cycles(
    experiments: list[ExperimentConfig],
    n_cycles: int,
    experiment_order: ExperimentOrder,
    study_design_hash: str,
    shuffle_seed: int | None = None,
) -> list[ExperimentConfig]:
    """Return the ordered execution sequence for n_cycles repetitions.

    sequential:    [A, A, A, B, B, B]  - all cycles of each experiment together
    interleave:    [A, B, A, B, A, B]  - one cycle of each experiment, repeated
    shuffle:       random per-cycle order, seeded from study_design_hash by default
    reverse:       alternating forward/backward per cycle - [A, B, B, A, A, B]
    latin_square:  Williams balanced latin square (counterbalances carryover effects)
    """
    if experiment_order == ExperimentOrder.SEQUENTIAL:
        return [exp for exp in experiments for _ in range(n_cycles)]

    if experiment_order == ExperimentOrder.INTERLEAVE:
        return experiments * n_cycles

    if experiment_order == ExperimentOrder.REVERSE:
        result: list[ExperimentConfig] = []
        for i in range(n_cycles):
            cycle = list(experiments) if i % 2 == 0 else list(reversed(experiments))
            result.extend(cycle)
        return result

    if experiment_order == ExperimentOrder.LATIN_SQUARE:
        return _williams_latin_square(experiments, n_cycles)

    # shuffle
    seed = shuffle_seed if shuffle_seed is not None else int(study_design_hash, 16) & 0xFFFFFFFF
    rng = random.Random(seed)
    result = []
    for _ in range(n_cycles):
        cycle = list(experiments)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result


def cycle_boundary_indices(
    n_unique: int,
    n_cycles: int,
    experiment_order: ExperimentOrder,
) -> frozenset[int]:
    """Return the sequence indices at which a cycle gap should fire.

    A cycle gap (``cycle_gap_seconds``) is the larger thermal-equalisation pause
    inserted between the major repeated units of an :func:`apply_cycles`
    execution sequence, distinct from the small ``experiment_gap_seconds`` that
    settles the machine between every consecutive pair. Where those boundaries
    fall depends on how the chosen ``experiment_order`` laid the sequence out,
    so the semantics live here next to the code that builds the sequence rather
    than as positional modulo math in the runner.

    Cycle gaps exist only when the sweep is actually repeated (``n_cycles >=
    2``); with a single cycle there is nothing to gap between, so the result is
    empty for every order.

    sequential ``[A, A, A, B, B, B]`` (with two or more distinct configs):
        The sequence is ``n_unique`` contiguous blocks, each block holding all
        ``n_cycles`` repetitions of one config. Consecutive identical
        repetitions inside a block are separated only by the small experiment
        gap; the larger cycle gap fires at the boundary between config blocks,
        i.e. at indices ``n_cycles, 2*n_cycles, ...``. This is the only coherent
        contiguous boundary in a sequential layout (the individual cycles are
        interleaved across the blocks, so a "between full passes" position does
        not exist), and it matches the thermal intent: a full cooldown when the
        schedule switches to a genuinely different config so each block starts
        from a comparable thermal floor.

    interleave / reverse / shuffle / latin_square ``[A, B, A, B, A, B]``:
        The sequence is ``n_cycles`` contiguous passes over the ``n_unique``
        configs. The cycle gap fires between full passes, i.e. at indices
        ``n_unique, 2*n_unique, ...``.

    The single-config case (``n_unique == 1``) has no distinct blocks to
    separate; there every repetition is itself a full cycle, so sequential
    falls back to the pass rule and behaves identically to interleave. The
    final boundary (the end of the sequence) is never included.
    """
    if n_unique <= 0 or n_cycles < 2:
        return frozenset()
    if experiment_order == ExperimentOrder.SEQUENTIAL and n_unique >= 2:
        block, n_blocks = n_cycles, n_unique
    else:
        block, n_blocks = n_unique, n_cycles
    return frozenset(block * k for k in range(1, n_blocks))


def _williams_latin_square(
    experiments: list[ExperimentConfig],
    n_cycles: int,
) -> list[ExperimentConfig]:
    """Generate a Williams balanced latin square ordering.

    A Williams design is a latin square where each condition follows every other
    condition exactly once across rows, balancing first-order carryover effects.
    When n_cycles > k (number of experiments), cycles repeat the square rows.
    When n_cycles < k, the first n_cycles rows are used.
    """
    k = len(experiments)
    if k == 0:
        return []

    # Build Williams square rows (works for both even and odd k)
    rows: list[list[int]] = []
    for i in range(k):
        row: list[int] = [0] * k
        for j in range(k):
            if j == 0:
                row[j] = i
            elif j % 2 == 1:
                row[j] = (i + (j + 1) // 2) % k
            else:
                row[j] = (i - j // 2) % k
        rows.append(row)

    result: list[ExperimentConfig] = []
    for cycle_idx in range(n_cycles):
        row = rows[cycle_idx % k]
        result.extend(experiments[idx] for idx in row)
    return result
