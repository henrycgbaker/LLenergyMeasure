"""Extract validator-bearing source regions from an engine file, chunked to
fit a small-LLM context window.

We do NOT feed whole 3500-line files. Instead we pull the function / method
bodies that actually carry validation logic (a body is "validator-bearing"
if it contains any of: raise, isinstance, minor_issues, warnings.warn,
logger.warning, ' not in ', a comparison guard, or a @field_validator /
@model_validator / @validator decorator). Each emitted chunk is a contiguous
slice of <= ~max_chars characters, packed greedily from those bodies in
source order. This keeps within qwen2.5-coder-7b's 32k token budget while
covering the regions where invariants live.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TRIGGER_SUBSTR = (
    "raise ",
    "isinstance(",
    "minor_issues",
    "warnings.warn",
    "logger.warning",
    "logger.warning_once",
    " not in ",
    " in {",
    "field_validator",
    "model_validator",
    "@validator",
    "_verify_args",
    "_VERIFY",
)


def _is_validator_body(src: str) -> bool:
    return any(t in src for t in _TRIGGER_SUBSTR)


def _node_segment(lines: list[str], node: ast.AST) -> tuple[int, int, str]:
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    seg = "\n".join(lines[start:end])
    return start, end, seg


def validator_bodies(path: Path) -> list[tuple[str, str]]:
    """Return [(qualname, source_segment)] for every function/method whose body
    looks validator-bearing, in source order.
    """
    text = path.read_text()
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[str, str]] = []

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def _handle_fn(self, node: ast.AST) -> None:
            _s, _e, seg = _node_segment(lines, node)
            if _is_validator_body(seg):
                qual = ".".join([*self.stack, node.name])  # type: ignore[attr-defined]
                out.append((qual, seg))

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self._handle_fn(node)
            # do not descend into nested fns for separate emission
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node) -> None:  # type: ignore[no-untyped-def]
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

    V().visit(tree)
    return out


def chunk_validator_source(files: list[Path], *, max_chars: int = 22_000) -> list[str]:
    """Greedily pack validator bodies (across the given files, in order) into
    chunks of <= max_chars. ~48k chars ~= 12-16k tokens of source, leaving
    room for the floor catalogue + prompt within a 32k window.
    """
    bodies: list[tuple[str, str]] = []
    for f in files:
        if not f.exists():
            continue
        rel = f.name
        for qual, seg in validator_bodies(f):
            header = f"# ===== file={rel}  qualname={qual} =====\n"
            bodies.append((rel, header + seg))

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for _rel, seg in bodies:
        seg_len = len(seg) + 2
        if size + seg_len > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            size = 0
        # a single body bigger than max_chars: hard-truncate it
        if seg_len > max_chars:
            seg = seg[: max_chars - 200] + "\n# ...[truncated]..."
            seg_len = len(seg) + 2
        buf.append(seg)
        size += seg_len
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# Per (engine, version_slug) -> ordered list of validator files to mine.
def source_files_for(engine: str, version_slug: str) -> list[Path]:
    base = Path(f"/tmp/trial_{engine}_{version_slug}_venv/src")
    if engine == "transformers":
        root = base / "transformers"
        return [
            root / "generation" / "configuration_utils.py",
            root / "utils" / "quantization_config.py",
            root / "modeling_utils.py",
        ]
    if engine == "vllm":
        root = base / "vllm"
        cands = [root / "sampling_params.py"]
        if (root / "config.py").exists():
            cands.append(root / "config.py")
        cfg_pkg = root / "config"
        if cfg_pkg.is_dir():
            # the dataclass-config files most likely to carry validators
            for name in (
                "model.py",
                "cache.py",
                "scheduler.py",
                "parallel.py",
                "lora.py",
                "speculative.py",
                "structured_outputs.py",
                "vllm.py",
            ):
                p = cfg_pkg / name
                if p.exists():
                    cands.append(p)
        if (root / "engine" / "arg_utils.py").exists():
            cands.append(root / "engine" / "arg_utils.py")
        return cands
    if engine == "tensorrt":
        root = base / "tensorrt_llm"
        cands = [
            root / "llmapi" / "llm_args.py",
            root / "sampling_params.py",
            root / "plugin" / "plugin.py",  # PluginConfig - the wave-4 source-coverage gap
        ]
        return [p for p in cands if p.exists()]
    return []


if __name__ == "__main__":
    import sys

    e, v = sys.argv[1], sys.argv[2]
    files = source_files_for(e, v)
    chunks = chunk_validator_source(files)
    print(f"{e}/{v}: {len(files)} files -> {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {len(c)} chars")
