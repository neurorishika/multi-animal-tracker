#!/usr/bin/env python3
"""Add conservative return annotations to common Qt methods.

This codemod is intentionally narrow:
- only class methods are considered
- only methods with known Qt event names plus ``__init__`` are annotated
- only single-line ``def ...:`` signatures are rewritten

Run without ``--apply`` to preview edits.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGETS = (
    "src/hydra_suite/classkit/gui",
    "src/hydra_suite/detectkit/gui",
    "src/hydra_suite/filterkit/gui",
    "src/hydra_suite/posekit/gui",
    "src/hydra_suite/refinekit/gui",
    "src/hydra_suite/trackerkit/gui",
    "src/hydra_suite/widgets",
    "src/hydra_suite/launcher/app.py",
)


KNOWN_RETURNS = {
    "__init__": "None",
    "changeEvent": "None",
    "closeEvent": "None",
    "contextMenuEvent": "None",
    "dragEnterEvent": "None",
    "dragLeaveEvent": "None",
    "dragMoveEvent": "None",
    "dropEvent": "None",
    "enterEvent": "None",
    "focusInEvent": "None",
    "focusOutEvent": "None",
    "hideEvent": "None",
    "inputMethodEvent": "None",
    "keyPressEvent": "None",
    "keyReleaseEvent": "None",
    "leaveEvent": "None",
    "mouseDoubleClickEvent": "None",
    "mouseMoveEvent": "None",
    "mousePressEvent": "None",
    "mouseReleaseEvent": "None",
    "paintEvent": "None",
    "resizeEvent": "None",
    "showEvent": "None",
    "tabletEvent": "None",
    "wheelEvent": "None",
    "eventFilter": "bool",
}


@dataclass(frozen=True)
class Edit:
    line_no: int
    method_name: str
    return_type: str
    old_line: str
    new_line: str


class ClassMethodCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.class_depth = 0
        self.methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.class_depth:
            self.methods.append(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self.class_depth:
            self.methods.append(node)
        self.generic_visit(node)


def parse_method_overrides(raw_values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw_value in raw_values:
        if ":" not in raw_value:
            raise SystemExit(
                f"Invalid --method override: {raw_value!r}; expected name:type"
            )
        name, return_type = raw_value.split(":", 1)
        name = name.strip()
        return_type = return_type.strip()
        if not name or not return_type:
            raise SystemExit(
                f"Invalid --method override: {raw_value!r}; expected name:type"
            )
        overrides[name] = return_type
    return overrides


def iter_python_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.is_file() and path.suffix == ".py":
            files.append(path)
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for path in files:
        if path not in seen:
            unique_files.append(path)
            seen.add(path)
    return unique_files


def make_signature_pattern(method_name: str) -> re.Pattern[str]:
    return re.compile(
        rf"^(?P<indent>\s*)(?P<async>async\s+)?def\s+{re.escape(method_name)}\s*\((?P<params>[^)]*)\)"
        rf"(?:\s*->\s*(?P<return_annotation>[^:]+))?(?P<suffix>\s*:\s*(?:#.*)?)$"
    )


def _is_object_annotation(annotation: ast.expr | None) -> bool:
    return isinstance(annotation, ast.Name) and annotation.id == "object"


def _strip_object_annotations(
    params: str, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    annotated_names = {
        arg.arg
        for arg in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        if _is_object_annotation(arg.annotation)
    }
    if node.args.vararg is not None and _is_object_annotation(
        node.args.vararg.annotation
    ):
        annotated_names.add(node.args.vararg.arg)
    if node.args.kwarg is not None and _is_object_annotation(
        node.args.kwarg.annotation
    ):
        annotated_names.add(node.args.kwarg.arg)

    updated = params
    for name in sorted(annotated_names, key=len, reverse=True):
        updated = re.sub(rf"(?<!\w)({re.escape(name)})\s*:\s*object\b", r"\1", updated)
    return updated


def collect_edits(path: Path, method_returns: dict[str, str]) -> list[Edit]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    collector = ClassMethodCollector()
    collector.visit(tree)
    lines = source.splitlines()
    edits: list[Edit] = []
    seen_lines: set[int] = set()

    for node in collector.methods:
        return_type = method_returns.get(node.name)
        if return_type is None:
            continue

        line_no = node.lineno
        if line_no in seen_lines or line_no > len(lines):
            continue

        old_line = lines[line_no - 1]

        pattern = make_signature_pattern(node.name)
        match = pattern.match(old_line)
        if not match:
            continue

        current_return = match.group("return_annotation")
        current_return = current_return.strip() if current_return is not None else None
        normalized_params = _strip_object_annotations(match.group("params"), node)
        expected_return = return_type
        return_is_correct = current_return == expected_return
        params_are_unchanged = normalized_params == match.group("params")
        if return_is_correct and params_are_unchanged:
            continue

        async_prefix = match.group("async") or ""
        new_line = (
            f"{match.group('indent')}{async_prefix}def {node.name}"
            f"({normalized_params}) -> {expected_return}{match.group('suffix')}"
        )
        if new_line == old_line:
            continue

        edits.append(
            Edit(
                line_no=line_no,
                method_name=node.name,
                return_type=return_type,
                old_line=old_line,
                new_line=new_line,
            )
        )
        seen_lines.add(line_no)

    return edits


def apply_edits(path: Path, edits: list[Edit]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for edit in edits:
        lines[edit.line_no - 1] = edit.new_line
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="Files or directories to scan")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write edits in place instead of printing a preview",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME:TYPE",
        help="Add or override a method return mapping",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    method_returns = dict(KNOWN_RETURNS)
    method_returns.update(parse_method_overrides(args.method))

    targets = args.targets or list(DEFAULT_TARGETS)
    files = iter_python_files(targets)

    total_edits = 0
    changed_files = 0
    for path in files:
        edits = collect_edits(path, method_returns)
        if not edits:
            continue
        changed_files += 1
        total_edits += len(edits)
        print(f"{path}:")
        for edit in edits:
            print(
                f"  L{edit.line_no}: {edit.method_name} -> {edit.return_type}",
                file=sys.stdout,
            )
        if args.apply:
            apply_edits(path, edits)

    print(
        f"Processed {len(files)} file(s); "
        f"{changed_files} file(s) would change; {total_edits} edit(s) total."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
