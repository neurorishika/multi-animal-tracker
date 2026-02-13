#!/usr/bin/env python3
"""Documentation quality audit for src/multi_tracker.

Checks:
- Module docstring coverage
- Public symbol docstring coverage (classes/functions/methods)
- Public function/method full type annotation coverage
- Optional baseline non-regression
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Metrics:
    module_total: int = 0
    module_doc: int = 0
    symbol_total: int = 0
    symbol_doc: int = 0
    function_total: int = 0
    function_fully_typed: int = 0

    @property
    def module_doc_pct(self) -> float:
        return (
            100.0 * self.module_doc / self.module_total if self.module_total else 100.0
        )

    @property
    def symbol_doc_pct(self) -> float:
        return (
            100.0 * self.symbol_doc / self.symbol_total if self.symbol_total else 100.0
        )

    @property
    def function_typed_pct(self) -> float:
        return (
            100.0 * self.function_fully_typed / self.function_total
            if self.function_total
            else 100.0
        )


def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def function_fully_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    total = 0
    annotated = 0
    args = node.args

    for arg in args.posonlyargs + args.args + args.kwonlyargs:
        if arg.arg in {"self", "cls"}:
            continue
        total += 1
        annotated += arg.annotation is not None

    if args.vararg is not None:
        total += 1
        annotated += args.vararg.annotation is not None

    if args.kwarg is not None:
        total += 1
        annotated += args.kwarg.annotation is not None

    total += 1
    annotated += node.returns is not None
    return total == 0 or annotated == total


def collect_symbol_metrics(
    body: list[ast.stmt],
    metrics: Metrics,
    missing_doc: list[str],
    missing_typed: list[str],
    module_path: str,
    scope: str = "",
) -> None:
    for node in body:
        if isinstance(node, ast.ClassDef):
            fq = f"{scope}.{node.name}" if scope else node.name
            if not node.name.startswith("_"):
                metrics.symbol_total += 1
                has_doc = bool(ast.get_docstring(node))
                metrics.symbol_doc += int(has_doc)
                if not has_doc:
                    missing_doc.append(f"{module_path}:{node.lineno} class {fq}")
            collect_symbol_metrics(
                node.body, metrics, missing_doc, missing_typed, module_path, fq
            )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fq = f"{scope}.{node.name}" if scope else node.name
            if not node.name.startswith("_"):
                metrics.symbol_total += 1
                has_doc = bool(ast.get_docstring(node))
                metrics.symbol_doc += int(has_doc)
                if not has_doc:
                    missing_doc.append(f"{module_path}:{node.lineno} function {fq}")

                metrics.function_total += 1
                typed = function_fully_typed(node)
                metrics.function_fully_typed += int(typed)
                if not typed:
                    missing_typed.append(f"{module_path}:{node.lineno} function {fq}")

            collect_symbol_metrics(
                node.body, metrics, missing_doc, missing_typed, module_path, scope
            )


def compute_metrics(src_root: Path) -> tuple[Metrics, list[str], list[str], list[str]]:
    metrics = Metrics()
    missing_module_doc: list[str] = []
    missing_symbol_doc: list[str] = []
    missing_typed: list[str] = []

    for py_file in iter_py_files(src_root):
        rel = py_file.as_posix()
        metrics.module_total += 1

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: failed parsing {rel}: {exc}", file=sys.stderr)
            continue

        has_mod_doc = bool(ast.get_docstring(tree))
        metrics.module_doc += int(has_mod_doc)
        if not has_mod_doc:
            missing_module_doc.append(rel)

        collect_symbol_metrics(
            tree.body,
            metrics,
            missing_symbol_doc,
            missing_typed,
            rel,
        )

    return metrics, missing_module_doc, missing_symbol_doc, missing_typed


def rounded(metrics: Metrics) -> dict[str, float | int]:
    return {
        "module_total": metrics.module_total,
        "module_doc": metrics.module_doc,
        "module_doc_pct": round(metrics.module_doc_pct, 2),
        "symbol_total": metrics.symbol_total,
        "symbol_doc": metrics.symbol_doc,
        "symbol_doc_pct": round(metrics.symbol_doc_pct, 2),
        "function_total": metrics.function_total,
        "function_fully_typed": metrics.function_fully_typed,
        "function_typed_pct": round(metrics.function_typed_pct, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit documentation quality in code")
    parser.add_argument("--src", default="src/multi_tracker", help="source root")
    parser.add_argument("--baseline", default="docs/doc-quality-baseline.json")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--min-module-doc", type=float, default=90.0)
    parser.add_argument("--min-symbol-doc", type=float, default=55.0)
    parser.add_argument("--min-typed-func", type=float, default=20.0)
    parser.add_argument("--max-report-items", type=int, default=20)
    args = parser.parse_args()

    src_root = Path(args.src)
    metrics, missing_mod, missing_sym, missing_typed = compute_metrics(src_root)
    summary = rounded(metrics)

    print("Documentation Quality Report")
    print(
        "- module doc coverage: {module_doc_pct}% ({module_doc}/{module_total})".format(
            **summary
        )
    )
    print(
        "- symbol doc coverage: {symbol_doc_pct}% ({symbol_doc}/{symbol_total})".format(
            **summary
        )
    )
    print(
        "- function full typing coverage: {function_typed_pct}% ({function_fully_typed}/{function_total})".format(
            **summary
        )
    )

    baseline_path = Path(args.baseline)
    failed = False

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote baseline: {baseline_path}")

    if baseline_path.exists() and not args.write_baseline:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        regressions: list[str] = []
        for key in ("module_doc_pct", "symbol_doc_pct", "function_typed_pct"):
            if summary[key] + 1e-9 < baseline.get(key, 0.0):
                regressions.append(
                    f"{key}: {summary[key]} < baseline {baseline.get(key)}"
                )
        if regressions:
            failed = True
            print("\nRegression detected vs baseline:")
            for r in regressions:
                print(f"- {r}")

    threshold_failures = []
    if summary["module_doc_pct"] < args.min_module_doc:
        threshold_failures.append(
            f"module_doc_pct {summary['module_doc_pct']} < {args.min_module_doc}"
        )
    if summary["symbol_doc_pct"] < args.min_symbol_doc:
        threshold_failures.append(
            f"symbol_doc_pct {summary['symbol_doc_pct']} < {args.min_symbol_doc}"
        )
    if summary["function_typed_pct"] < args.min_typed_func:
        threshold_failures.append(
            f"function_typed_pct {summary['function_typed_pct']} < {args.min_typed_func}"
        )

    if threshold_failures:
        failed = True
        print("\nThreshold check failed:")
        for item in threshold_failures:
            print(f"- {item}")

    max_items = max(0, args.max_report_items)
    if missing_mod[:max_items]:
        print("\nMissing module docstrings (sample):")
        for item in missing_mod[:max_items]:
            print(f"- {item}")

    if missing_sym[:max_items]:
        print("\nMissing public symbol docstrings (sample):")
        for item in missing_sym[:max_items]:
            print(f"- {item}")

    if missing_typed[:max_items]:
        print("\nPublic functions missing full type annotations (sample):")
        for item in missing_typed[:max_items]:
            print(f"- {item}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
