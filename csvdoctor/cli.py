"""Command line interface for csv-doctor.

Four verbs, one mental model:

``profile``  what is in this file and what is wrong with it
``validate`` does it satisfy my rules
``repair``   rewrite it, with a plan first
``sniff``    just tell me the dialect

Every verb supports ``--json`` for machine consumption and uses exit codes
``0`` clean, ``1`` warnings, ``2`` errors / usage problems, so it drops straight
into CI without a wrapper script.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from .checks import check_profile, exit_code
from .dialect import Dialect, MalformedCsv, TableReader, load_table, read_text, sniff
from .profile import ProfileOptions, profile_table
from .repair import RepairOptions, apply_repair, infer_column_types, plan_repair
from .rules import RuleError, load_rules, validate_table

PROG = "csv-doctor"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed for tests and shell completion)."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Diagnose and repair messy CSV files.",
        epilog="exit codes: 0 clean, 1 warnings, 2 errors or usage problems",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("path", help="CSV file to inspect")
    common.add_argument("--delimiter", help="override delimiter detection")
    common.add_argument("--quotechar", default='"', help="quote character (default: \")")
    common.add_argument(
        "--no-header", action="store_true", help="treat the first row as data"
    )
    common.add_argument("--json", action="store_true", help="emit JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", parents=[common], help="describe the file")
    profile.add_argument("--max-distinct", type=int, default=10000)
    profile.add_argument("--sample-size", type=int, default=2000)
    profile.add_argument("--max-rows", type=int, default=None)
    profile.add_argument("--top", type=int, default=5, help="top values per column")
    profile.add_argument(
        "--strict", action="store_true", help="treat warnings as errors"
    )
    profile.add_argument(
        "--quiet", action="store_true", help="print nothing unless something is wrong"
    )

    validate = sub.add_parser("validate", parents=[common], help="check against rules")
    validate.add_argument("--rules", required=True, help="JSON rule file")
    validate.add_argument("--max-findings", type=int, default=None)
    validate.add_argument(
        "--fail-on",
        choices=["error", "warning"],
        default="error",
        help="lowest severity that makes the run fail (default: error)",
    )

    repair = sub.add_parser("repair", parents=[common], help="rewrite the file")
    repair.add_argument("--out", required=True, help="path to write")
    repair.add_argument("--dry-run", action="store_true", help="plan only")
    repair.add_argument("--trim", dest="trim", action="store_true", default=True)
    repair.add_argument("--no-trim", dest="trim", action="store_false")
    repair.add_argument("--collapse-whitespace", action="store_true")
    repair.add_argument("--nulls", action="store_true", help="unify null tokens to empty")
    repair.add_argument(
        "--pad-short-rows", action="store_true", default=True, help="(default)"
    )
    repair.add_argument(
        "--drop-extra-cells", action="store_true", help="truncate long rows"
    )
    repair.add_argument("--drop-duplicates", action="store_true")
    repair.add_argument("--dates", action="store_true", help="normalise dates to ISO 8601")
    repair.add_argument("--numbers", action="store_true", help="normalise numbers")
    repair.add_argument("--encoding", default="utf-8")
    repair.add_argument("--line-terminator", choices=["lf", "crlf"], default=None)
    repair.add_argument("--keep-header", action="store_true", help="do not rename columns")

    sniff_cmd = sub.add_parser("sniff", parents=[common], help="report the dialect")
    sniff_cmd.set_defaults(path_required=True)

    return parser


def _dialect(args) -> Dialect:
    return Dialect(
        delimiter=args.delimiter or ",",
        quotechar=args.quotechar,
        has_header=not args.no_header,
    )


def _read_table(args):
    return load_table(
        args.path,
        None
        if args.delimiter is None or args.no_header
        else Dialect(args.delimiter, args.quotechar, has_header=True),
    )


def cmd_profile(args) -> int:
    try:
        dialect, header, rows, lines = _read_table(args)
    except MalformedCsv as exc:
        return _fail(str(exc), args.json)

    options = ProfileOptions(
        sample_size=args.sample_size,
        max_distinct=args.max_distinct,
        max_rows=args.max_rows,
        top_values=args.top,
    )
    if args.no_header:
        options = options
    profile = profile_table(header, rows, lines, options=options, dialect=dialect.to_dict())
    if args.no_header:
        profile.column_count = len(header)
    recommendations = check_profile(profile)
    worst = "info"
    for item in recommendations:
        if item.severity == "error":
            worst = "error"
            break
        if item.severity == "warning":
            worst = "warning"

    if args.json:
        payload = profile.to_dict()
        payload["recommendations"] = [item.to_dict() for item in recommendations]
        print(json.dumps(payload, indent=2, sort_keys=False))
    elif not args.quiet or worst in ("warning", "error"):
        print(_render_profile(profile, recommendations, dialect))

    code = exit_code(worst)
    if args.strict and code == 1:
        return 2
    return code


def _render_profile(profile, recommendations, dialect) -> str:
    width = max([len(column.name) for column in profile.columns] + [7])
    lines: List[str] = []
    lines.append(f"{profile.row_count} rows x {profile.column_count} columns")
    layout = dialect.to_dict()
    lines.append(
        "delimiter {!r}  quote {!r}  encoding {}  header {}".format(
            layout["delimiter"],
            layout["quotechar"],
            layout["encoding"],
            "yes" if layout["has_header"] else "no",
        )
    )
    lines.append("")
    lines.append(
        f"{'column'.ljust(width)}  {'kind':<9} {'nulls':>7} {'distinct':>8}  notes"
    )
    lines.append("-" * (width + 32))
    for column in profile.columns:
        notes = []
        if column.date_ambiguous:
            notes.append("ambiguous dates")
        elif column.date_format:
            notes.append(column.date_format)
        if column.numeric and column.numeric.median is not None:
            notes.append(
                f"median {_num(column.numeric.median)} range "
                f"{_num(column.numeric.minimum)}..{_num(column.numeric.maximum)}"
            )
        if column.outliers:
            notes.append(f"{len(column.outliers)} outliers")
        if column.distinct_truncated:
            notes.append("distinct truncated")
        nulls = f"{column.nulls} ({column.null_rate:.0%})" if column.nulls else "0"
        distinct = f"{column.distinct}+" if column.distinct_truncated else str(column.distinct)
        lines.append(
            f"{column.name.ljust(width)}  {column.kind:<9} {nulls:>7} {distinct:>8}  "
            + ", ".join(notes)
        )

    if profile.warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"  - {warning}" for warning in profile.warnings)

    if recommendations:
        lines.append("")
        lines.append("recommendations:")
        for item in recommendations:
            marker = {"error": "x", "warning": "!", "info": "-"}[item.severity]
            lines.append(f"  {marker} [{item.code}] {item.message}")
            if item.fix:
                lines.append(f"      fix: {item.fix}")
    return "\n".join(lines)


def _num(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return str(int(value)) if value == int(value) else f"{value:g}"


def cmd_validate(args) -> int:
    try:
        spec = load_rules(args.rules)
    except RuleError as exc:
        return _fail(str(exc), args.json)

    try:
        header, findings, rows = validate_table(
            args.path, spec, dialect=None, max_findings=args.max_findings
        )
    except MalformedCsv as exc:
        return _fail(str(exc), args.json)

    errors = sum(1 for finding in findings if finding.severity == "error")
    warnings = sum(1 for finding in findings if finding.severity == "warning")

    if args.json:
        print(
            json.dumps(
                {
                    "file": args.path,
                    "rules": spec.name or args.rules,
                    "rows": rows,
                    "columns": header,
                    "errors": errors,
                    "warnings": warnings,
                    "findings": [finding.to_dict() for finding in findings],
                },
                indent=2,
            )
        )
    elif findings:
        for finding in findings:
            where = f"row {finding.row}" if finding.row is not None else "file"
            column = f" [{finding.column}]" if finding.column else ""
            print(f"{finding.severity}: {where}{column} {finding.message}")
        print(f"\n{errors} error(s), {warnings} warning(s), {rows} row(s) checked")
    else:
        print(f"ok: {rows} row(s) satisfy {spec.name or args.rules}")

    if errors:
        return 2
    if warnings and args.fail_on == "warning":
        return 1
    return 0


def _repair_options(args) -> RepairOptions:
    options = RepairOptions(
        trim_whitespace=args.trim,
        normalize_header=not args.keep_header,
        unify_nulls=args.nulls,
        pad_short_rows=args.pad_short_rows,
        drop_extra_cells=args.drop_extra_cells,
        drop_duplicate_rows=args.drop_duplicates,
        coerce_dates=args.dates,
        coerce_numbers=args.numbers,
        encoding=args.encoding,
        collapse_whitespace=args.collapse_whitespace,
    )
    return options


def cmd_repair(args) -> int:
    try:
        dialect, header, rows, lines = load_table(
            args.path,
            None
            if args.delimiter is None or args.no_header
            else Dialect(args.delimiter, args.quotechar, has_header=True),
        )
    except MalformedCsv as exc:
        return _fail(str(exc), args.json)

    options = _repair_options(args)
    if args.line_terminator:
        options = RepairOptions(
            **{
                **options.__dict__,
                "line_terminator": "\r\n" if args.line_terminator == "crlf" else "\n",
            }
        )
    types = infer_column_types(header, rows)
    plan = plan_repair(header, rows, options=options, column_types=types)

    result = apply_repair(
        args.path,
        args.out,
        options=options,
        dialect=None,
        column_types=types,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"{'would repair' if args.dry_run else 'repaired'}: {args.path}")
        print(f"  rows read:    {result.rows_read}")
        if args.dry_run:
            print("  rows to write: " + str(result.rows_read - plan.dropped_rows))
        else:
            print(f"  rows written: {result.rows_written}")
            print(f"  output:       {args.out}")
        if plan.header != plan.original_header:
            print(f"  header:       {plan.header}")
        for action in plan.actions:
            if action.count:
                print(f"  {action.name}: {action.count} ({action.detail})")
        for note in plan.notes:
            print(f"  note: {note}")
        if not plan.changed:
            print("  nothing to change")
    return 0


def cmd_sniff(args) -> int:
    try:
        text, encoding, had_bom = read_text(args.path)
    except (MalformedCsv, OSError) as exc:
        return _fail(str(exc), args.json)
    found = sniff(text[:65536])
    found = Dialect(
        found.delimiter,
        found.quotechar,
        encoding,
        found.lineterminator,
        found.has_header and not args.no_header,
    )
    if args.json:
        print(json.dumps({"file": args.path, **found.to_dict(), "bom": had_bom}, indent=2))
    else:
        layout = found.to_dict()
        print(f"delimiter:   {layout['delimiter']!r}")
        print(f"quotechar:   {layout['quotechar']!r}")
        print(f"encoding:    {layout['encoding']}")
        print(f"line ending: {layout['line_terminator']}")
        print(f"header:      {'yes' if layout['has_header'] else 'no'}")
        print(f"bom:         {'yes' if had_bom else 'no'}")
    return 0


def _fail(message: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps({"error": message}), file=sys.stderr)
    else:
        print(f"error: {message}", file=sys.stderr)
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``csv-doctor`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "profile": cmd_profile,
        "validate": cmd_validate,
        "repair": cmd_repair,
        "sniff": cmd_sniff,
    }
    handler = handlers[args.command]
    try:
        return handler(args)
    except BrokenPipeError:
        return 0
    except MalformedCsv as exc:
        return _fail(str(exc), getattr(args, "json", False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
