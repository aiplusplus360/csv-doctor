"""Non-destructive repair of messy tables.

Repair is split in two phases on purpose.  :func:`plan_repair` inspects a table
and returns a :class:`RepairPlan` describing every change it *would* make;
:func:`apply_repair` then executes that plan.  Nothing is ever written without a
plan you can print, diff or reject first.

>>> plan = plan_repair([" Name ", "Name", ""], [[" ada ", "1", ""], ["grace", "2", "x", "extra"]])
>>> plan.header
['name', 'name_2', 'column_3']
>>> plan.truncate_rows
0
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .dialect import Dialect, TableReader
from .values import (
    ColumnType,
    format_number,
    infer_column_type,
    is_null,
    normalize_date,
    parse_number,
    unique_column_names,
)

__all__ = [
    "RepairOptions",
    "RepairAction",
    "RepairPlan",
    "plan_repair",
    "apply_repair",
    "RepairResult",
]


@dataclass(frozen=True)
class RepairOptions:
    """Which repairs to perform.  Every flag defaults to off-safe behaviour."""

    trim_whitespace: bool = True
    normalize_header: bool = True
    unify_nulls: bool = False
    pad_short_rows: bool = True
    drop_extra_cells: bool = False
    drop_duplicate_rows: bool = False
    coerce_dates: bool = False
    coerce_numbers: bool = False
    strip_bom: bool = True
    delimiter: Optional[str] = None
    quotechar: Optional[str] = None
    line_terminator: Optional[str] = None
    encoding: str = "utf-8"
    collapse_whitespace: bool = False
    casefold_headers: bool = True

    def to_dict(self) -> dict:
        return {
            "trim_whitespace": self.trim_whitespace,
            "normalize_header": self.normalize_header,
            "unify_nulls": self.unify_nulls,
            "pad_short_rows": self.pad_short_rows,
            "drop_extra_cells": self.drop_extra_cells,
            "drop_duplicate_rows": self.drop_duplicate_rows,
            "coerce_dates": self.coerce_dates,
            "coerce_numbers": self.coerce_numbers,
            "strip_bom": self.strip_bom,
            "collapse_whitespace": self.collapse_whitespace,
        }


@dataclass
class RepairAction:
    """One class of change, with how many times it applied."""

    name: str
    count: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return {"action": self.name, "count": self.count, "detail": self.detail}


@dataclass
class RepairPlan:
    """A description of every change a repair run would make."""

    header: List[str] = field(default_factory=list)
    original_header: List[str] = field(default_factory=list)
    actions: List[RepairAction] = field(default_factory=list)
    pad_rows: int = 0
    truncate_rows: int = 0
    dropped_rows: int = 0
    null_values: int = 0
    coerced_dates: int = 0
    coerced_numbers: int = 0
    rewritten_dates: int = 0
    rewritten_numbers: int = 0
    notes: List[str] = field(default_factory=list)

    def action(self, name: str) -> RepairAction:
        for action in self.actions:
            if action.name == name:
                return action
        action = RepairAction(name)
        self.actions.append(action)
        return action

    @property
    def changed(self) -> bool:
        return any(action.count for action in self.actions)

    def to_dict(self) -> dict:
        return {
            "changed": self.changed,
            "header": self.header,
            "original_header": self.original_header,
            "pad_rows": self.pad_rows,
            "truncate_rows": self.truncate_rows,
            "dropped_rows": self.dropped_rows,
            "actions": [action.to_dict() for action in self.actions],
            "notes": self.notes,
        }


@dataclass
class RepairResult:
    """Outcome of :func:`apply_repair`."""

    rows_read: int = 0
    rows_written: int = 0
    output: Optional[str] = None
    plan: RepairPlan = field(default_factory=RepairPlan)

    def to_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "output": self.output,
            "plan": self.plan.to_dict(),
        }


def _slug(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        return f"{fallback}_{cleaned}"
    return cleaned


def _clean_header(
    header: Sequence[str], options: RepairOptions, actions: List[RepairAction]
) -> List[str]:
    original = list(header)
    if not options.normalize_header:
        cleaned = [cell.strip() for cell in original]
    elif options.casefold_headers:
        cleaned = unique_column_names(original)
    else:
        seen: Dict[str, int] = {}
        cleaned = []
        for index, cell in enumerate(original):
            base = _slug(cell, f"column_{index + 1}")
            count = seen.get(base, 0) + 1
            seen[base] = count
            cleaned.append(base if count == 1 else f"{base}_{count}")
    changed = sum(1 for a, b in zip(original, cleaned) if a != b) + abs(
        len(original) - len(cleaned)
    )
    if changed:
        actions.append(
            RepairAction("header", changed, f"{changed} column name(s) rewritten")
        )
    return cleaned


def plan_repair(
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    options: Optional[RepairOptions] = None,
    column_types: Optional[Dict[int, ColumnType]] = None,
) -> RepairPlan:
    """Inspect rows and describe the repairs that would be applied."""
    options = options or RepairOptions()
    original_header = list(header)
    actions: List[RepairAction] = []
    cleaned = _clean_header(original_header, options, actions)
    plan = RepairPlan(
        original_header=list(original_header),
        header=cleaned,
        actions=actions,
    )
    width = len(plan.header)

    types = column_types or {}
    seen = set()
    action_details = {
        "pad_short_rows": "rows with fewer cells than the header were padded",
        "drop_extra_cells": "trailing cells beyond the header width were dropped",
        "unify_nulls": "null spellings were replaced with an empty cell",
        "coerce_dates": "non-ISO dates were rewritten as ISO 8601",
        "coerce_numbers": "currency symbols and separators were removed",
        "drop_duplicate_rows": "rows identical after normalisation were dropped",
    }

    def bump(name: str) -> None:
        action = plan.action(name)
        action.count += 1
        action.detail = action.detail or action_details.get(name, "")

    for fields in rows:
        if len(fields) < width:
            if options.pad_short_rows:
                plan.pad_rows += 1
                bump("pad_short_rows")
        elif len(fields) > width:
            if options.drop_extra_cells:
                plan.truncate_rows += 1
                bump("drop_extra_cells")

        normalized: List[str] = []
        for index in range(width):
            token = fields[index] if index < len(fields) else ""
            value = token
            if options.trim_whitespace:
                value = value.strip()
            if options.collapse_whitespace:
                value = " ".join(value.split())
            if options.unify_nulls and is_null(value):
                if value != "":
                    plan.null_values += 1
                    bump("unify_nulls")
                value = ""

            column_type = types.get(index)
            if column_type is not None:
                if options.coerce_dates and column_type.kind in ("date", "datetime"):
                    if column_type.dates.format and value:
                        iso = normalize_date(value, column_type.dates.format)
                        if iso is not None and iso != value:
                            plan.rewritten_dates += 1
                            bump("coerce_dates")
                            value = iso
                        elif iso is not None:
                            plan.coerced_dates += 1
                elif options.coerce_numbers and column_type.kind in ("integer", "decimal"):
                    if column_type.number_style and value and not is_null(value):
                        number = parse_number(value, style=column_type.number_style)
                        if number is not None:
                            rendered = format_number(number)
                            if rendered != value:
                                plan.rewritten_numbers += 1
                                bump("coerce_numbers")
                                value = rendered
                            else:
                                plan.coerced_numbers += 1
            normalized.append(value)

        if options.drop_duplicate_rows:
            key = tuple(normalized)
            if key in seen:
                plan.dropped_rows += 1
                bump("drop_duplicate_rows")
                continue
            seen.add(key)

    if plan.pad_rows:
        plan.notes.append(
            f"{plan.pad_rows} short row(s) padded with empty cells"
        )
    if plan.truncate_rows:
        plan.notes.append(
            f"{plan.truncate_rows} row(s) had trailing cells dropped"
        )
    if plan.dropped_rows:
        plan.notes.append(f"{plan.dropped_rows} exact duplicate row(s) dropped")
    return plan


def apply_repair(
    path: str,
    output: str,
    *,
    options: Optional[RepairOptions] = None,
    dialect: Optional[Dialect] = None,
    column_types: Optional[Dict[int, ColumnType]] = None,
    dry_run: bool = False,
) -> RepairResult:
    """Rewrite ``path`` into ``output`` applying ``options``.

    >>> import tempfile, pathlib
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> source = folder / "in.csv"
    >>> _ = source.write_text(" Name ,age\\n ada , 30 \\n", encoding="utf-8")
    >>> target = folder / "out.csv"
    >>> result = apply_repair(str(source), str(target))
    >>> target.read_text(encoding="utf-8").strip()
    'name,age\\nada,30'
    """
    options = options or RepairOptions()
    with TableReader(path, dialect) as table:
        active = Dialect(
            layout_delimiter(table.dialect, options),
            layout_quotechar(table.dialect, options),
            options.encoding if options.strip_bom else table.dialect.encoding,
            options.line_terminator or table.dialect.lineterminator,
            table.dialect.has_header,
        )
        header = list(table.header)
        raw_rows = list(table.rows())

    plan = plan_repair(
        header, [fields for _, fields in raw_rows], options=options, column_types=column_types
    )
    result = RepairResult(rows_read=len(raw_rows), plan=plan, output=None if dry_run else output)
    if dry_run:
        return result

    width = len(plan.header)
    types = column_types or {}
    directory = os.path.dirname(os.path.abspath(output))
    if directory:
        os.makedirs(directory, exist_ok=True)

    seen = set()
    buffer = io.StringIO()
    writer = csv.writer(
        buffer,
        delimiter=active.delimiter,
        quotechar=active.quotechar,
        lineterminator=active.lineterminator,
    )
    writer.writerow(plan.header)

    with open(output, "w", encoding=options.encoding, newline="") as handle:
        written = 0
        for _, fields in raw_rows:
            row = _normalize_row(fields, width, options, types)
            if options.drop_duplicate_rows:
                key = tuple(row)
                if key in seen:
                    continue
                seen.add(key)
            writer.writerow(row)
            written += 1
            if buffer.tell() > 1 << 20:
                handle.write(buffer.getvalue())
                buffer.seek(0)
                buffer.truncate()
        handle.write(buffer.getvalue())
    result.rows_written = written
    return result


def layout_delimiter(found: Dialect, options: RepairOptions) -> str:
    return options.delimiter or found.delimiter


def layout_quotechar(found: Dialect, options: RepairOptions) -> str:
    return options.quotechar or found.quotechar


def _normalize_row(
    fields: Sequence[str],
    width: int,
    options: RepairOptions,
    types: Dict[int, ColumnType],
) -> List[str]:
    row: List[str] = []
    for index in range(width):
        value = fields[index] if index < len(fields) else ""
        if options.trim_whitespace:
            value = value.strip()
        if options.collapse_whitespace:
            value = " ".join(value.split())
        if options.unify_nulls and is_null(value):
            value = ""

        column_type = types.get(index)
        if column_type is not None and value:
            if options.coerce_dates and column_type.kind in ("date", "datetime"):
                if column_type.dates.format:
                    iso = normalize_date(value, column_type.dates.format)
                    if iso is not None:
                        value = iso
            elif options.coerce_numbers and column_type.kind in ("integer", "decimal"):
                number = parse_number(value, style=column_type.number_style)
                if number is not None:
                    value = format_number(number)
        row.append(value)
    return row


def infer_column_types(
    header: Sequence[str], rows: Iterable[Sequence[str]]
) -> Dict[int, ColumnType]:
    """Infer a type for every column from materialised rows."""
    columns: List[List[str]] = [[] for _ in header]
    for fields in rows:
        for index in range(len(header)):
            columns[index].append(fields[index] if index < len(fields) else "")
    return {
        index: infer_column_type(values) for index, values in enumerate(columns)
    }
