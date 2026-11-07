"""Turn a profile into advice: the "so what?" layer of csv-doctor.

Each check inspects a :class:`~csvdoctor.profile.TableProfile` and emits a
:class:`Recommendation` with a severity, a concrete fix and the exact CLI flag
that applies it.  Nothing here reads the file again.

>>> from csvdoctor.profile import profile_table
>>> profile = profile_table(["Name", "Name"], [["a", "1"], ["b", "2"]])
>>> [r.code for r in check_profile(profile) if r.severity == "error"]
['duplicate_headers']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .profile import ColumnProfile, TableProfile

__all__ = ["Recommendation", "check_profile", "SEVERITY_ORDER", "exit_code"]

SEVERITY_ORDER = {"error": 3, "warning": 2, "info": 1}


@dataclass(frozen=True)
class Recommendation:
    """One actionable observation about a table."""

    code: str
    severity: str
    message: str
    column: Optional[str] = None
    fix: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "column": self.column,
            "message": self.message,
            "fix": self.fix,
        }


def exit_code(severity: str) -> int:
    """Map a severity to a shell exit code.

    >>> exit_code("info"), exit_code("warning"), exit_code("error")
    (0, 1, 2)
    """
    return {"info": 0, "warning": 1, "error": 2}.get(severity, 2)


def _add(
    found: List[Recommendation],
    code: str,
    severity: str,
    message: str,
    column: Optional[str] = None,
    fix: str = "",
) -> None:
    found.append(Recommendation(code, severity, message, column, fix))


def check_profile(
    profile: TableProfile, *, null_warn: float = 0.2, null_error: float = 0.8
) -> List[Recommendation]:
    """Return recommendations sorted by severity then column name."""
    found: List[Recommendation] = []

    if profile.column_count == 0:
        _add(found, "empty_file", "error", "the file has no header row")
        return found

    if not profile.row_count:
        _add(found, "no_rows", "warning", "the header parsed but there are no data rows")

    names = [column.name for column in profile.columns]
    seen: dict = {}
    duplicates = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        if seen[name] == 2:
            duplicates.append(name)
    for name in duplicates:
        _add(
            found,
            "duplicate_headers",
            "error",
            f"column name {name!r} appears more than once",
            name,
            "csv-doctor repair --out fixed.csv",
        )

    if profile.ragged_rows:
        severity = "error" if profile.ragged_rows > profile.row_count * 0.05 else "warning"
        _add(
            found,
            "ragged_rows",
            severity,
            f"{profile.ragged_rows} of {profile.row_count} rows have the wrong number of cells",
            None,
            "csv-doctor repair --out fixed.csv --pad-short-rows --drop-extra-cells",
        )

    if profile.duplicate_rows:
        share = profile.duplicate_rows / profile.row_count if profile.row_count else 0
        _add(
            found,
            "duplicate_rows",
            "warning" if share < 0.5 else "error",
            f"{profile.duplicate_rows} row(s) are exact duplicates",
            None,
            "csv-doctor repair --out fixed.csv --drop-duplicates",
        )

    for column in profile.columns:
        _check_column(column, found, null_warn, null_error)

    found.sort(key=lambda item: (-SEVERITY_ORDER.get(item.severity, 0), item.column or "", item.code))
    return found


def _check_column(
    column: ColumnProfile,
    found: List[Recommendation],
    null_warn: float,
    null_error: float,
) -> None:
    if column.total == 0:
        return
    if column.filled == 0:
        _add(
            found,
            "mostly_null",
            "error",
            f"column {column.name!r} is entirely empty",
            column.name,
            "verify the export or drop the column",
        )
        return

    if column.null_rate >= null_error:
        _add(
            found,
            "mostly_null",
            "error",
            f"column {column.name!r} is {column.null_rate:.0%} empty",
            column.name,
            "verify the export; consider --require " + column.name,
        )
    elif column.null_rate >= null_warn:
        _add(
            found,
            "nulls",
            "warning",
            f"column {column.name!r} is {column.null_rate:.0%} empty",
            column.name,
            "csv-doctor repair --out fixed.csv --nulls",
        )

    if column.unique:
        _add(
            found,
            "unique_column",
            "info",
            f"column {column.name!r} is unique across all {column.filled} rows",
            column.name,
            "good candidate key: --unique " + column.name,
        )

    if column.kind in ("date", "datetime"):
        if column.date_ambiguous:
            _add(
                found,
                "ambiguous_dates",
                "error",
                f"column {column.name!r} parses as more than one date format",
                column.name,
                "set an explicit format in rules.json before normalising",
            )
        elif column.kind == "date":
            _add(
                found,
                "normalisable_dates",
                "info",
                f"column {column.name!r} is a consistent date column",
                column.name,
                "csv-doctor repair --out fixed.csv --dates",
            )
    elif column.kind == "text" and column.filled and column.date_parsed:
        share = column.date_parsed / column.filled
        if share >= 0.8:
            _add(
                found,
                "mixed_date_formats",
                "error",
                f"column {column.name!r} is {share:.0%} date-like but uses "
                f"more than one format",
                column.name,
                "normalise to ISO 8601 upstream; --dates refuses to guess",
            )

    if column.kind in ("integer", "decimal") and column.numeric:
        stats = column.numeric
        if stats.negative and column.name.casefold() in (
            "age",
            "quantity",
            "count",
            "price",
            "amount",
        ):
            _add(
                found,
                "unexpected_negative",
                "warning",
                f"column {column.name!r} has {stats.negative} negative value(s)",
                column.name,
                "add a range rule with min: 0",
            )
        if column.outliers:
            worst = column.outliers[0]
            _add(
                found,
                "outliers",
                "info",
                f"{len(column.outliers)} outlier(s) in {column.name!r}; "
                f"largest at row {worst.row} ({worst.value})",
                column.name,
                "review the flagged rows before loading",
            )

    if column.kind == "text" and column.lengths and column.lengths.maximum > 512:
        _add(
            found,
            "very_long_values",
            "warning",
            f"column {column.name!r} has values up to {column.lengths.maximum} characters",
            column.name,
            "the delimiter may be wrong, or this column needs a text field",
        )

    if column.distinct_truncated:
        _add(
            found,
            "high_cardinality",
            "info",
            f"column {column.name!r} has more distinct values than tracked",
            column.name,
            "raise --max-distinct or exclude the column from top-value reporting",
        )
