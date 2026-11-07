"""Declarative validation rules for CSV files.

A rule file is JSON (or YAML-free inline text) describing expectations about
columns.  Validation is streaming and deterministic: findings come out in file
order, so two runs over the same input produce byte-identical reports.

    {
      "columns": {
        "email": [{"rule": "required"}, {"rule": "pattern", "value": "^[^@]+@[^@]+$"}],
        "age":   [{"rule": "range", "min": 0, "max": 130}],
        "tier":  [{"rule": "allowed", "values": ["free", "pro", "enterprise"]}]
      },
      "unique": [["id"], ["email"]],
      "required_columns": ["id", "email"],
      "max_rows": 1000000
    }

>>> spec = parse_rules({"columns": {"age": [{"rule": "range", "min": 0, "max": 10}]}})
>>> findings = validate_rows(["age"], [["5"], ["40"]], spec)
>>> [(f.rule, f.row, f.value) for f in findings]
[('range', 2, '40')]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .values import is_null, parse_number, parse_temporal

__all__ = [
    "SEVERITIES",
    "Finding",
    "RuleSpec",
    "RuleError",
    "parse_rules",
    "load_rules",
    "validate_rows",
    "validate_table",
]

SEVERITIES = ("error", "warning")


class RuleError(ValueError):
    """Raised when a rule file is structurally invalid."""


@dataclass(frozen=True)
class Finding:
    """One violated expectation."""

    rule: str
    severity: str
    column: Optional[str]
    row: Optional[int]
    value: Optional[str]
    message: str

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "column": self.column,
            "row": self.row,
            "value": self.value,
            "message": self.message,
        }


@dataclass
class RuleSpec:
    """A parsed rule set."""

    columns: Dict[str, List[dict]] = field(default_factory=dict)
    unique: List[List[str]] = field(default_factory=list)
    required_columns: List[str] = field(default_factory=list)
    forbidden_columns: List[str] = field(default_factory=list)
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "columns": self.columns,
            "unique": self.unique,
            "required_columns": self.required_columns,
            "forbidden_columns": self.forbidden_columns,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
        }


_SUPPORTED = {
    "required",
    "not_null",
    "pattern",
    "range",
    "allowed",
    "max_length",
    "min_length",
    "type",
    "unique",
}


def parse_rules(payload: dict) -> RuleSpec:
    """Validate and normalise a rule document.

    >>> parse_rules({"columns": {"x": [{"rule": "required"}]}}).columns
    {'x': [{'rule': 'required', 'severity': 'error'}]}
    >>> parse_rules({"columns": {"x": [{"rule": "nope"}]}})
    Traceback (most recent call last):
    ...
    csvdoctor.rules.RuleError: unknown rule 'nope' for column 'x'
    """
    if not isinstance(payload, dict):
        raise RuleError("rule document must be a JSON object")

    spec = RuleSpec(name=str(payload.get("name", "")))
    columns = payload.get("columns", {})
    if not isinstance(columns, dict):
        raise RuleError("'columns' must be an object of column -> rules")

    for column, rules in columns.items():
        if isinstance(rules, str):
            rules = [{"rule": rules}]
        if not isinstance(rules, list):
            raise RuleError(f"rules for column {column!r} must be a list")
        normalized = []
        for rule in rules:
            if isinstance(rule, str):
                rule = {"rule": rule}
            if not isinstance(rule, dict) or "rule" not in rule:
                raise RuleError(f"invalid rule for column {column!r}: {rule!r}")
            name = rule["rule"]
            if name not in _SUPPORTED:
                raise RuleError(f"unknown rule {name!r} for column {column!r}")
            entry = dict(rule)
            if name == "pattern":
                try:
                    re.compile(entry.get("value", ""))
                except re.error as exc:
                    raise RuleError(f"bad pattern for column {column!r}: {exc}") from exc
            if name == "allowed" and not isinstance(entry.get("values"), list):
                raise RuleError(f"'allowed' for column {column!r} needs a values list")
            entry.setdefault("severity", "error")
            if entry["severity"] not in SEVERITIES:
                raise RuleError(
                    f"severity for column {column!r} must be one of {SEVERITIES}"
                )
            normalized.append(entry)
        spec.columns[column] = normalized

    unique = payload.get("unique", [])
    if unique and not isinstance(unique, list):
        raise RuleError("'unique' must be a list of column groups")
    for group in unique:
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, list) or not group:
            raise RuleError(f"invalid unique group: {group!r}")
        spec.unique.append([str(name) for name in group])

    for key, target in (
        ("required_columns", spec.required_columns),
        ("forbidden_columns", spec.forbidden_columns),
    ):
        value = payload.get(key, [])
        if not isinstance(value, list):
            raise RuleError(f"'{key}' must be a list")
        target.extend(str(item) for item in value)

    for key in ("min_rows", "max_rows"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, int) or value < 0:
                raise RuleError(f"'{key}' must be a non-negative integer")
            setattr(spec, key, value)
    return spec


def load_rules(path: str) -> RuleSpec:
    """Read rules from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise RuleError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuleError(f"{path} is not valid JSON: {exc}") from exc
    return parse_rules(payload)


def _match(value: str, rule: dict, column_type: Optional[str]) -> Optional[str]:
    """Return a failure message, or ``None`` when the value satisfies ``rule``."""
    name = rule["rule"]
    if name in ("required", "not_null"):
        if value.strip() == "":
            return "value is required"
        return None
    if name == "pattern":
        pattern = re.compile(rule.get("value", ""))
        if not pattern.search(value):
            return f"does not match {rule.get('value')!r}"
        return None
    if name == "range":
        number = parse_number(value)
        if number is None:
            return "not a number"
        low = rule.get("min")
        high = rule.get("max")
        if low is not None and number < low:
            return f"{value} is below {low}"
        if high is not None and number > high:
            return f"{value} is above {high}"
        return None
    if name == "allowed":
        allowed = rule.get("values", [])
        if value not in allowed:
            return f"{value!r} is not one of {allowed}"
        return None
    if name == "max_length":
        limit = int(rule.get("value", 0))
        if len(value) > limit:
            return f"length {len(value)} exceeds {limit}"
        return None
    if name == "min_length":
        limit = int(rule.get("value", 0))
        if len(value) < limit:
            return f"length {len(value)} is under {limit}"
        return None
    if name == "type":
        expected = rule.get("value", "text")
        if not _has_type(value, expected):
            return f"{value!r} is not a valid {expected}"
        return None
    if name == "unique":
        return None  # handled by the streaming unique check
    return None


def _has_type(value: str, expected: str) -> bool:
    if expected in ("number", "integer", "decimal"):
        number = parse_number(value)
        if number is None:
            return False
        return number == int(number) if expected == "integer" else True
    if expected in ("date", "datetime"):
        return parse_temporal(value) is not None
    if expected == "email":
        return bool(re.match(r"^[^@\s]+@[^@\s.]+\.[^@\s.]+$", value))
    if expected == "url":
        return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://\S+$", value))
    if expected == "boolean":
        return value.strip().casefold() in ("true", "false", "yes", "no", "t", "f", "y", "n")
    if expected == "uuid":
        return bool(
            re.match(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
                value,
            )
        )
    return True


def _header_findings(header: Sequence[str], spec: RuleSpec, dialect_ok: bool = True) -> List[Finding]:
    findings: List[Finding] = []
    present = set(header)
    if not dialect_ok:
        findings.append(
            Finding("header", "error", None, None, None, "empty file: no header row")
        )
        return findings

    for column in spec.required_columns:
        if column not in present:
            findings.append(
                Finding(
                    "required_column",
                    "error",
                    column,
                    None,
                    None,
                    f"required column {column!r} is missing",
                )
            )
    for column in spec.forbidden_columns:
        if column in present:
            findings.append(
                Finding(
                    "forbidden_column",
                    "error",
                    column,
                    None,
                    None,
                    f"column {column!r} must not be present",
                )
            )
    for column in spec.columns:
        if column not in present and column not in spec.required_columns:
            findings.append(
                Finding(
                    "unknown_column",
                    "warning",
                    column,
                    None,
                    None,
                    f"rules reference column {column!r} which is not in the file",
                )
            )
    for group in spec.unique:
        for column in group:
            if column not in present:
                findings.append(
                    Finding(
                        "unknown_column",
                        "warning",
                        column,
                        None,
                        None,
                        f"unique group references missing column {column!r}",
                    )
                )
    return findings


def validate_rows(
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    spec: RuleSpec,
    lines: Optional[Iterable[int]] = None,
    *,
    column_types: Optional[Dict[str, str]] = None,
    max_findings: Optional[int] = None,
) -> List[Finding]:
    """Validate materialised rows, returning findings in file order."""
    findings = _header_findings(list(header), spec)
    rows = list(rows)
    types = column_types or {}
    indexes = {name: index for index, name in enumerate(header)}
    unique_seen: Dict[Tuple[str, ...], int] = {}

    def emit(finding: Finding) -> None:
        if max_findings is not None and len(findings) >= max_findings:
            return
        findings.append(finding)

    total = len(rows)
    if spec.min_rows is not None and total < spec.min_rows:
        emit(
            Finding(
                "min_rows",
                "error",
                None,
                None,
                str(total),
                f"only {total} row(s); expected at least {spec.min_rows}",
            )
        )
    if spec.max_rows is not None and total > spec.max_rows:
        emit(
            Finding(
                "max_rows",
                "error",
                None,
                None,
                str(total),
                f"{total} row(s); expected at most {spec.max_rows}",
            )
        )

    for position, fields in enumerate(rows):
        line = lines[position] if lines is not None else position + 1
        for column, rules in spec.columns.items():
            if column not in indexes:
                continue
            index = indexes[column]
            value = fields[index].strip() if index < len(fields) else ""
            if is_null(value):
                for rule in rules:
                    if rule["rule"] in ("required", "not_null"):
                        emit(
                            Finding(
                                rule["rule"],
                                rule.get("severity", "error"),
                                column,
                                line,
                                value,
                                "value is required",
                            )
                        )
                continue
            for rule in rules:
                if rule["rule"] == "unique":
                    continue
                message = _match(value, rule, types.get(column))
                if message is not None:
                    emit(
                        Finding(
                            rule["rule"],
                            rule.get("severity", "error"),
                            column,
                            line,
                            value,
                            message,
                        )
                    )

        for group in spec.unique:
            if any(column not in indexes for column in group):
                continue
            key = tuple(
                fields[indexes[column]].strip() if indexes[column] < len(fields) else ""
                for column in group
            )
            if all(is_null(part) for part in key):
                continue
            if key in unique_seen:
                first_line = unique_seen[key]
                emit(
                    Finding(
                        "unique",
                        "error",
                        ",".join(group),
                        line,
                        "|".join(key),
                        f"duplicate of row {first_line} on ({', '.join(group)})",
                    )
                )
            else:
                unique_seen[key] = line
    return findings


def validate_table(
    path: str,
    spec: RuleSpec,
    *,
    dialect=None,
    max_findings: Optional[int] = None,
    limit: Optional[int] = None,
) -> Tuple[List[str], List[Finding], int]:
    """Stream a file through ``spec``, returning ``(header, findings, rows)``."""
    from .dialect import TableReader
    from .values import infer_column_type

    with TableReader(path, dialect) as table:
        header = list(table.header)
        columns: Dict[str, List[str]] = {name: [] for name in header}
        lines: List[int] = []
        for line, fields in table.rows():
            lines.append(line)
            for index, name in enumerate(header):
                columns[name].append(fields[index] if index < len(fields) else "")
            if limit is not None and len(lines) >= limit:
                break

    types = {name: infer_column_type(values).kind for name, values in columns.items()}
    rows = [list(row) for row in zip(*columns.values())] if columns else []
    findings = validate_rows(
        header,
        rows,
        spec,
        lines,
        column_types=types,
        max_findings=max_findings,
    )
    return header, findings, len(lines)
