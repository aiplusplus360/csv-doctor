"""Bounded-memory profiling for CSV tables.

The profiler makes one pass over the file.  Per column it keeps counters, a
reservoir sample of numeric values and a capped value counter, so memory stays
proportional to the number of columns - not the number of rows.

>>> from csvdoctor.dialect import load_table
>>> import pathlib, tempfile
>>> folder = pathlib.Path(tempfile.mkdtemp())
>>> path = folder / "t.csv"
>>> _ = path.write_text("id,score,label\\n1,10,ok\\n2,20,ok\\n3,,bad\\n", encoding="utf-8")
>>> dialect, header, rows, lines = load_table(str(path))
>>> profile = profile_table(header, rows, lines)
>>> profile.row_count
3
>>> [c.name for c in profile.columns]
['id', 'score', 'label']
>>> profile.column("score").nulls
1
>>> profile.column("id").kind
'integer'
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .values import (
    ColumnType,
    infer_column_type,
    is_null,
    parse_number,
    parse_temporal,
)

__all__ = [
    "Outlier",
    "NumericStats",
    "LengthStats",
    "ColumnProfile",
    "TableProfile",
    "ProfileOptions",
    "profile_table",
]

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_MAX_DISTINCT = 10000
DEFAULT_MAX_DUPLICATE_TRACKING = 250_000


@dataclass
class Outlier:
    """A value far from the column median."""

    row: int
    value: float
    score: float

    def to_dict(self) -> dict:
        return {"row": self.row, "value": _tidy(self.value), "score": round(self.score, 2)}


@dataclass
class NumericStats:
    count: int = 0
    zero: int = 0
    negative: int = 0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    stdev: Optional[float] = None
    median: Optional[float] = None
    p05: Optional[float] = None
    p95: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "zero": self.zero,
            "negative": self.negative,
            "min": _tidy(self.minimum),
            "max": _tidy(self.maximum),
            "mean": _round(self.mean),
            "stdev": _round(self.stdev),
            "median": _round(self.median),
            "p05": _round(self.p05),
            "p95": _round(self.p95),
        }


@dataclass
class LengthStats:
    minimum: int = 0
    maximum: int = 0
    mean: float = 0.0
    median: float = 0.0

    def to_dict(self) -> dict:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
        }


@dataclass
class ColumnProfile:
    name: str
    index: int
    kind: str = "null"
    total: int = 0
    nulls: int = 0
    blanks: int = 0
    distinct: int = 0
    distinct_truncated: bool = False
    number_style: str = "auto"
    date_format: Optional[str] = None
    date_ambiguous: bool = False
    numeric: Optional[NumericStats] = None
    lengths: Optional[LengthStats] = None
    top_values: List[Tuple[str, int]] = field(default_factory=list)
    outliers: List[Outlier] = field(default_factory=list)
    date_parsed: int = 0

    @property
    def filled(self) -> int:
        return self.total - self.nulls

    @property
    def null_rate(self) -> float:
        return (self.nulls / self.total) if self.total else 0.0

    @property
    def unique(self) -> bool:
        return not self.distinct_truncated and self.distinct == self.filled and self.filled > 0

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "kind": self.kind,
            "total": self.total,
            "nulls": self.nulls,
            "null_rate": round(self.null_rate, 4),
            "blanks": self.blanks,
            "distinct": self.distinct,
            "distinct_truncated": self.distinct_truncated,
            "unique": self.unique,
            "number_style": self.number_style,
            "date_format": self.date_format,
            "date_ambiguous": self.date_ambiguous,
            "top_values": [{"value": value, "count": count} for value, count in self.top_values],
            "outliers": [outlier.to_dict() for outlier in self.outliers],
            "date_parsed": self.date_parsed,
        }
        if self.numeric is not None:
            payload["numeric"] = self.numeric.to_dict()
        if self.lengths is not None:
            payload["lengths"] = self.lengths.to_dict()
        return payload


@dataclass
class TableProfile:
    row_count: int = 0
    column_count: int = 0
    columns: List[ColumnProfile] = field(default_factory=list)
    duplicate_rows: int = 0
    duplicates_approximate: bool = False
    ragged_rows: int = 0
    ragged_lines: List[int] = field(default_factory=list)
    dialect: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    generated_at: str = ""

    def column(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "rows": self.row_count,
            "columns": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "duplicates_approximate": self.duplicates_approximate,
            "ragged_rows": self.ragged_rows,
            "ragged_lines": self.ragged_lines,
            "dialect": self.dialect,
            "warnings": self.warnings,
            "column_profiles": [column.to_dict() for column in self.columns],
        }


@dataclass(frozen=True)
class ProfileOptions:
    sample_size: int = DEFAULT_SAMPLE_SIZE
    max_distinct: int = DEFAULT_MAX_DISTINCT
    max_duplicate_tracking: int = DEFAULT_MAX_DUPLICATE_TRACKING
    outlier_threshold: float = 3.5
    top_values: int = 5
    max_outliers: int = 5
    max_rows: Optional[int] = None


class _ColumnAccumulator:
    """Streaming accumulator for one column."""

    def __init__(self, name: str, index: int, options: ProfileOptions):
        self.name = name
        self.index = index
        self.options = options
        self.total = 0
        self.nulls = 0
        self.blanks = 0
        self.counts: Dict[str, int] = {}
        self.distinct_truncated = False
        self.numeric_values: List[Tuple[float, int]] = []
        self.numeric_seen = 0
        self.length_values: List[Tuple[int, int]] = []
        self.length_seen = 0
        self.zero = 0
        self.negative = 0
        self.date_parsed = 0
        self.type_values: List[str] = []
        self._rng = random.Random(_seed(name))

    def add(self, token: str, row: int) -> None:
        self.total += 1
        if is_null(token):
            self.nulls += 1
            if token.strip():
                self.blanks += 1
            return

        text = token.strip()
        self._bump(text)
        if len(self.type_values) < 1000:
            self.type_values.append(text)
        if parse_temporal(text) is not None:
            self.date_parsed += 1
        self._sample_length(len(text), row)

        number = parse_number(text)
        if number is not None:
            if number == 0:
                self.zero += 1
            elif number < 0:
                self.negative += 1
            self._sample_number(number, row)

    def _bump(self, text: str) -> None:
        counts = self.counts
        counts[text] = counts.get(text, 0) + 1
        limit = self.options.max_distinct
        if len(counts) > limit:
            self.distinct_truncated = True
            keep = {key: value for key, value in counts.items() if value > 1}
            if len(keep) > limit // 2:
                ranked = sorted(keep.items(), key=lambda item: (-item[1], item[0]))
                keep = dict(ranked[: limit // 2])
            self.counts = keep

    def _sample_number(self, value: float, row: int) -> None:
        self.numeric_seen += 1
        reservoir = self.numeric_values
        if len(reservoir) < self.options.sample_size:
            reservoir.append((value, row))
            return
        index = self._rng.randrange(self.numeric_seen)
        if index < len(reservoir):
            reservoir[index] = (value, row)

    def _sample_length(self, length: int, row: int) -> None:
        self.length_seen += 1
        reservoir = self.length_values
        if len(reservoir) < self.options.sample_size:
            reservoir.append((length, row))
            return
        index = self._rng.randrange(self.length_seen)
        if index < len(reservoir):
            reservoir[index] = (length, row)

    def finish(self) -> ColumnProfile:
        options = self.options
        ordered = sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))
        top = ordered[: options.top_values]
        sample = self.type_values or list(self.counts.keys())
        column_type: ColumnType = infer_column_type(sample)
        profile = ColumnProfile(
            name=self.name,
            index=self.index,
            kind=column_type.kind,
            total=self.total,
            nulls=self.nulls,
            blanks=self.blanks,
            distinct=len(self.counts),
            distinct_truncated=self.distinct_truncated,
            number_style=column_type.number_style,
            date_format=column_type.dates.format,
            date_ambiguous=column_type.dates.ambiguous,
            top_values=top,
            date_parsed=self.date_parsed,
        )
        filled = max(self.total - self.nulls, 0)
        mostly_numeric = filled and self.numeric_seen / filled >= 0.5
        if self.numeric_seen and (numeric_kind(column_type.kind) or mostly_numeric):
            profile.numeric = _numeric_stats(self)
            profile.outliers = _numeric_outliers(self, profile.numeric, options)
        if self.length_seen:
            lengths = [length for length, _ in self.length_values]
            profile.lengths = LengthStats(
                minimum=min(lengths),
                maximum=max(lengths),
                mean=sum(lengths) / len(lengths),
                median=statistics.median(lengths),
            )
        return profile


def numeric_kind(kind: str) -> bool:
    return kind in ("integer", "decimal")


def _numeric_stats(accumulator: _ColumnAccumulator) -> NumericStats:
    sample = sorted(value for value, _ in accumulator.numeric_values)
    stats = NumericStats(
        count=accumulator.numeric_seen,
        zero=accumulator.zero,
        negative=accumulator.negative,
        minimum=sample[0],
        maximum=sample[-1],
        mean=sum(sample) / len(sample),
        median=statistics.median(sample),
        p05=_percentile(sample, 0.05),
        p95=_percentile(sample, 0.95),
    )
    if len(sample) > 1:
        stats.stdev = statistics.stdev(sample)
    return stats


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _numeric_outliers(
    accumulator: _ColumnAccumulator, stats: NumericStats, options: ProfileOptions
) -> List[Outlier]:
    sample = accumulator.numeric_values
    if len(sample) < 8:
        return []
    values = sorted(value for value, _ in sample)
    median = statistics.median(values)
    deviations = sorted(abs(value - median) for value in values)
    mad = statistics.median(deviations)
    if mad > 0:
        scores = [
            (row, value, 0.6745 * (value - median) / mad) for value, row in sample
        ]
    elif stats.stdev:
        scores = [
            (row, value, (value - median) / stats.stdev) for value, row in sample
        ]
    else:
        return []
    flagged = [
        Outlier(row, value, abs(score))
        for row, value, score in scores
        if abs(score) >= options.outlier_threshold
    ]
    flagged.sort(key=lambda outlier: (-outlier.score, outlier.row))
    return flagged[: options.max_outliers]


def _seed(name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def _tidy(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return int(value) if value == int(value) else round(value, 10)


def profile_table(
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
    lines: Optional[Iterable[int]] = None,
    *,
    options: Optional[ProfileOptions] = None,
    dialect: Optional[dict] = None,
) -> TableProfile:
    """Profile a table from an iterable of rows.

    >>> profile = profile_table(["a", "b"], [["1", "x"], ["2", ""]])
    >>> profile.row_count, profile.column("b").nulls
    (2, 1)
    """
    options = options or ProfileOptions()
    header = list(header)
    accumulators = [_ColumnAccumulator(name, index, options) for index, name in enumerate(header)]
    seen_rows: set = set()
    duplicates = 0
    duplicates_approximate = False
    ragged = 0
    ragged_lines: List[int] = []
    row_count = 0
    line_iterator = iter(lines) if lines is not None else None

    for fields in rows:
        line = next(line_iterator, row_count + 1) if line_iterator is not None else row_count + 1
        if options.max_rows is not None and row_count >= options.max_rows:
            break
        row_count += 1

        if len(fields) != len(header):
            ragged += 1
            if len(ragged_lines) < 10:
                ragged_lines.append(line)

        for index, accumulator in enumerate(accumulators):
            token = fields[index] if index < len(fields) else ""
            accumulator.add(token, line)

        digest = hashlib.blake2b(
            "\x1f".join(fields).encode("utf-8", "replace"), digest_size=8
        ).digest()
        if digest in seen_rows:
            duplicates += 1
        elif len(seen_rows) < options.max_duplicate_tracking:
            seen_rows.add(digest)
        else:
            duplicates_approximate = True

    columns = [accumulator.finish() for accumulator in accumulators]
    profile = TableProfile(
        row_count=row_count,
        column_count=len(header),
        columns=columns,
        duplicate_rows=duplicates,
        duplicates_approximate=duplicates_approximate,
        ragged_rows=ragged,
        ragged_lines=ragged_lines,
        dialect=dialect or {},
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    profile.warnings = _warnings(profile, header)
    return profile


def _warnings(profile: TableProfile, header: Sequence[str]) -> List[str]:
    warnings: List[str] = []
    if len(set(header)) != len(header):
        warnings.append("duplicate column names in header")
    if profile.ragged_rows:
        warnings.append(f"{profile.ragged_rows} row(s) do not match the header width")
    if profile.duplicate_rows:
        warnings.append(f"{profile.duplicate_rows} duplicate row(s)")
    empty = [column.name for column in profile.columns if column.total and not column.filled]
    if empty:
        warnings.append("empty columns: " + ", ".join(empty))
    for column in profile.columns:
        if column.kind == "text" and column.filled and column.lengths:
            if column.lengths.maximum > 200:
                warnings.append(f"column {column.name!r} has very long values")
        if column.nulls and column.total and column.null_rate == 1.0:
            warnings.append(f"column {column.name!r} is entirely null")
    return warnings
