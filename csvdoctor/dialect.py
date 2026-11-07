"""Delimiter sniffing and streaming reads for messy CSV files.

The reader is deliberately forgiving: it accepts ragged rows, mixed line
endings, byte order marks and four common encodings.  It never rewrites
anything - see :mod:`csvdoctor.repair` for that.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from typing import IO, Iterable, Iterator, List, Optional, Sequence, Tuple

from .values import looks_like_header

__all__ = [
    "CANDIDATE_DELIMITERS",
    "Dialect",
    "MalformedCsv",
    "sniff",
    "read_text",
    "TableReader",
    "load_table",
]

CANDIDATE_DELIMITERS = (",", ";", "\t", "|")
CANDIDATE_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
SNIFF_BYTES = 65536


class MalformedCsv(ValueError):
    """Raised when the file cannot be parsed as CSV at all."""


@dataclass(frozen=True)
class Dialect:
    """The physical shape of a delimited file."""

    delimiter: str = ","
    quotechar: str = '"'
    encoding: str = "utf-8-sig"
    lineterminator: str = "\n"
    has_header: bool = True

    def to_dict(self) -> dict:
        return {
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
            "encoding": self.encoding,
            "line_terminator": {"\r\n": "crlf", "\n": "lf", "\r": "cr"}.get(
                self.lineterminator, self.lineterminator
            ),
            "has_header": self.has_header,
        }

    def as_csv(self) -> dict:
        """Keyword arguments for :func:`csv.reader`/:func:`csv.writer`."""
        return {"delimiter": self.delimiter, "quotechar": self.quotechar}


def _count_outside_quotes(line: str, delimiter: str, quotechar: str) -> int:
    count = 0
    in_quotes = False
    for char in line:
        if char == quotechar:
            in_quotes = not in_quotes
        elif char == delimiter and not in_quotes:
            count += 1
    return count


def sniff(text: str) -> Dialect:
    """Infer delimiter, quote char, header and line terminator from a sample.

    >>> sniff("a;b;c\\n1;2;3\\n").delimiter
    ';'
    >>> sniff("a,b\\n1,2\\n").has_header
    True
    """
    sample = text.lstrip("\ufeff")
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return Dialect()

    quotechar = '"'
    if sample.count("'") > sample.count('"') and sample.count("'") % 2 == 0:
        quotechar = "'"

    best: Tuple[float, str] = (-1.0, ",")
    for delimiter in CANDIDATE_DELIMITERS:
        counts = [_count_outside_quotes(line, delimiter, quotechar) for line in lines[:50]]
        if not counts or max(counts) == 0:
            continue
        positive = [c for c in counts if c > 0]
        consistency = len(positive) / len(counts)
        mean = sum(positive) / len(positive)
        if mean < 1:
            continue
        score = consistency * (mean + 1.0)
        if consistency == 1.0 and len(set(positive)) > 1:
            score -= 3.0  # ragged columns are a worse signal than clean ones
        if score > best[0]:
            best = (score, delimiter)

    delimiter = best[1]
    rows = _split_sample(sample, delimiter, quotechar)
    header = looks_like_header(rows[0], rows[1] if len(rows) > 1 else []) if rows else True
    terminator = "\r\n" if "\r\n" in sample else ("\r" if "\r" in sample else "\n")
    return Dialect(delimiter, quotechar, "utf-8-sig", terminator, header)


def _split_sample(sample: str, delimiter: str, quotechar: str) -> List[List[str]]:
    try:
        return list(csv.reader(io.StringIO(sample), delimiter=delimiter, quotechar=quotechar))
    except csv.Error:
        return [line.split(delimiter) for line in sample.splitlines()]


def read_text(path: str) -> Tuple[str, str, bool]:
    """Read a file, returning ``(text, encoding, had_bom)``.

    Falls back through UTF-8, cp1252 and latin-1 so that a stray Windows-1252
    byte never makes the whole file unreadable.

    >>> import tempfile, pathlib
    >>> p = pathlib.Path(tempfile.mkdtemp()) / "bom.csv"
    >>> _ = p.write_bytes(b"\\xef\\xbb\\xbfa,b\\n1,2\\n")
    >>> read_text(str(p))[2]
    True
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    for encoding in CANDIDATE_ENCODINGS:
        try:
            return raw.decode(encoding), encoding, had_bom
        except UnicodeDecodeError:
            continue
    raise MalformedCsv(f"cannot decode {path} as any of {', '.join(CANDIDATE_ENCODINGS)}")


class TableReader:
    """Streaming CSV reader that yields ``(line_number, fields)``.

    Use as a context manager.  ``header`` is already removed from ``rows()``
    when the dialect says the first line is a header.

    >>> import tempfile, pathlib
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> path = folder / "t.csv"
    >>> _ = path.write_text("name,score\\nada,3\\ngrace,4\\n", encoding="utf-8")
    >>> with TableReader(str(path)) as table:
    ...     header = table.header
    ...     rows = [row for _, row in table.rows()]
    >>> header, rows
    (['name', 'score'], [['ada', '3'], ['grace', '4']])
    """

    def __init__(self, path: str, dialect: Optional[Dialect] = None):
        self.path = path
        self._explicit = dialect
        self._handle: Optional[IO[str]] = None
        self.dialect = dialect or Dialect()
        self.header: List[str] = []
        self._first_data: List[str] = []

    def __enter__(self) -> "TableReader":
        if not os.path.exists(self.path):
            raise MalformedCsv(f"cannot read {self.path}: no such file")
        text, encoding, _ = read_text(self.path)
        if self._explicit is None:
            sniffed = sniff(text[:SNIFF_BYTES])
            self.dialect = Dialect(
                sniffed.delimiter,
                sniffed.quotechar,
                encoding,
                sniffed.lineterminator,
                sniffed.has_header,
            )
        else:
            terminator = "\r\n" if "\r\n" in text[:SNIFF_BYTES] else "\n"
            self.dialect = Dialect(
                self._explicit.delimiter,
                self._explicit.quotechar,
                encoding,
                terminator,
                self._explicit.has_header,
            )
        self._handle = open(self.path, "r", encoding=encoding, newline="")
        reader = csv.reader(self._handle, **self.dialect.as_csv())
        try:
            first = next(reader, None)
        except csv.Error as exc:
            self._handle.close()
            raise MalformedCsv(f"{self.path}: {exc}") from exc
        if first is None:
            self.header = []
        elif self.dialect.has_header:
            self.header = [cell for cell in first]
        else:
            self.header = [f"column_{i + 1}" for i in range(len(first))]
        self._first_data = [] if self.dialect.has_header else [first]
        self._reader = reader
        return self

    def rows(self) -> Iterator[Tuple[int, List[str]]]:
        """Yield ``(line_number, fields)`` for every data row."""
        for fields in self._first_data:
            yield 1, fields
        self._first_data = []
        assert self._handle is not None
        while True:
            try:
                fields = next(self._reader)
            except StopIteration:
                return
            except csv.Error as exc:
                raise MalformedCsv(f"{self.path}: {exc}") from exc
            yield self._reader.line_num, fields

    def __exit__(self, *exc_info) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        self.__exit__()


def load_table(
    path: str, dialect: Optional[Dialect] = None, *, limit: Optional[int] = None
) -> Tuple[Dialect, List[str], List[List[str]], List[int]]:
    """Load a whole small table, returning ``(dialect, header, rows, lines)``."""
    with TableReader(path, dialect) as table:
        rows: List[List[str]] = []
        lines: List[int] = []
        for line, fields in table.rows():
            rows.append(fields)
            lines.append(line)
            if limit is not None and len(rows) >= limit:
                break
        return table.dialect, table.header, rows, lines
