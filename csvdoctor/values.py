"""Value parsing and type inference for messy tabular data.

Everything here is pure and string based: it never mutates the input and never
guesses silently.  Where a guess is unavoidable - is ``1.234`` one thousand two
hundred or one point two three four? - the caller is handed the ambiguity so it
can decide (``NumberStyle.UNKNOWN``, ``DateFormats.ambiguous``).

>>> parse_number("1,234.50")
1234.5
>>> parse_number("(42)")
-42.0
>>> infer_kind(["1", "2", "3"])
'integer'
>>> infer_kind(["2024-01-02", "2024-02-03"])
'date'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "NULL_TOKENS",
    "TRUE_TOKENS",
    "FALSE_TOKENS",
    "KINDS",
    "NumberStyle",
    "DateFormats",
    "ColumnType",
    "is_null",
    "is_boolean_token",
    "parse_boolean",
    "parse_number",
    "is_percent",
    "infer_number_style",
    "detect_date_formats",
    "parse_temporal",
    "normalize_date",
    "format_number",
    "infer_kind",
    "infer_column_type",
    "normalize_column_name",
    "unique_column_names",
]

NULL_TOKENS = frozenset(
    {"", "null", "none", "nil", "n/a", "na", "nan", "n.a.", "-", "--", "?", "#n/a"}
)
TRUE_TOKENS = frozenset({"true", "t", "yes", "y"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n"})

KINDS = (
    "boolean",
    "integer",
    "decimal",
    "date",
    "datetime",
    "email",
    "url",
    "uuid",
    "text",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://\S+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9 _.-]*$")  # reserved for future checks

_CURRENCY = "$€£¥₹"


def is_null(token: str) -> bool:
    """True when a token carries no value.

    >>> is_null("  N/A ")
    True
    >>> is_null("0")
    False
    """
    return token.strip().casefold() in NULL_TOKENS


def is_boolean_token(token: str) -> bool:
    folded = token.strip().casefold()
    return folded in TRUE_TOKENS or folded in FALSE_TOKENS


def parse_boolean(token: str) -> Optional[bool]:
    """Parse a boolean token, or return ``None`` when it is not one.

    >>> parse_boolean("YES"), parse_boolean("f"), parse_boolean("2")
    (True, False, None)
    """
    folded = token.strip().casefold()
    if folded in TRUE_TOKENS:
        return True
    if folded in FALSE_TOKENS:
        return False
    return None


def is_percent(token: str) -> bool:
    """True for tokens like ``12.5%``.

    >>> is_percent("12.5%")
    True
    """
    return token.strip().endswith("%")


def parse_number(token: str, *, style: str = "auto") -> Optional[float]:
    """Parse a human formatted number.

    Handles currency symbols, thousands separators, European decimal commas,
    accounting parentheses for negatives and a leading ``+``.

    >>> parse_number("€1.234,56", style="eu")
    1234.56
    >>> parse_number("-3")
    -3.0
    >>> parse_number("abc") is None
    True
    """
    text = token.strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    if text.endswith("%"):
        text = text[:-1].strip()
    text = text.strip(_CURRENCY).strip()
    text = text.replace("\u00a0", "").replace(" ", "").replace("_", "")
    text = text.replace("\u2212", "-")

    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-"):
        negative = not negative
        text = text[1:]

    if not text:
        return None

    decimal, group = _separators(text, style)
    if decimal is None or decimal not in text:
        if group and group in text and not _valid_grouping(text, group):
            return None
        cleaned = text.replace(",", "").replace(".", "").replace("'", "")
        if not cleaned.isdigit():
            return None
        value = float(cleaned)
    else:
        head, _, tail = text.rpartition(decimal)
        if not tail.isdigit() or not head:
            return None
        head = head.replace(group, "") if group else head
        head = head.replace(",", "").replace(".", "").replace("'", "")
        if not head.isdigit():
            return None
        value = float(f"{head}.{tail}")

    return -value if negative else value


def _separators(text: str, style: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(decimal_separator, group_separator)`` for a numeric token."""
    commas = text.count(",")
    dots = text.count(".")
    quotes = text.count("'")

    if style == "eu":
        return ",", "."
    if style == "us":
        return ".", ","

    if commas and dots:
        if text.rfind(",") > text.rfind("."):
            return ",", "."
        return ".", ","

    if quotes and not commas and not dots:
        if quotes >= 2:
            return None, "'"
        return None, "'"

    if commas:
        return (",", None) if _looks_like_decimal(text, ",") else (None, ",")
    if dots:
        return (".", None) if _looks_like_decimal(text, ".") else (None, ".")
    return None, None


def _valid_grouping(text: str, separator: str) -> bool:
    """A group separator must produce 1-3 leading digits then exact groups of 3.

    >>> _valid_grouping("1,234", ",")
    True
    >>> _valid_grouping("1.2.3.4", ".")
    False
    """
    parts = text.split(separator)
    if len(parts) < 2:
        return False
    if not 1 <= len(parts[0]) <= 3:
        return False
    return all(len(part) == 3 for part in parts[1:])


def _looks_like_decimal(text: str, separator: str) -> bool:
    """Heuristic for a single separator: is it a decimal point or a grouping?

    A group separator is only plausible between digit groups, so ``12,34`` is a
    decimal while ``1,234`` is a grouped thousand.  When a token could be read
    either way (``1.234``) we keep it as a decimal: that direction never
    multiplies a value by a thousand.
    """
    parts = text.split(separator)
    if len(parts) != 2:
        return False
    head, tail = parts
    if not head.isdigit() or not tail.isdigit():
        return False
    if len(tail) != 3:
        return True
    if len(head) > 3:
        return True
    return head == "0"


def infer_number_style(values: Iterable[str]) -> str:
    """Vote on the decimal separator used by a column.

    >>> infer_number_style(["1.234,56", "12,5"])
    'eu'
    >>> infer_number_style(["1,234.56", "12.5"])
    'us'
    >>> infer_number_style(["1", "2"])
    'auto'
    """
    eu = us = 0
    for token in values:
        text = token.strip().strip(_CURRENCY).strip()
        if not text or text.endswith("%"):
            continue
        commas = text.count(",")
        dots = text.count(".")
        if commas and dots:
            if text.rfind(",") > text.rfind("."):
                eu += 2
            else:
                us += 2
        elif commas == 1:
            tail = text.rsplit(",", 1)[1]
            if len(tail) in (1, 2):
                eu += 2
        elif dots == 1:
            tail = text.rsplit(".", 1)[1]
            if len(tail) in (1, 2):
                us += 2
    if eu > us:
        return "eu"
    if us > eu:
        return "us"
    return "auto"


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
)

_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
)


@dataclass(frozen=True)
class DateFormats:
    """The formats that parse a whole column, and whether they disagree.

    >>> found = detect_date_formats(["01/02/2024", "03/04/2024"])
    >>> found.ambiguous
    True
    """

    formats: Tuple[str, ...] = ()
    datetime: bool = False

    @property
    def ambiguous(self) -> bool:
        """True when more than one interpretation parses every value."""
        return len(self.formats) > 1

    @property
    def format(self) -> Optional[str]:
        """The unique format, or ``None`` when the column is ambiguous."""
        return self.formats[0] if len(self.formats) == 1 else None


def _try_formats(token: str, formats: Sequence[str]) -> List[str]:
    matched = []
    for fmt in formats:
        try:
            datetime.strptime(token, fmt)
        except ValueError:
            continue
        matched.append(fmt)
    return matched


def detect_date_formats(values: Iterable[str]) -> DateFormats:
    """Find every format that explains all values in a column.

    A column is only safe to normalise when exactly one format survives.
    """
    tokens = [v.strip() for v in values if v.strip()]
    if not tokens:
        return DateFormats()

    for formats, is_datetime in ((_DATETIME_FORMATS, True), (_DATE_FORMATS, False)):
        candidates = _try_formats(tokens[0], formats)
        for token in tokens[1:]:
            if not candidates:
                break
            candidates = [fmt for fmt in candidates if _try_formats(token, [fmt])]
        if candidates:
            return DateFormats(tuple(candidates), is_datetime)

    iso = []
    try:
        for token in tokens:
            datetime.fromisoformat(token.replace("Z", "+00:00"))
        iso.append("iso")
    except ValueError:
        pass
    if iso:
        has_time = any("T" in token or " " in token.strip() for token in tokens)
        return DateFormats(("iso",), has_time)
    return DateFormats()


def parse_temporal(token: str, fmt: Optional[str] = None) -> Optional[datetime]:
    """Parse one date or datetime token.

    >>> parse_temporal("2024-02-03").date().isoformat()
    '2024-02-03'
    >>> parse_temporal("garbage") is None
    True
    """
    text = token.strip()
    if not text:
        return None
    if fmt == "iso" or fmt is None:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            if fmt == "iso":
                return None
    for candidate in (_DATETIME_FORMATS + _DATE_FORMATS) if fmt is None else (fmt,):
        try:
            return datetime.strptime(text, candidate)
        except ValueError:
            continue
    return None


def normalize_date(token: str, fmt: str, *, datetime_ok: bool = True) -> Optional[str]:
    """Render a token as ISO 8601, or ``None`` when it does not parse.

    >>> normalize_date("01/03/2024", "%d/%m/%Y")
    '2024-03-01'
    """
    parsed = parse_temporal(token, fmt)
    if parsed is None:
        return None
    if parsed.hour or parsed.minute or parsed.second:
        if not datetime_ok:
            return None
        return parsed.isoformat()
    return parsed.date().isoformat()


def format_number(value: float) -> str:
    """Render an integral float without a trailing ``.0``.

    >>> format_number(1234.0), format_number(1.5)
    ('1234', '1.5')
    """
    if value == int(value):
        return str(int(value))
    return repr(round(value, 10))


@dataclass(frozen=True)
class ColumnType:
    """Inferred type information for one column."""

    kind: str = "text"
    number_style: str = "auto"
    dates: DateFormats = DateFormats()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "number_style": self.number_style,
            "date_format": self.dates.format,
            "date_ambiguous": self.dates.ambiguous,
        }


def infer_kind(values: Iterable[str], *, style: str = "auto") -> str:
    """Infer the narrowest kind that accepts every non-null value.

    >>> infer_kind(["1.5", "2"]), infer_kind(["yes", "NO"])
    ('decimal', 'boolean')
    >>> infer_kind(["alice@example.com"])
    'email'
    """
    tokens = [v.strip() for v in values if not is_null(v)]
    if not tokens:
        return "null"
    if all(is_boolean_token(t) for t in tokens):
        return "boolean"

    numbers = [parse_number(t, style=style) for t in tokens]
    if all(n is not None for n in numbers):
        if all(_is_integral(t, n) for t, n in zip(tokens, numbers)):
            return "integer"
        return "decimal"

    if all(_EMAIL_RE.match(t) for t in tokens):
        return "email"
    if all(_URL_RE.match(t) for t in tokens):
        return "url"
    if all(_UUID_RE.match(t) for t in tokens):
        return "uuid"

    dates = detect_date_formats(tokens)
    if dates.formats:
        return "datetime" if dates.datetime else "date"

    return "text"


def _is_integral(token: str, value: float) -> bool:
    if value != int(value):
        return False
    text = token.strip().strip(_CURRENCY).strip()
    if "." in text or "," in text:
        # A grouped thousand such as 1,234 is integral; a decimal point is not
        # integral unless every digit after it is zero.
        decimal, _ = _separators(text, "auto")
        if decimal is None:
            return True
        tail = text.rsplit(decimal, 1)[1]
        return set(tail) <= {"0"}
    return True


def infer_column_type(values: Iterable[str], *, style: Optional[str] = None) -> ColumnType:
    """Infer a :class:`ColumnType` for a column, including number/date style."""
    tokens = [v.strip() for v in values if not is_null(v)]
    if not tokens:
        return ColumnType("null")
    number_style = style or infer_number_style(tokens)
    kind = infer_kind(tokens, style=number_style)
    dates = detect_date_formats(tokens) if kind in ("date", "datetime") else DateFormats()
    return ColumnType(kind, number_style, dates)


def normalize_column_name(name: str, *, fallback: str = "column") -> str:
    """Turn a header cell into a stable snake_case identifier.

    >>> normalize_column_name("  First Name (billing) ")
    'first_name_billing'
    >>> normalize_column_name("")
    'column'
    """
    cleaned = name.strip().casefold()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned


def unique_column_names(names: Iterable[str]) -> List[str]:
    """Normalise names and suffix duplicates so the header stays addressable.

    Unnamed columns become ``column_<position>`` so nothing is lost.

    >>> unique_column_names(["Name", "name", "E-mail"])
    ['name', 'name_2', 'e_mail']
    >>> unique_column_names(["", "  "])
    ['column_1', 'column_2']
    """
    seen: dict = {}
    result = []
    for index, name in enumerate(names):
        candidate = normalize_column_name(name, fallback=f"column_{index + 1}")
        count = seen.get(candidate, 0) + 1
        seen[candidate] = count
        result.append(candidate if count == 1 else f"{candidate}_{count}")
    return result


def looks_like_header(row: Sequence[str], following: Sequence[str]) -> bool:
    """Guess whether ``row`` is a header rather than data.

    A row is treated as data when any cell parses as a number, date or boolean.
    Everything else is assumed to be a header - that is the common case, and a
    false positive is cheaper than renaming every column.

    >>> looks_like_header(["id", "email"], ["1", "a@b.co"])
    True
    >>> looks_like_header(["1", "2"], ["3", "4"])
    False
    >>> looks_like_header(["a", "a", ""], ["1", "2", ""])
    True
    """
    if not row:
        return False
    if all(not cell.strip() for cell in row):
        return False
    if all(is_null(cell) for cell in row):
        return False
    for cell in row:
        text = cell.strip()
        if not text:
            continue
        if parse_number(text) is not None:
            return False
        if is_boolean_token(text):
            return False
        if parse_temporal(text) is not None:
            return False
    return True
