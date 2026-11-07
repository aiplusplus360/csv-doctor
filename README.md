# csv-doctor

**Stop loading broken CSVs into production.** Profile any CSV in one pass — types, nulls, outliers, ragged rows, duplicate keys — then repair it with a plan you can read *before* anything is written.

[![CI](https://github.com/aiplusplus360/csv-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/aiplusplus360/csv-doctor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.13-blue)](pyproject.toml)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

```console
$ csv-doctor profile examples/messy.csv
10 rows x 6 columns
delimiter ','  quote '"'  encoding utf-8-sig  header yes

column       kind        nulls distinct  notes
---------------------------------------------
id           integer         0       10  median 1005.5 range 1001..1010
Full Name    text      1 (10%)        8
Signup Date  text            0        9
MRR          integer   1 (10%)        6  median 1299 range -50..1299
Status       text            0        3
Notes        text      2 (20%)        7

recommendations:
  x [ragged_rows] 1 of 10 rows have the wrong number of cells
      fix: csv-doctor repair --out fixed.csv --pad-short-rows --drop-extra-cells
  x [mixed_date_formats] column 'Signup Date' is 100% date-like but uses more than one format
      fix: normalise to ISO 8601 upstream; --dates refuses to guess
  ! [nulls] column 'Notes' is 20% empty
  - [unique_column] column 'id' is unique across all 10 rows
      fix: good candidate key: --unique id
```

```console
$ csv-doctor validate examples/messy.csv --rules examples/rules.json
error: row 3 [Status] 'active' is not one of ['Active', 'Churned', 'Trial']
warning: row 5 [MRR] -50 is below 0
error: row 7 [MRR] value is required
error: row 10 [id] duplicate of row 9 on (id)

4 error(s), 1 warning(s), 10 row(s) checked
```

```console
$ csv-doctor repair examples/messy.csv --out fixed.csv --drop-extra-cells --drop-duplicates --numbers
repaired: examples/messy.csv
  rows read:    10
  rows written: 9
  header:       ['id', 'full_name', 'signup_date', 'mrr', 'status', 'notes']
  header: 5 (5 column name(s) rewritten)
  coerce_numbers: 7 (currency symbols and separators were removed)
  drop_duplicate_rows: 1 (rows identical after normalisation were dropped)
  drop_extra_cells: 1 (trailing cells beyond the header width were dropped)
```

---

## Why this exists

Every data pipeline starts with `pd.read_csv()` and ends three weeks later with a
`ValueError` about a column that was *sometimes* a date and *sometimes* a string,
plus a row count nobody can explain. The failure is never in the analysis — it is
in the assumption that the file is what it claims to be.

csv-doctor answers four questions in order, and refuses to answer the ones that
cannot be answered safely:

| Question | Command | Hard part |
|---|---|---|
| What is actually in this file? | `profile` | Sniffing the delimiter, quote char and encoding from a 64 KB sample, then inferring the narrowest type per column *without* buffering the file |
| What is wrong with it? | `profile` | Distinguishing "this column is 90% empty" from "this column is empty" and "these are two date formats in one column" |
| Does it meet my contract? | `validate` | Rules that are declarative, ordered and reproducible — findings come out in file order |
| Can I fix it? | `repair` | Making every change explicit in a plan before a byte is rewritten |

The design rule is **never guess silently**. Where a guess is genuinely ambiguous —
is `01/02/2024` January 2nd or February 1st? — the tool says so and leaves the
value alone. `profile` reports `ambiguous_dates` or `mixed_date_formats`, and
`repair --dates` skips the column rather than picking an interpretation.

### Where other tools go wrong

| Detail | Why it matters |
|---|---|
| `1.234,56` vs `1,234.56` | The same token is a thousand times different depending on locale. csv-doctor votes across the column for a decimal separator and reports the winner. |
| `(42)` is `-42` | Accounting exports use parentheses for negatives. Treated as a number, not as text with punctuation. |
| `1,234` is a thousand, `1.234` is not | A single separator is a group separator only if the digit groups are groups of three; otherwise it is a decimal point. |
| A BOM is not part of the first column name | Read as `utf-8-sig`, and the original encoding is reported rather than silently rewritten. |
| One stray `\x92` kills the whole file | Falls back from UTF-8 to cp1252 to latin-1 instead of raising on a Windows export. |
| Ragged rows are reported, not silently dropped | `profile` counts them; `repair` only pads or truncates when you ask. |
| Deduplication is bounded | A hash set capped by `max_duplicate_tracking`, and the report tells you when it became approximate. |

---

## Install

```bash
pip install csv-doctor
```

Python 3.9+, standard library only. From a checkout:

```bash
pip install -e ".[dev]"
```

---

## Commands

| Command | Purpose | Exit codes |
|---|---|---|
| `csv-doctor profile FILE` | Describe the file and recommend fixes | `0` clean, `1` warnings, `2` errors (`--strict` promotes `1` to `2`) |
| `csv-doctor validate FILE --rules RULES.json` | Check against a contract | `0` pass, `1` warnings with `--fail-on warning`, `2` errors |
| `csv-doctor repair FILE --out OUT` | Rewrite the file | `0` wrote or planned, `2` input problem |
| `csv-doctor sniff FILE` | Report the dialect only | `0` |

Every command takes `--json` and prints a stable object. `--no-header` treats row one
as data and generates `column_1..n`. `--delimiter` and `--quotechar` override
detection when sniffing is wrong.

### Profile options

| Flag | Default | What it controls |
|---|---|---|
| `--max-distinct N` | `10000` | Distinct values tracked per column; beyond this counts become a lower bound and `distinct_truncated` is set |
| `--sample-size N` | `2000` | Reservoir size for numeric and length statistics — constant memory regardless of file size |
| `--max-rows N` | none | Stop after N rows |
| `--top N` | `5` | Most common values reported per column |
| `--strict` | off | Treat warnings as errors |

### Outlier detection

Outliers use the median absolute deviation, not standard deviation: one absurd value
cannot inflate the threshold that is supposed to catch it. A value is flagged when
`0.6745 * (x - median) / MAD` exceeds `3.5`. Columns with fewer than eight numeric
values are never flagged.

### Repair flags

| Flag | Effect |
|---|---|
| `--dry-run` | Print the plan, write nothing |
| `--no-trim` | Keep leading/trailing whitespace (trimming is on by default) |
| `--collapse-whitespace` | Turn runs of spaces into one |
| `--nulls` | Rewrite `N/A`, `-`, `null`, `?` … to an empty cell |
| `--drop-extra-cells` | Truncate rows longer than the header (short rows are always padded) |
| `--drop-duplicates` | Drop rows identical after normalisation |
| `--dates` | Rewrite single-format date columns as ISO 8601; ambiguous columns are skipped |
| `--numbers` | Strip currency symbols and separators, normalise to a plain decimal |
| `--keep-header` | Do not rename columns |
| `--encoding` | Output encoding (default `utf-8`) |
| `--line-terminator lf\|crlf` | Output line endings |

Repair is idempotent: running it on its own output reports `changed: false`.

---

## Rule files

Rules are JSON, with a deliberately small vocabulary so a reviewer can read the
contract in ten seconds.

```json
{
  "name": "customer-export",
  "required_columns": ["id", "email"],
  "forbidden_columns": ["password"],
  "unique": [["id"], ["email", "tenant"]],
  "min_rows": 1,
  "max_rows": 500000,
  "columns": {
    "id":    [{ "rule": "required" }, { "rule": "type", "value": "integer" }],
    "email": [{ "rule": "pattern", "value": "^[^@]+@[^@]+\\.[^@]+$" }],
    "age":   [{ "rule": "range", "min": 0, "max": 130 }],
    "tier":  [{ "rule": "allowed", "values": ["free", "pro", "enterprise"] }],
    "notes": [{ "rule": "max_length", "value": 200, "severity": "warning" }]
  }
}
```

| Rule | Applies to | Notes |
|---|---|---|
| `required` / `not_null` | column | Empty and null-token values fail |
| `pattern` | column | Python regex, `re.search` semantics; compiled once at load |
| `range` | column | `min`/`max` inclusive; a non-numeric value is its own finding |
| `allowed` | column | Exact match against a list |
| `min_length` / `max_length` | column | Character count |
| `type` | column | `integer`, `decimal`, `date`, `datetime`, `email`, `url`, `boolean`, `uuid` |
| `unique` | group | Duplicates report the first row they collided with; all-null keys are skipped |

Invalid rule files are a **usage error** (`exit 2`), not a finding — a typo in your
contract should never look like bad data.

---

## Library use

```python
from csvdoctor.dialect import load_table
from csvdoctor.profile import profile_table

dialect, header, rows, lines = load_table("export.csv")
profile = profile_table(header, rows, lines, dialect=dialect.to_dict())

for column in profile.columns:
    print(column.name, column.kind, f"{column.null_rate:.0%} null")
    if column.outliers:
        print("  outliers:", [o.to_dict() for o in column.outliers])
```

```python
from csvdoctor.repair import RepairOptions, apply_repair, infer_column_types
from csvdoctor.dialect import load_table

_, header, rows, _ = load_table("export.csv")
types = infer_column_types(header, rows)
apply_repair(
    "export.csv",
    "clean.csv",
    options=RepairOptions(coerce_dates=True, coerce_numbers=True, drop_duplicate_rows=True),
    column_types=types,
)
```

```python
from csvdoctor.rules import load_rules, validate_table

spec = load_rules("rules.json")
header, findings, checked = validate_table("export.csv", spec)
errors = [f for f in findings if f.severity == "error"]
```

`csvdoctor.profile_table` accepts any iterable of rows, so it works on a CSV, a
database cursor, or a generator of API pages.

---

## Design notes

- **One pass, bounded memory.** Profiling keeps counters, two reservoirs and a capped
  distinct map per column. Memory scales with columns, not rows.
- **Plan then apply.** `plan_repair` is pure and returns dataclasses; `apply_repair`
  executes them. That split is what makes `--dry-run` trustworthy and the library
  usable from a UI.
- **Deterministic.** Reservoir sampling is seeded from the column name, so the same
  file always produces the same statistics and the same JSON.
- **Types are the narrowest fit.** A column of `1, 2, 3` is an integer column, not a
  string column with numeric-looking values; a column with `1, 2.5` is a decimal.
- **Ambiguity is a finding.** `01/02/2024` produces `ambiguous_dates` rather than a
  50% chance of silently corrupting every date in the file.

## Testing

```bash
python -m unittest discover -s tests -v   # 142 tests
python examples/demo.py                   # end-to-end, asserts its own output
python -m doctest csvdoctor/*.py
```

The suite covers delimiter sniffing, encoding fallback, locale number parsing,
ambiguous dates, ragged rows, deduplication, rule parsing errors, exit codes, and
byte-level repair output.

## License

MIT.
