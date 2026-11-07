"""End-to-end walkthrough of csv-doctor.

Run it from the repository root:

    python examples/demo.py

It writes only to a temporary directory and asserts every claim it makes, so it
doubles as a smoke test.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csvdoctor.checks import check_profile
from csvdoctor.dialect import load_table
from csvdoctor.profile import profile_table
from csvdoctor.repair import RepairOptions, apply_repair, infer_column_types
from csvdoctor.rules import parse_rules, validate_table

HERE = os.path.dirname(os.path.abspath(__file__))
MESSY = os.path.join(HERE, "messy.csv")
RULES = os.path.join(HERE, "rules.json")

BAR = "=" * 72


def heading(text: str) -> None:
    print(f"\n{BAR}\n{text}\n{BAR}")


def main() -> int:
    heading("1. what is actually in this file?")
    dialect, header, rows, lines = load_table(MESSY)
    profile = profile_table(header, rows, lines, dialect=dialect.to_dict())
    print(f"dialect: delimiter {dialect.delimiter!r}, encoding {dialect.encoding}")
    print(f"{profile.row_count} rows, {profile.column_count} columns, {profile.ragged_rows} ragged")
    for column in profile.columns:
        nulls = f"{column.null_rate:.0%} null" if column.nulls else "complete"
        print(f"  {column.name:<12} {column.kind:<9} {nulls}")
        assert column.kind in (
            "integer",
            "decimal",
            "text",
            "date",
            "datetime",
            "boolean",
            "email",
            "url",
            "uuid",
            "null",
        )

    heading("2. what should I do about it?")
    recommendations = check_profile(profile)
    for item in recommendations:
        print(f"[{item.severity}] {item.code}: {item.message}")
        if item.fix:
            print(f"        -> {item.fix}")
    codes = {item.code for item in recommendations}
    assert "ragged_rows" in codes, codes
    assert "mixed_date_formats" in codes, codes

    heading("3. does it satisfy my rules?")
    spec = parse_rules(json.load(open(RULES, encoding="utf-8")))
    _, findings, checked = validate_table(MESSY, spec)
    for finding in findings:
        print(f"row {finding.row}: {finding.severity} {finding.message}")
    assert findings, "the messy example should violate its own rules"
    assert checked == profile.row_count

    heading("4. what would a repair change?")
    types = infer_column_types(header, rows)
    plan_source = os.path.join(tempfile.mkdtemp(), "dry.csv")
    result = apply_repair(
        MESSY,
        plan_source,
        options=RepairOptions(
            drop_extra_cells=True,
            drop_duplicate_rows=True,
            coerce_numbers=True,
        ),
        column_types=types,
        dry_run=True,
    )
    print(json.dumps(result.plan.to_dict(), indent=2))
    assert result.plan.header != result.plan.original_header
    assert not os.path.exists(plan_source), "a dry run must not write anything"

    heading("5. apply it")
    folder = tempfile.mkdtemp()
    target = os.path.join(folder, "fixed.csv")
    applied = apply_repair(
        MESSY,
        target,
        options=RepairOptions(
            drop_extra_cells=True,
            drop_duplicate_rows=True,
            coerce_numbers=True,
        ),
        column_types=types,
    )
    print(f"wrote {applied.rows_written} rows to {target}")
    with open(target, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < 4:
                print("   " + line.rstrip())
    assert applied.rows_written == applied.rows_read - result.plan.dropped_rows

    heading("6. repairing twice is a no-op")
    second = os.path.join(folder, "fixed-again.csv")
    again = apply_repair(target, second, options=RepairOptions(
        drop_extra_cells=True,
        drop_duplicate_rows=True,
        coerce_numbers=True,
    ))
    print(f"changed: {again.plan.changed}")
    assert again.plan.changed is False

    heading("7. normalising dates and European numbers")
    orders = os.path.join(HERE, "orders.csv")
    order_dialect, order_header, order_rows, order_lines = load_table(orders)
    order_types = infer_column_types(order_header, order_rows)
    print(f"placed_on format: {order_types[1].dates.format}")
    print(f"total number style: {order_types[2].number_style}")
    assert order_types[1].dates.format == "%d/%m/%Y"
    assert order_types[2].number_style == "eu"
    normalised = os.path.join(folder, "orders.csv")
    apply_repair(
        orders,
        normalised,
        options=RepairOptions(coerce_dates=True, coerce_numbers=True),
        column_types=order_types,
    )
    with open(normalised, encoding="utf-8") as handle:
        for line in handle:
            print("   " + line.rstrip())
    with open(normalised, encoding="utf-8") as handle:
        text = handle.read()
    assert "2024-01-13" in text and "1299" in text and "-12" in text, text

    print("\nall demo assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
