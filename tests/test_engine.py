"""Profiling, repair, rules, checks and the CLI contract."""

import io
import json
import os
import tempfile
import unittest

from contextlib import redirect_stderr, redirect_stdout

from csvdoctor.checks import check_profile, exit_code
from csvdoctor.cli import main
from csvdoctor.dialect import Dialect, load_table
from csvdoctor.profile import ProfileOptions, profile_table
from csvdoctor.repair import (
    RepairOptions,
    apply_repair,
    infer_column_types,
    plan_repair,
)
from csvdoctor.rules import (
    RuleError,
    load_rules,
    parse_rules,
    validate_rows,
    validate_table,
)


class TempDirMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def path(self, name):
        return os.path.join(self.dir, name)

    def write(self, name, text, encoding="utf-8", newline=""):
        target = self.path(name)
        with open(target, "w", encoding=encoding, newline=newline) as handle:
            handle.write(text)
        return target

    def write_json(self, name, payload):
        target = self.path(name)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return target

    def profile(self, name="t.csv"):
        path = self.path(name)
        dialect, header, rows, lines = load_table(path)
        return profile_table(header, rows, lines, dialect=dialect.to_dict())


class ProfileTests(TempDirMixin, unittest.TestCase):
    def test_counts_and_kinds(self):
        self.write("t.csv", "id,score,label\n1,10,ok\n2,20,ok\n3,,bad\n")
        profile = self.profile()
        self.assertEqual(profile.row_count, 3)
        self.assertEqual(profile.column_count, 3)
        self.assertEqual(profile.column("id").kind, "integer")
        self.assertEqual(profile.column("label").kind, "text")
        self.assertEqual(profile.column("score").nulls, 1)
        self.assertAlmostEqual(profile.column("score").null_rate, 1 / 3)

    def test_numeric_stats(self):
        self.write("t.csv", "v\n" + "".join(f"{i}\n" for i in range(1, 11)))
        numeric = self.profile().column("v").numeric
        self.assertEqual(numeric.count, 10)
        self.assertEqual(numeric.minimum, 1)
        self.assertEqual(numeric.maximum, 10)
        self.assertAlmostEqual(numeric.mean, 5.5)
        self.assertAlmostEqual(numeric.median, 5.5)

    def test_top_values_are_ordered(self):
        self.write("t.csv", "c\na\na\nb\nc\na\n")
        column = self.profile().column("c")
        self.assertEqual(column.top_values[0], ("a", 3))
        self.assertEqual(column.distinct, 3)

    def test_distinct_truncation(self):
        rows = "".join(f"v{i}\n" for i in range(200))
        self.write("t.csv", "c\n" + rows)
        options = ProfileOptions(max_distinct=10)
        dialect, header, data, lines = load_table(self.path("t.csv"))
        profile = profile_table(header, data, lines, options=options)
        self.assertTrue(profile.column("c").distinct_truncated)

    def test_outliers(self):
        values = [10] * 40 + [1000]
        self.write("t.csv", "v\n" + "".join(f"{v}\n" for v in values))
        column = self.profile().column("v")
        self.assertTrue(column.outliers, column.to_dict())
        self.assertEqual(column.outliers[0].value, 1000)

    def test_duplicate_rows(self):
        self.write("t.csv", "a,b\n1,x\n1,x\n2,y\n")
        self.assertEqual(self.profile().duplicate_rows, 1)

    def test_duplicates_are_approximate_past_the_cap(self):
        rows = "".join(f"{i},x\n" for i in range(20))
        self.write("t.csv", "a,b\n" + rows + "0,x\n")
        dialect, header, data, lines = load_table(self.path("t.csv"))
        profile = profile_table(
            header, data, lines, options=ProfileOptions(max_duplicate_tracking=5)
        )
        self.assertGreaterEqual(profile.duplicate_rows, 1)
        self.assertTrue(profile.duplicates_approximate)

    def test_ragged_rows(self):
        self.write("t.csv", "a,b,c\n1,2\n1,2,3\n1\n")
        profile = self.profile()
        self.assertEqual(profile.ragged_rows, 2)
        self.assertEqual(profile.ragged_lines, [2, 4])

    def test_max_rows(self):
        self.write("t.csv", "a\n" + "".join(f"{i}\n" for i in range(50)))
        dialect, header, data, lines = load_table(self.path("t.csv"))
        profile = profile_table(header, data, lines, options=ProfileOptions(max_rows=5))
        self.assertEqual(profile.row_count, 5)

    def test_warnings(self):
        self.write("t.csv", "a,a,\n1,2,\n")
        profile = self.profile()
        joined = " ".join(profile.warnings)
        self.assertIn("duplicate column names", joined)
        self.assertIn("entirely null", joined)

    def test_blank_vs_null(self):
        self.write("t.csv", "a,b\n,NA\n1,x\n")
        column = self.profile().column("b")
        self.assertEqual(column.nulls, 1)
        self.assertEqual(column.blanks, 1)

    def test_lengths(self):
        self.write("t.csv", "name\nada\ngrace\n")
        lengths = self.profile().column("name").lengths
        self.assertEqual(lengths.minimum, 3)
        self.assertEqual(lengths.maximum, 5)

    def test_json_shape(self):
        self.write("t.csv", "a\n1\n")
        payload = self.profile().to_dict()
        for key in ("rows", "columns", "dialect", "column_profiles", "warnings"):
            self.assertIn(key, payload)
        self.assertIn("numeric", payload["column_profiles"][0])

    def test_deterministic(self):
        self.write("t.csv", "a\n" + "".join(f"{i}\n" for i in range(100)))
        first = self.profile().to_dict()
        second = self.profile().to_dict()
        first.pop("generated_at")
        second.pop("generated_at")
        self.assertEqual(first, second)


class CheckTests(TempDirMixin, unittest.TestCase):
    def codes(self, text):
        self.write("t.csv", text)
        return [item.code for item in check_profile(self.profile())]

    def test_duplicate_headers(self):
        self.assertIn("duplicate_headers", self.codes("a,a\n1,2\n"))

    def test_ragged(self):
        self.assertIn("ragged_rows", self.codes("a,b\n1\n2\n"))

    def test_duplicate_rows(self):
        self.assertIn("duplicate_rows", self.codes("a\n1\n1\n"))

    def test_nulls(self):
        self.assertIn("nulls", self.codes("a,b\n1,\n2,\n3,x\n"))

    def test_mostly_null(self):
        self.assertIn("mostly_null", self.codes("a,b\n1,\n2,\n3,\n"))

    def test_unique_column(self):
        self.assertIn("unique_column", self.codes("id\n1\n2\n3\n"))

    def test_mixed_dates(self):
        self.assertIn(
            "mixed_date_formats",
            self.codes("d\n2024-01-05\n05/02/2024\n2024-03-07\n2024-04-08\n"),
        )

    def test_normalisable_dates(self):
        self.assertIn("normalisable_dates", self.codes("d\n2024-01-05\n2024-02-06\n"))

    def test_ambiguous_dates(self):
        self.assertIn(
            "ambiguous_dates", self.codes("d\n01/02/2024\n03/04/2024\n05/06/2024\n")
        )

    def test_negative_age(self):
        self.assertIn("unexpected_negative", self.codes("age\n30\n-1\n40\n"))

    def test_outliers(self):
        rows = "".join("10\n" for _ in range(30)) + "9999\n"
        self.assertIn("outliers", self.codes("v\n" + rows))

    def test_empty_file(self):
        self.write("t.csv", "")
        profile = self.profile()
        self.assertIn("empty_file", [item.code for item in check_profile(profile)])

    def test_sorted_by_severity(self):
        self.write("t.csv", "a,a\n1,2\n1,2\n")
        severities = [item.severity for item in check_profile(self.profile())]
        order = {"error": 3, "warning": 2, "info": 1}
        self.assertEqual(
            severities, sorted(severities, key=lambda s: -order[s])
        )

    def test_exit_codes(self):
        self.assertEqual(exit_code("info"), 0)
        self.assertEqual(exit_code("warning"), 1)
        self.assertEqual(exit_code("error"), 2)
        self.assertEqual(exit_code("nonsense"), 2)


class PlanTests(TempDirMixin, unittest.TestCase):
    def test_header_is_normalised(self):
        plan = plan_repair([" Name ", "Name", ""], [["a", "b", "c"]])
        self.assertEqual(plan.header, ["name", "name_2", "column_3"])
        self.assertTrue(plan.changed)

    def test_keep_header(self):
        plan = plan_repair(["A B"], [["1"]], options=RepairOptions(normalize_header=False))
        self.assertEqual(plan.header, ["A B"])

    def test_padding(self):
        plan = plan_repair(["a", "b"], [["1"]], options=RepairOptions())
        self.assertEqual(plan.pad_rows, 1)

    def test_truncation(self):
        plan = plan_repair(
            ["a", "b"], [["1", "2", "3"]], options=RepairOptions(drop_extra_cells=True)
        )
        self.assertEqual(plan.truncate_rows, 1)

    def test_extra_cells_kept_by_default(self):
        plan = plan_repair(["a", "b"], [["1", "2", "3"]])
        self.assertEqual(plan.truncate_rows, 0)

    def test_null_unification(self):
        plan = plan_repair(
            ["a"], [["N/A"], ["-"], [""]], options=RepairOptions(unify_nulls=True)
        )
        self.assertEqual(plan.null_values, 2)

    def test_duplicates(self):
        plan = plan_repair(
            ["a", "b"],
            [["1", "x"], ["1", "x"], ["2", "y"]],
            options=RepairOptions(drop_duplicate_rows=True),
        )
        self.assertEqual(plan.dropped_rows, 1)

    def test_duplicates_respect_trimming(self):
        plan = plan_repair(
            ["a"], [[" x "], ["x"]], options=RepairOptions(drop_duplicate_rows=True)
        )
        self.assertEqual(plan.dropped_rows, 1)

    def test_number_rewrite(self):
        types = infer_column_types(["amount"], [["$1,234.50"]])
        plan = plan_repair(
            ["amount"],
            [["$1,234.50"]],
            options=RepairOptions(coerce_numbers=True),
            column_types=types,
        )
        self.assertEqual(plan.rewritten_numbers, 1)

    def test_plan_json(self):
        plan = plan_repair([" A "], [["x"]])
        payload = plan.to_dict()
        self.assertIn("actions", payload)
        self.assertTrue(payload["changed"])


class RepairTests(TempDirMixin, unittest.TestCase):
    def repair(self, text, **options):
        source = self.write("in.csv", text)
        target = self.path("out.csv")
        result = apply_repair(source, target, options=RepairOptions(**options))
        with open(target, encoding="utf-8", newline="") as handle:
            return result, handle.read()

    def test_trim_and_header(self):
        result, output = self.repair(" Name ,Age\n ada , 30 \n")
        self.assertEqual(output.strip(), "name,age\nada,30")
        self.assertEqual(result.rows_written, 1)

    def test_pad_short_rows(self):
        _, output = self.repair("a,b,c\n1,2\n")
        self.assertEqual(output.strip().splitlines()[1], "1,2,")

    def test_drop_extra_cells(self):
        _, output = self.repair("a,b\n1,2,3\n", drop_extra_cells=True)
        self.assertEqual(output.strip().splitlines()[1], "1,2")

    def test_drop_duplicates(self):
        result, output = self.repair(
            "a,b\n1,x\n1,x\n2,y\n", drop_duplicate_rows=True
        )
        self.assertEqual(result.rows_written, 2)
        self.assertNotIn("1,x\ndefined", output)

    def test_null_unification(self):
        _, output = self.repair("a,b\n1,NA\n2,-\n", unify_nulls=True)
        self.assertEqual(output.strip().splitlines(), ["a,b", "1,", "2,"])

    def test_date_coercion(self):
        source = self.write("d.csv", "when\n05/01/2024\n06/02/2024\n")
        target = self.path("d-out.csv")
        types = infer_column_types(["when"], [["05/01/2024"], ["06/02/2024"]])
        self.assertEqual(set(types[0].dates.formats), {"%d/%m/%Y", "%m/%d/%Y"})
        apply_repair(
            source,
            target,
            options=RepairOptions(coerce_dates=True),
            column_types=types,
        )
        with open(target, encoding="utf-8") as handle:
            # ambiguous formats must be left alone rather than guessed
            self.assertIn("05/01/2024", handle.read())

    def test_iso_date_coercion(self):
        source = self.write("d.csv", "when\n2024-01-05\n2024-02-06\n")
        target = self.path("d-out.csv")
        types = infer_column_types(["when"], [["2024-01-05"], ["2024-02-06"]])
        apply_repair(
            source,
            target,
            options=RepairOptions(coerce_dates=True),
            column_types=types,
        )
        with open(target, encoding="utf-8") as handle:
            self.assertIn("2024-01-05", handle.read())

    def test_dry_run_writes_nothing(self):
        source = self.write("in.csv", " A \n 1 \n")
        target = self.path("out.csv")
        result = apply_repair(source, target, dry_run=True)
        self.assertFalse(os.path.exists(target))
        self.assertIsNone(result.output)
        self.assertEqual(result.plan.header, ["a"])

    def test_idempotent(self):
        source = self.write("in.csv", " Name ,Age\n ada , 30 \n\n")
        first = self.path("first.csv")
        second = self.path("second.csv")
        options = RepairOptions()
        apply_repair(source, first, options=options)
        result = apply_repair(first, second, options=options)
        self.assertFalse(result.plan.changed)

    def test_quotes_round_trip(self):
        _, output = self.repair('a,b\n"x,y",z\n')
        self.assertIn('"x,y"', output)

    def test_crlf_output(self):
        source = self.write("in.csv", "a,b\n1,2\n")
        target = self.path("out.csv")
        options = RepairOptions(line_terminator="\r\n")
        apply_repair(source, target, options=options)
        with open(target, "rb") as handle:
            self.assertIn(b"\r\n", handle.read())

    def test_delimiter_override(self):
        source = self.write("in.csv", "a;b\n1;2\n")
        target = self.path("out.csv")
        apply_repair(source, target, options=RepairOptions(delimiter=";"))
        with open(target, encoding="utf-8") as handle:
            self.assertIn("a;b", handle.read())

    def test_cp1252_input_utf8_output(self):
        source = self.path("in.csv")
        with open(source, "wb") as handle:
            handle.write("name\nJos\u00e9\n".encode("cp1252"))
        target = self.path("out.csv")
        result = apply_repair(source, target)
        self.assertEqual(result.rows_written, 1)
        with open(target, encoding="utf-8") as handle:
            self.assertIn("Jos\u00e9", handle.read())

    def test_collapse_whitespace(self):
        _, output = self.repair("a\n  x   y  \n", collapse_whitespace=True)
        self.assertEqual(output.strip().splitlines()[1], "x y")

    def test_result_json(self):
        source = self.write("in.csv", " A \n1\n")
        target = self.path("out.csv")
        payload = apply_repair(source, target).to_dict()
        self.assertEqual(payload["rows_read"], 1)
        self.assertEqual(payload["rows_written"], 1)
        self.assertIn("plan", payload)


class RuleParsingTests(unittest.TestCase):
    def test_minimal(self):
        spec = parse_rules({"columns": {"x": ["required"]}})
        self.assertEqual(spec.columns["x"][0]["rule"], "required")
        self.assertEqual(spec.columns["x"][0]["severity"], "error")

    def test_all_sections(self):
        spec = parse_rules(
            {
                "name": "demo",
                "columns": {"a": [{"rule": "range", "min": 1, "max": 5}]},
                "unique": ["id", ["a", "b"]],
                "required_columns": ["id"],
                "forbidden_columns": ["secret"],
                "min_rows": 1,
                "max_rows": 10,
            }
        )
        self.assertEqual(spec.name, "demo")
        self.assertEqual(spec.unique, [["id"], ["a", "b"]])
        self.assertEqual(spec.required_columns, ["id"])
        self.assertEqual(spec.forbidden_columns, ["secret"])
        self.assertEqual((spec.min_rows, spec.max_rows), (1, 10))

    def test_errors(self):
        cases = [
            ([], "JSON object"),
            ({"columns": []}, "must be"),
            ({"columns": {"x": [{"rule": "bogus"}]}}, "unknown rule"),
            ({"columns": {"x": [{"rule": "pattern", "value": "("}]}}, "bad pattern"),
            ({"columns": {"x": [{"rule": "allowed"}]}}, "values list"),
            ({"columns": {"x": [{"rule": "required", "severity": "fatal"}]}}, "severity"),
            ({"unique": "id"}, "list of column groups"),
            ({"min_rows": -1}, "non-negative"),
        ]
        for payload, fragment in cases:
            with self.assertRaises(RuleError) as caught:
                parse_rules(payload)
            self.assertIn(fragment, str(caught.exception))

    def test_load_rules_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "r.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"columns": {"a": ["required"]}}, handle)
            self.assertEqual(load_rules(path).columns["a"][0]["rule"], "required")

    def test_load_bad_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "r.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{")
            with self.assertRaises(RuleError):
                load_rules(path)
            with self.assertRaises(RuleError):
                load_rules(os.path.join(folder, "missing.json"))


class ValidateTests(unittest.TestCase):
    def spec(self, payload):
        return parse_rules(payload)

    def test_required(self):
        findings = validate_rows(
            ["a"], [["1"], [""]], self.spec({"columns": {"a": ["required"]}})
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].row, 2)

    def test_pattern(self):
        spec = self.spec({"columns": {"e": [{"rule": "pattern", "value": "@"}]}})
        findings = validate_rows(["e"], [["a@b.co"], ["nope"]], spec)
        self.assertEqual([f.rule for f in findings], ["pattern"])

    def test_range(self):
        spec = self.spec({"columns": {"n": [{"rule": "range", "min": 0, "max": 10}]}})
        findings = validate_rows(["n"], [["5"], ["-1"], ["11"], ["abc"]], spec)
        self.assertEqual(len(findings), 3)
        self.assertIn("not a number", findings[2].message)

    def test_allowed(self):
        spec = self.spec({"columns": {"t": [{"rule": "allowed", "values": ["a", "b"]}]}})
        findings = validate_rows(["t"], [["a"], ["c"]], spec)
        self.assertEqual(len(findings), 1)
        self.assertIn("not one of", findings[0].message)

    def test_lengths(self):
        spec = self.spec(
            {
                "columns": {
                    "s": [
                        {"rule": "min_length", "value": 2},
                        {"rule": "max_length", "value": 3},
                    ]
                }
            }
        )
        findings = validate_rows(["s"], [["a"], ["abcd"], ["ok"]], spec)
        self.assertEqual(sorted(f.rule for f in findings), ["max_length", "min_length"])

    def test_types(self):
        spec = self.spec(
            {
                "columns": {
                    "n": [{"rule": "type", "value": "integer"}],
                    "d": [{"rule": "type", "value": "date"}],
                    "e": [{"rule": "type", "value": "email"}],
                    "u": [{"rule": "type", "value": "url"}],
                    "b": [{"rule": "type", "value": "boolean"}],
                }
            }
        )
        findings = validate_rows(
            ["n", "d", "e", "u", "b"],
            [["1.5", "not-a-date", "plain", "no-scheme", "maybe"]],
            spec,
        )
        self.assertEqual(len(findings), 5)

    def test_warning_severity(self):
        spec = self.spec(
            {"columns": {"n": [{"rule": "range", "min": 0, "severity": "warning"}]}}
        )
        findings = validate_rows(["n"], [["-1"]], spec)
        self.assertEqual(findings[0].severity, "warning")

    def test_unique_group(self):
        spec = self.spec({"unique": [["a", "b"]]})
        findings = validate_rows(
            ["a", "b"], [["1", "x"], ["1", "x"], ["1", "y"]], spec
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("duplicate of row 1", findings[0].message)

    def test_unique_skips_all_null_keys(self):
        spec = self.spec({"unique": ["a"]})
        findings = validate_rows(["a"], [[""], [""]], spec)
        self.assertEqual(findings, [])

    def test_missing_required_column(self):
        spec = self.spec({"required_columns": ["id"]})
        findings = validate_rows(["a"], [["1"]], spec)
        self.assertEqual(findings[0].rule, "required_column")

    def test_forbidden_column(self):
        spec = self.spec({"forbidden_columns": ["password"]})
        findings = validate_rows(["password"], [["x"]], spec)
        self.assertEqual(findings[0].rule, "forbidden_column")

    def test_unknown_rule_column_warns(self):
        spec = self.spec({"columns": {"ghost": ["required"]}})
        findings = validate_rows(["a"], [["1"]], spec)
        self.assertEqual(findings[0].severity, "warning")

    def test_row_bounds(self):
        spec = self.spec({"min_rows": 2, "max_rows": 3})
        self.assertEqual(len(validate_rows(["a"], [["1"]], spec)), 1)
        self.assertEqual(len(validate_rows(["a"], [["1"], ["2"], ["3"], ["4"]], spec)), 1)

    def test_max_findings(self):
        spec = self.spec({"columns": {"a": ["required"]}})
        findings = validate_rows(["a"], [[""]] * 50, spec, max_findings=5)
        self.assertEqual(len(findings), 5)

    def test_line_numbers_follow_file(self):
        spec = self.spec({"columns": {"a": ["required"]}})
        findings = validate_rows(["a"], [["1"], [""]], spec, lines=[2, 9])
        self.assertEqual(findings[0].row, 9)

    def test_validate_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "t.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("id,name\n1,ada\n1,grace\n")
            spec = self.spec({"unique": ["id"]})
            header, findings, rows = validate_table(path, spec)
            self.assertEqual(header, ["id", "name"])
            self.assertEqual(rows, 2)
            self.assertEqual(len(findings), 1)


class CliTests(TempDirMixin, unittest.TestCase):
    def run_cli(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def setUp(self):
        super().setUp()
        self.messy = self.write(
            "messy.csv",
            "id,Full Name,MRR\n1, Ada ,1299.00\n2,Grace,899\n3,,499\n3,,499\n",
        )

    def test_profile_exit_zero_or_one(self):
        code, out, _ = self.run_cli("profile", self.messy)
        self.assertIn("4 rows", out)
        self.assertIn(code, (0, 1))

    def test_profile_json(self):
        code, out, _ = self.run_cli("profile", self.messy, "--json")
        payload = json.loads(out)
        self.assertEqual(payload["rows"], 4)
        self.assertIn("recommendations", payload)
        self.assertIn(code, (0, 1, 2))

    def test_profile_strict_promotes_warnings(self):
        code, _, _ = self.run_cli("profile", self.messy, "--json")
        strict_code, _, _ = self.run_cli("profile", self.messy, "--json", "--strict")
        self.assertEqual(code, 1)
        self.assertEqual(strict_code, 2)

    def test_profile_quiet(self):
        path = self.write("clean.csv", "a\n1\n2\n")
        code, out, _ = self.run_cli("profile", path, "--quiet")
        self.assertEqual(code, 0)
        self.assertEqual(out, "")

    def test_profile_missing_file(self):
        code, _, err = self.run_cli("profile", self.path("nope.csv"))
        self.assertEqual(code, 2)
        self.assertIn("no such file", err)

    def test_profile_missing_file_json(self):
        code, _, err = self.run_cli("profile", self.path("nope.csv"), "--json")
        self.assertEqual(code, 2)
        self.assertIn("error", json.loads(err))

    def test_sniff(self):
        code, out, _ = self.run_cli("sniff", self.messy)
        self.assertEqual(code, 0)
        self.assertIn("delimiter", out)

    def test_sniff_json(self):
        code, out, _ = self.run_cli("sniff", self.messy, "--json")
        payload = json.loads(out)
        self.assertEqual(payload["delimiter"], ",")
        self.assertIn("bom", payload)

    def test_semicolon_detection(self):
        path = self.write("semi.csv", "a;b\n1;2\n")
        code, out, _ = self.run_cli("sniff", path, "--json")
        self.assertEqual(json.loads(out)["delimiter"], ";")

    def test_repair_dry_run(self):
        code, out, _ = self.run_cli(
            "repair", self.messy, "--out", self.path("out.csv"), "--dry-run"
        )
        self.assertEqual(code, 0)
        self.assertIn("would repair", out)
        self.assertFalse(os.path.exists(self.path("out.csv")))

    def test_repair_writes(self):
        target = self.path("out.csv")
        code, out, _ = self.run_cli(
            "repair", self.messy, "--out", target, "--drop-duplicates", "--json"
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["rows_written"], 3)
        with open(target, encoding="utf-8") as handle:
            self.assertTrue(handle.read().startswith("id,full_name,mrr"))

    def test_repair_without_flags_keeps_values(self):
        target = self.path("out.csv")
        self.run_cli("repair", self.messy, "--out", target, "--no-trim")
        with open(target, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("1299.00", text)
        self.assertIn(" Ada ", text)

    def test_validate_clean(self):
        path = self.write("good.csv", "id,name\n1,ada\n")
        rules = self.write_json("r.json", {"columns": {"id": ["required"]}})
        code, out, _ = self.run_cli("validate", path, "--rules", rules)
        self.assertEqual(code, 0)
        self.assertIn("ok:", out)

    def test_validate_errors(self):
        rules = self.write_json("r.json", {"unique": ["id"]})
        code, out, _ = self.run_cli("validate", self.messy, "--rules", rules)
        self.assertIn("error:", out)
        self.assertEqual(code, 2)

    def test_validate_json_shape(self):
        rules = self.write_json("r.json", {"required_columns": ["id"]})
        code, out, _ = self.run_cli("validate", self.messy, "--rules", rules, "--json")
        payload = json.loads(out)
        self.assertEqual(payload["rows"], 4)
        self.assertIn("findings", payload)

    def test_validate_bad_rules(self):
        rules = self.write_json("r.json", {"columns": {"a": [{"rule": "bogus"}]}})
        code, _, err = self.run_cli("validate", self.messy, "--rules", rules)
        self.assertEqual(code, 2)
        self.assertIn("unknown rule", err)

    def test_validate_fail_on_warning(self):
        rules = self.write_json(
            "r.json",
            {"columns": {"MRR": [{"rule": "range", "max": 1000, "severity": "warning"}]}},
        )
        code, _, _ = self.run_cli(
            "validate", self.messy, "--rules", rules, "--fail-on", "warning"
        )
        self.assertEqual(code, 1)

    def test_no_header(self):
        path = self.write("raw.csv", "1,2\n3,4\n")
        code, out, _ = self.run_cli("profile", path, "--no-header", "--json")
        payload = json.loads(out)
        self.assertEqual(payload["columns"], 2)

    def test_version_flag(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer), self.assertRaises(SystemExit) as caught:
            main(["--version"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("csv-doctor", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
