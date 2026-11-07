"""Value parsing, type inference and dialect detection."""

import os
import tempfile
import unittest

from csvdoctor.dialect import (
    Dialect,
    MalformedCsv,
    TableReader,
    load_table,
    read_text,
    sniff,
)
from csvdoctor.values import (
    detect_date_formats,
    format_number,
    infer_column_type,
    infer_kind,
    infer_number_style,
    is_boolean_token,
    is_null,
    is_percent,
    looks_like_header,
    normalize_column_name,
    normalize_date,
    parse_boolean,
    parse_number,
    parse_temporal,
    unique_column_names,
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


class NullTokenTests(unittest.TestCase):
    def test_common_spellings(self):
        for token in ("", " ", "null", "NULL", "None", "n/a", "N/A", "nan", "-", "?", "NaN"):
            self.assertTrue(is_null(token), token)

    def test_values_are_not_null(self):
        for token in ("0", "no", "none of the above", "n/a-ish"):
            self.assertFalse(is_null(token), token)


class BooleanTests(unittest.TestCase):
    def test_parse(self):
        self.assertIs(parse_boolean("YES"), True)
        self.assertIs(parse_boolean("t"), True)
        self.assertIs(parse_boolean("No"), False)
        self.assertIsNone(parse_boolean("maybe"))

    def test_token_detection(self):
        self.assertTrue(is_boolean_token(" y "))
        self.assertFalse(is_boolean_token("yep"))


class NumberTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_number("42"), 42.0)
        self.assertEqual(parse_number("-3.5"), -3.5)
        self.assertEqual(parse_number("+7"), 7.0)

    def test_grouping(self):
        self.assertEqual(parse_number("1,234"), 1234.0)
        self.assertEqual(parse_number("1,234.56"), 1234.56)
        self.assertEqual(parse_number("1.234,56"), 1234.56)

    def test_style_overrides(self):
        self.assertEqual(parse_number("1.234", style="eu"), 1234.0)
        self.assertEqual(parse_number("1.234", style="us"), 1.234)
        self.assertEqual(parse_number("1,234", style="eu"), 1.234)

    def test_currency_and_percent(self):
        self.assertEqual(parse_number("$1,000.50"), 1000.5)
        self.assertEqual(parse_number("12.5%"), 12.5)
        self.assertTrue(is_percent("3%"))
        self.assertFalse(is_percent("3"))

    def test_accounting_negatives(self):
        self.assertEqual(parse_number("(42)"), -42.0)
        self.assertEqual(parse_number("($1,200)"), -1200.0)

    def test_unicode_minus_and_nbsp(self):
        self.assertEqual(parse_number("\u22125"), -5.0)
        self.assertEqual(parse_number("1\u00a0000"), 1000.0)

    def test_rejects_non_numbers(self):
        for token in ("abc", "1.2.3.4", "--", "1-2", ""):
            self.assertIsNone(parse_number(token), token)

    def test_single_separator_heuristic(self):
        self.assertEqual(parse_number("12,34"), 12.34)
        self.assertEqual(parse_number("1,234"), 1234.0)
        self.assertEqual(parse_number("0.500"), 0.5)

    def test_style_vote(self):
        self.assertEqual(infer_number_style(["1.234,56", "12,5"]), "eu")
        self.assertEqual(infer_number_style(["1,234.56", "12.5"]), "us")
        self.assertEqual(infer_number_style(["1", "2"]), "auto")

    def test_format_number(self):
        self.assertEqual(format_number(1234.0), "1234")
        self.assertEqual(format_number(1.5), "1.5")
        self.assertEqual(format_number(-0.0), "0")


class DateTests(unittest.TestCase):
    def test_single_format(self):
        found = detect_date_formats(["2024-01-05", "2024-02-06"])
        self.assertEqual(found.formats, ("%Y-%m-%d",))
        self.assertEqual(found.format, "%Y-%m-%d")
        self.assertFalse(found.ambiguous)
        self.assertFalse(found.datetime)

    def test_ambiguity(self):
        found = detect_date_formats(["01/02/2024", "03/04/2024"])
        self.assertTrue(found.ambiguous)
        self.assertIsNone(found.format)
        self.assertIn("%d/%m/%Y", found.formats)
        self.assertIn("%m/%d/%Y", found.formats)

    def test_named_months(self):
        self.assertEqual(detect_date_formats(["5 Jan 2024", "6 Feb 2024"]).format, "%d %b %Y")

    def test_datetime_column(self):
        found = detect_date_formats(["2024-01-05T10:00:00", "2024-01-06T11:30:00"])
        self.assertTrue(found.datetime)
        self.assertEqual(found.format, "%Y-%m-%dT%H:%M:%S")

    def test_iso_offset(self):
        found = detect_date_formats(["2024-01-05T10:00:00+02:00"])
        self.assertEqual(found.format, "iso")
        self.assertTrue(found.datetime)

    def test_not_a_date(self):
        self.assertEqual(detect_date_formats(["hello", "world"]).formats, ())
        self.assertEqual(detect_date_formats([]).formats, ())

    def test_normalize(self):
        self.assertEqual(normalize_date("05/01/2024", "%d/%m/%Y"), "2024-01-05")
        self.assertIsNone(normalize_date("nope", "%d/%m/%Y"))
        self.assertEqual(normalize_date("bad", "%d/%m/%Y", datetime_ok=False), None)

    def test_parse_temporal_iso(self):
        parsed = parse_temporal("2024-01-05T10:00:00Z")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2024)
        self.assertIsNone(parse_temporal(""))


class KindTests(unittest.TestCase):
    def test_narrowest_kind(self):
        cases = [
            (["1", "2"], "integer"),
            (["1", "2.5"], "decimal"),
            (["yes", "no"], "boolean"),
            (["a@b.co", "c@d.org"], "email"),
            (["https://a.b", "http://c.d"], "url"),
            (["2024-01-05"], "date"),
            (["2024-01-05T10:00:00"], "datetime"),
            (["x", "1"], "text"),
            ([""], "null"),
            (["n/a", "null"], "null"),
        ]
        for values, expected in cases:
            self.assertEqual(infer_kind(values), expected, values)

    def test_uuid(self):
        value = "123e4567-e89b-12d3-a456-426614174000"
        self.assertEqual(infer_kind([value]), "uuid")

    def test_grouped_numbers_are_integral(self):
        self.assertEqual(infer_kind(["1,234", "5,678"]), "integer")

    def test_decimals_with_zero_tail(self):
        self.assertEqual(infer_kind(["1299.00", "5"]), "integer")

    def test_number_style_propagates(self):
        column = infer_column_type(["1.234,56", "12,50"])
        self.assertEqual(column.kind, "decimal")
        self.assertEqual(column.number_style, "eu")
        self.assertEqual(column.to_dict()["number_style"], "eu")


class HeaderTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_column_name("  First Name (billing) "), "first_name_billing")
        self.assertEqual(normalize_column_name(""), "column")
        self.assertEqual(normalize_column_name("2fast"), "column_2fast")

    def test_unique(self):
        self.assertEqual(
            unique_column_names(["Name", "name", "E-mail"]),
            ["name", "name_2", "e_mail"],
        )
        self.assertEqual(unique_column_names(["", "  "]), ["column_1", "column_2"])

    def test_header_detection(self):
        self.assertTrue(looks_like_header(["id", "email"], ["1", "a@b.co"]))
        self.assertFalse(looks_like_header(["1", "2"], ["3", "4"]))
        self.assertFalse(looks_like_header(["", ""], ["1", "2"]))
        self.assertTrue(looks_like_header(["id", "email"], []))


class SniffTests(TempDirMixin, unittest.TestCase):
    def test_comma(self):
        self.assertEqual(sniff("a,b,c\n1,2,3\n").delimiter, ",")

    def test_semicolon(self):
        self.assertEqual(sniff("a;b;c\n1;2;3\n").delimiter, ";")

    def test_tab(self):
        self.assertEqual(sniff("a\tb\tc\n1\t2\t3\n").delimiter, "\t")

    def test_pipe(self):
        self.assertEqual(sniff("a|b|c\n1|2|3\n").delimiter, "|")

    def test_delimiter_inside_quotes_is_ignored(self):
        found = sniff('a;b\n"x;y";z\n')
        self.assertEqual(found.delimiter, ";")

    def test_single_quote(self):
        found = sniff("a,b\n'x,y',z\n")
        self.assertEqual(found.quotechar, "'")

    def test_line_endings(self):
        self.assertEqual(sniff("a,b\r\n1,2\r\n").lineterminator, "\r\n")
        self.assertEqual(sniff("a,b\n1,2\n").lineterminator, "\n")

    def test_header_guess(self):
        self.assertTrue(sniff("id,amount\n1,10\n").has_header)
        self.assertFalse(sniff("1,2\n3,4\n").has_header)

    def test_empty(self):
        found = sniff("")
        self.assertEqual(found.delimiter, ",")


class ReaderTests(TempDirMixin, unittest.TestCase):
    def test_reads_rows_with_line_numbers(self):
        path = self.write("t.csv", "name,score\nada,3\ngrace,4\n")
        with TableReader(path) as table:
            self.assertEqual(table.header, ["name", "score"])
            rows = list(table.rows())
        self.assertEqual([fields for _, fields in rows], [["ada", "3"], ["grace", "4"]])
        self.assertEqual([line for line, _ in rows], [2, 3])

    def test_no_header(self):
        path = self.write("t.csv", "1,2\n3,4\n")
        dialect = Dialect(",", '"', has_header=False)
        with TableReader(path, dialect) as table:
            self.assertEqual(table.header, ["column_1", "column_2"])
            self.assertEqual(len(list(table.rows())), 2)

    def test_ragged_rows_are_preserved(self):
        path = self.write("t.csv", "a,b,c\n1,2\n1,2,3,4\n")
        _, header, rows, _ = load_table(path)
        self.assertEqual(len(header), 3)
        self.assertEqual(rows[0], ["1", "2"])
        self.assertEqual(rows[1], ["1", "2", "3", "4"])

    def test_bom_is_stripped(self):
        target = self.path("bom.csv")
        with open(target, "wb") as handle:
            handle.write(b"\xef\xbb\xbfa,b\n1,2\n")
        text, encoding, had_bom = read_text(target)
        self.assertTrue(had_bom)
        self.assertEqual(encoding, "utf-8-sig")
        self.assertTrue(text.startswith("a,b"))
        _, header, _, _ = load_table(target)
        self.assertEqual(header, ["a", "b"])

    def test_cp1252_fallback(self):
        target = self.path("latin.csv")
        with open(target, "wb") as handle:
            handle.write("name,city\nJos\u00e9,M\u00fcnchen\n".encode("cp1252"))
        text, encoding, _ = read_text(target)
        self.assertEqual(encoding, "cp1252")
        self.assertIn("Jos\u00e9", text)

    def test_explicit_encoding_wins(self):
        target = self.path("latin.csv")
        with open(target, "wb") as handle:
            handle.write("name\nJos\u00e9\n".encode("cp1252"))
        _, header, rows, _ = load_table(target, Dialect(",", '"', "cp1252"))
        self.assertEqual(rows[0], ["Jos\u00e9"])

    def test_missing_file(self):
        with self.assertRaises(MalformedCsv):
            TableReader(self.path("nope.csv")).__enter__()

    def test_empty_file(self):
        path = self.write("empty.csv", "")
        with TableReader(path) as table:
            self.assertEqual(table.header, [])
            self.assertEqual(list(table.rows()), [])

    def test_crlf_and_quoted_newlines(self):
        path = self.write("q.csv", 'id,note\r\n1,"line one\nline two"\r\n')
        _, header, rows, _ = load_table(path)
        self.assertEqual(header, ["id", "note"])
        self.assertEqual(rows[0][1], "line one\nline two")

    def test_limit_stops_early(self):
        path = self.write("t.csv", "a\n1\n2\n3\n")
        _, _, rows, _ = load_table(path, limit=2)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
