"""csv-doctor: diagnose and repair messy CSV files.

>>> from csvdoctor.profile import profile_table
>>> profile = profile_table(["a"], [["1"]])
>>> profile.column("a").kind
'integer'
"""

from __future__ import annotations

from .checks import Recommendation, check_profile, exit_code
from .dialect import Dialect, MalformedCsv, TableReader, load_table, read_text, sniff
from .profile import (
    ColumnProfile,
    LengthStats,
    NumericStats,
    Outlier,
    ProfileOptions,
    TableProfile,
    profile_table,
)
from .repair import (
    RepairAction,
    RepairOptions,
    RepairPlan,
    RepairResult,
    apply_repair,
    plan_repair,
)
from .rules import (
    Finding,
    RuleError,
    RuleSpec,
    load_rules,
    parse_rules,
    validate_rows,
    validate_table,
)
from .values import ColumnType, infer_column_type, infer_kind, parse_number

__version__ = "1.0.0"

__all__ = [
    "Dialect",
    "MalformedCsv",
    "TableReader",
    "load_table",
    "read_text",
    "sniff",
    "ColumnProfile",
    "NumericStats",
    "LengthStats",
    "Outlier",
    "ProfileOptions",
    "TableProfile",
    "profile_table",
    "RepairAction",
    "RepairOptions",
    "RepairPlan",
    "RepairResult",
    "apply_repair",
    "plan_repair",
    "Finding",
    "RuleError",
    "RuleSpec",
    "load_rules",
    "parse_rules",
    "validate_rows",
    "validate_table",
    "ColumnType",
    "infer_column_type",
    "infer_kind",
    "parse_number",
    "Recommendation",
    "check_profile",
    "exit_code",
    "__version__",
]
