"""Validate challenge TSV outputs without external dependencies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["entity_id"] for row in csv.DictReader(stream, delimiter="\t")}


def read_output(path: Path, id_column: str, value_column: str) -> tuple[dict[str, list[str]], list[str]]:
    issues: list[str] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != [id_column, value_column]:
            issues.append(f"{path}: expected columns {[id_column, value_column]}, got {reader.fieldnames}")
            return {}, issues
        result: dict[str, list[str]] = {}
        for line, row in enumerate(reader, start=2):
            source_id = row[id_column]
            values = [] if not row[value_column] else row[value_column].split(",")
            if source_id in result:
                issues.append(f"{path}:{line}: duplicate Source 1 ID {source_id}")
            if len(values) != len(set(values)):
                issues.append(f"{path}:{line}: duplicated entity ID in list")
            result[source_id] = values
    return result, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    args = parser.parse_args()

    source1 = read_ids(args.test_dir / "test_source1.tsv")
    allowed = read_ids(args.test_dir / "test_source2.tsv") | read_ids(args.test_dir / "test_source3.tsv")
    matches, issues = read_output(args.matching, "source1_entity_id", "matched_entity_ids")
    candidates, candidate_issues = read_output(args.candidate, "source1_entity_id", "candidate_entity_ids")
    issues.extend(candidate_issues)
    for label, output in (("matching", matches), ("candidate", candidates)):
        missing, unexpected = source1 - set(output), set(output) - source1
        if missing: issues.append(f"{label}: missing {len(missing)} Source 1 rows")
        if unexpected: issues.append(f"{label}: contains {len(unexpected)} unknown Source 1 rows")
        for source_id, ids in output.items():
            invalid = set(ids) - allowed
            if invalid: issues.append(f"{label}: {source_id} has invalid Source 2/3 IDs: {sorted(invalid)[:3]}")
    for source_id, ids in matches.items():
        if not set(ids).issubset(set(candidates.get(source_id, []))):
            issues.append(f"matching: {source_id} has matches absent from candidates")
    if issues:
        raise SystemExit("FAIL\n" + "\n".join(f"- {issue}" for issue in issues))
    print("PASS")


if __name__ == "__main__":
    main()
