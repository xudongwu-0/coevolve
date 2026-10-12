"""Structural diversity and leakage checks for the larger teacher campaign."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .common import file_sha, read_json, write_json
from .retool import INITIAL_SOURCE
from .rewrite_data import EditExample, digest, parse_document, validate_example


def audit_rows(rows: list[dict], targets: dict, *, full_budget: bool) -> dict:
    if not rows:
        raise ValueError("Cannot audit an empty teacher dataset")
    counts, formats, families = Counter(), Counter(), defaultdict(set)
    schemas, source_contexts, seen_ids, seen_samples = {}, defaultdict(set), set(), set()
    collisions, duplicates = [], []
    string_edits = h0_count = retool_count = 0
    h0_values, _ = parse_document(INITIAL_SOURCE, "python")
    for row in rows:
        item = EditExample.model_validate(row)
        validate_example(item)
        before, declaration = parse_document(item.source, item.format)
        counts[item.split] += 1
        formats[item.format] += 1
        families[item.family].add(item.split)
        context = {"format": item.format, "declaration": declaration, "values": before}
        source_contexts[item.split].add(digest(context))
        sample = digest({"context": context, "instruction": item.instruction.strip()})
        if item.id in seen_ids or sample in seen_samples:
            duplicates.append(item.id)
        seen_ids.add(item.id)
        seen_samples.add(sample)
        # Ignoring module/class names and literal values also catches renamed copies.
        schema = digest({"format": item.format,
                         "fields": sorted((key, type(value).__name__) for key, value in before.items())})
        if schema in schemas and schemas[schema][0] != item.split:
            collisions.append({"first_id": schemas[schema][1], "second_id": item.id})
        else:
            schemas.setdefault(schema, (item.split, item.id))
        if item.split == "train":
            string_edits += any(isinstance(value, str) for value in item.changes.values())
            retool_like = set(h0_values).issubset(before) or "ReToolEnv" in declaration
            retool_count += retool_like
            h0_count += retool_like and all(
                key in before and type(before[key]) is type(value) and before[key] == value
                for key, value in h0_values.items()
            )
    train_count = counts["train"]
    if train_count < 1:
        raise ValueError("A campaign training snapshot needs at least one training example")
    errors = []
    leaking_families = sorted(name for name, splits in families.items() if len(splits) > 1)
    if duplicates:
        errors.append("Duplicate identifiers or canonical instruction/source pairs")
    if leaking_families:
        errors.append("Task family overlaps between splits")
    if collisions:
        errors.append("Source-field schema overlaps between splits")
    if h0_count / train_count > targets["harness0_max_train_fraction"]:
        errors.append("harness0 training fraction exceeds declared limit")
    if retool_count / train_count > targets["retool_family_max_train_fraction"]:
        errors.append("ReTool-family training fraction exceeds declared limit")
    if string_edits / train_count < 0.5:
        errors.append("Fewer than half of training records edit instruction/string content")
    if full_budget:
        expected = {"train": targets["train_valid_teacher_records"],
                    "dev": targets["dev_examples"], "test": targets["test_examples"]}
        if dict(counts) != expected:
            errors.append("Full campaign split counts differ from the declared targets")
        if len(source_contexts["train"]) < targets["minimum_independent_training_contexts"]:
            errors.append("Too few distinct canonical source contexts")
    return {
        "passed_structural_checks": not errors, "full_budget": full_budget,
        "counts": dict(counts), "formats": dict(formats), "errors": errors,
        "family_counts": {split: sum(split in value for value in families.values())
                          for split in ("train", "dev", "test")},
        "unique_canonical_sources": {split: len(value) for split, value in source_contexts.items()},
        "train_string_edit_fraction": string_edits / train_count,
        "harness0_train_fraction": h0_count / train_count,
        "retool_train_fraction": retool_count / train_count,
        "duplicate_ids_or_samples": duplicates, "leaking_families": leaking_families,
        "cross_split_schema_collisions": collisions,
        "limitations": [
            "Canonical-source diversity is not proof of independent semantic tasks.",
            "Renamed field schemas or paraphrased templates can evade structural deduplication.",
            "String edits are not automatically meaningful semantic rewrites.",
            "Independent content review and unseen-harness transfer evaluation are still required.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true",
                        help="Check a partial production batch, not the full target counts")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh immutable audit result")
    manifest = read_json(args.data.parent / "manifest.json")
    sha = file_sha(args.data)
    if sha != manifest["examples_sha256"]:
        raise ValueError("Teacher examples differ from the assembled manifest")
    result = audit_rows(read_json(args.data), read_json(args.campaign)["targets"],
                        full_budget=not args.calibration)
    result.update(data_sha256=sha, campaign_sha256=file_sha(args.campaign))
    write_json(args.output, result)
    if result["errors"]:
        raise ValueError(f"Teacher campaign audit failed: {result['errors']}")
    print({key: value for key, value in result.items()
           if key not in {"cross_split_schema_collisions", "limitations"}})


if __name__ == "__main__":
    main()
