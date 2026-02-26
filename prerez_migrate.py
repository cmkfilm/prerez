#!/usr/bin/env python3
"""
prerez_migrate.py — Add project prefix to existing ground_truth.tsv rows

Run this ONCE when upgrading to multi-project ground truth support.
Backs up the original file before modifying.

Usage:
    # Dry run — see what will change
    python3 prerez_migrate.py --project wwwars --dry-run

    # Apply
    python3 prerez_migrate.py --project wwwars

    # Custom GT path
    python3 prerez_migrate.py --project wwwars --gt ~/ground_truth.tsv

Before:
    file            native_res
    _V1-0001.mov    360
    _V1-0002.mov    1080

After:
    file                    native_res
    wwwars/_V1-0001.mov     360
    wwwars/_V1-0002.mov     1080

For subsequent projects, new rows are written with the project prefix
automatically by prerez_extract.py (via --project flag). This script
is only needed to migrate existing unprefixed rows.
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Migrate ground_truth.tsv to multi-project prefix format"
    )
    ap.add_argument("--project", required=True,
                    help="Project name to prefix existing rows with (e.g. 'wwwars')")
    ap.add_argument("--gt", default="~/ground_truth.tsv",
                    help="Path to ground truth TSV (default: ~/ground_truth.tsv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing")
    args = ap.parse_args()

    gt_path = Path(args.gt).expanduser().resolve()
    if not gt_path.exists():
        print(f"ERROR: {gt_path} not found")
        return

    prefix = args.project.replace("/", "_")
    lines = gt_path.read_text(encoding="utf-8").splitlines()

    if not lines:
        print("ERROR: file is empty")
        return

    header = lines[0]
    rows = lines[1:]

    # Check if already prefixed
    already_prefixed = sum(1 for r in rows if "\t" in r and "/" in r.split("\t")[0])
    unprefixed = sum(1 for r in rows if "\t" in r and "/" not in r.split("\t")[0])

    print(f"Ground truth: {gt_path}")
    print(f"Total rows:   {len(rows)}")
    print(f"  Already prefixed: {already_prefixed}")
    print(f"  Unprefixed:       {unprefixed}  ← will add '{prefix}/'")

    if unprefixed == 0:
        print("\nNothing to do — all rows already have a project prefix.")
        return

    # Show sample
    sample_unprefixed = [r for r in rows if "\t" in r and "/" not in r.split("\t")[0]][:3]
    print(f"\nSample rows to be updated:")
    for r in sample_unprefixed:
        parts = r.split("\t")
        print(f"  {parts[0]!r:30s} → {prefix}/{parts[0]!r}")

    if args.dry_run:
        print("\n(dry run — no files written)")
        return

    # Back up original
    backup = gt_path.with_suffix(".tsv.bak")
    shutil.copy2(gt_path, backup)
    print(f"\nBacked up to: {backup}")

    # Rewrite with prefix on unprefixed rows
    new_rows = []
    for row in rows:
        if not row.strip():
            new_rows.append(row)
            continue
        parts = row.split("\t")
        if "/" not in parts[0]:
            parts[0] = f"{prefix}/{parts[0]}"
        new_rows.append("\t".join(parts))

    output = "\n".join([header] + new_rows) + "\n"
    gt_path.write_text(output, encoding="utf-8")

    print(f"Updated {unprefixed} rows with prefix '{prefix}/'")
    print(f"\nDone. New rows for future projects will be prefixed automatically.")
    print(f"Use --project <name> when running prerez.py on each project.")


if __name__ == "__main__":
    main()
