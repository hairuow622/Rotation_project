#!/usr/bin/env python3
"""Convert a BED-like file to sorted BED3 format."""

import argparse
import re
from pathlib import Path


def chromosome_key(chromosome: str) -> tuple:
    """Return a natural genomic sort key (chr1 ... chr22, chrX, chrY, chrM)."""
    name = chromosome.removeprefix("chr")

    if name.isdigit():
        return (0, int(name), ())

    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if name.upper() in special:
        return (0, special[name.upper()], ())

    natural_name = tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    )
    return (1, 0, natural_name)


def convert_to_bed3(input_path: Path, output_path: Path) -> int:
    records = []

    with input_path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "track", "browser")):
                continue

            fields = stripped.split()
            if len(fields) < 3:
                raise ValueError(
                    f"{input_path}:{line_number}: expected at least 3 columns"
                )

            chromosome = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"{input_path}:{line_number}: start and end must be integers"
                ) from error

            if start < 0 or end < start:
                raise ValueError(
                    f"{input_path}:{line_number}: invalid interval {start}-{end}"
                )

            records.append((chromosome, start, end))

    records.sort(key=lambda record: (chromosome_key(record[0]), record[1], record[2]))

    with output_path.open("w") as output_file:
        for chromosome, start, end in records:
            output_file.write(f"{chromosome}\t{start}\t{end}\n")

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep the first three BED columns and sort genomic intervals."
    )
    parser.add_argument("input", type=Path, help="input BED or narrowPeak file")
    parser.add_argument("output", type=Path, help="output BED3 file")
    args = parser.parse_args()

    count = convert_to_bed3(args.input, args.output)
    print(f"Wrote {count} intervals to {args.output}")


if __name__ == "__main__":
    main()
