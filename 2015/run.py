#!/usr/bin/env python3
"""
Advent of Code runner with pretty formatting and statistics.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import subprocess
import sys


@dataclass
class Result:
    year: int
    day: int
    part: int
    answer: str
    time_ns: int


def format_time(ns: int, pretty: bool = False) -> str:
    """Format nanoseconds to appropriate unit."""
    if not pretty:
        return str(ns)

    if ns < 1_000:
        return f"{ns:>8.0f} ns"
    elif ns < 1_000_000:
        return f"{ns/1_000:>8.2f} µs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:>8.2f} ms"
    else:
        return f"{ns/1_000_000_000:>8.2f}  s"


def run_python(filepath: Path) -> list[str] | None:
    """Run a Python solution."""
    cmd = ["python", str(filepath)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None
        return result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, Exception):
        return None


def run_rust(filepath: Path, base: str) -> list[str] | None:
    """Compile and run a Rust solution."""
    exe = f"/tmp/{base}"
    compile = subprocess.run(
        ["rustc", "-O", str(filepath), "-o", exe], capture_output=True
    )
    if compile.returncode != 0:
        return None

    try:
        result = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None
        return result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, Exception):
        return None


def run_zig(filepath: Path, base: str) -> list[str] | None:
    """Compile and run a Zig solution."""
    exe = f"./{base}"
    compile = subprocess.run(
        ["zig", "build-exe", str(filepath), "-O", "ReleaseFast", "--name", base],
        capture_output=True,
        stderr=subprocess.DEVNULL,
    )
    if compile.returncode != 0:
        return None

    try:
        result = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None
        return result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, Exception):
        return None
    finally:
        # Clean up compiled files
        Path(exe).unlink(missing_ok=True)
        Path(f"{base}.o").unlink(missing_ok=True)


def parse_output(lines: list[str]) -> list[Result]:
    """Parse tab-separated output into Result objects."""
    results = []
    for line in lines:
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 5:
            results.append(
                Result(
                    year=int(parts[0]),
                    day=int(parts[1]),
                    part=int(parts[2]),
                    answer=parts[3],
                    time_ns=int(parts[4]),
                )
            )
    return results


def run_solution(year: int, day: int) -> list[Result] | None:
    """Run a solution and parse its output."""
    padded = f"{day:02d}"
    base = f"{year}-{padded}"

    # Check which file exists and run with appropriate runner
    py_file = Path(f"{base}.py")
    rs_file = Path(f"{base}.rs")
    zig_file = Path(f"{base}.zig")

    output = None
    if py_file.exists():
        output = run_python(py_file)
    elif rs_file.exists():
        output = run_rust(rs_file, base)
    elif zig_file.exists():
        output = run_zig(zig_file, base)

    if output is None:
        return None

    return parse_output(output)


def print_results(all_results: list[Result], pretty: bool = False, stats: bool = False):
    """Print results with optional formatting."""
    if not all_results:
        return

    if pretty:
        # Calculate column widths
        year_width = max(len(str(r.year)) for r in all_results)
        year_width = max(year_width, len("Year"))

        day_width = max(len(str(r.day)) for r in all_results)
        day_width = max(day_width, len("Day"))

        part_width = max(len(str(r.part)) for r in all_results)
        part_width = max(part_width, len("Part"))

        answer_width = max(len(str(r.answer)) for r in all_results)
        answer_width = max(answer_width, len("Answer"))

        time_width = 11  # For formatted time strings

        # Print header
        print(
            f"{'Year':<{year_width}} │ {'Day':<{day_width}} │ {'Part':<{part_width}} │ {'Answer':<{answer_width}} │ {'Time':>{time_width}}"
        )
        print(
            f"{'─' * year_width}─┼─{'─' * day_width}─┼─{'─' * part_width}─┼─{'─' * answer_width}─┼─{'─' * time_width}"
        )

        # Print results
        for r in all_results:
            time_str = format_time(r.time_ns, pretty=True)
            print(
                f"{r.year:<{year_width}} │ {r.day:<{day_width}} │ {r.part:<{part_width}} │ {r.answer:<{answer_width}} │ {time_str}"
            )

        if stats:
            total_time = sum(r.time_ns for r in all_results)
            avg_time = total_time // len(all_results)
            min_time = min(r.time_ns for r in all_results)
            max_time = max(r.time_ns for r in all_results)

            # Calculate total width for footer (including separators)
            total_label_width = (
                year_width + 3 + day_width + 3 + part_width + 3 + answer_width
            )

            print(
                f"{'─' * year_width}─┴─{'─' * day_width}─┴─{'─' * part_width}─┴─{'─' * answer_width}─┼─{'─' * time_width}"
            )
            print(
                f"{'Total':<{total_label_width}} │ {format_time(total_time, pretty=True)}"
            )
            print()
            print("Statistics:")
            print(f"  Total time: {format_time(total_time, pretty=True)}")
            print(f"  Average:    {format_time(avg_time, pretty=True)}")
            print(f"  Min:        {format_time(min_time, pretty=True)}")
            print(f"  Max:        {format_time(max_time, pretty=True)}")
            print(f"  Count:      {len(all_results)} parts")
    else:
        # Simple tab-separated output
        print("Year\tDay\tPart\tAnswer\tTime (ns)")
        for r in all_results:
            print(f"{r.year}\t{r.day}\t{r.part}\t{r.answer}\t{r.time_ns}")

        if stats:
            total_time = sum(r.time_ns for r in all_results)
            print(f"\nTotal time: {format_time(total_time, pretty=False)} ns")


def main():
    parser = argparse.ArgumentParser(description="Run Advent of Code solutions")
    parser.add_argument("year", type=int, help="Year to run")
    parser.add_argument(
        "day", type=int, nargs="?", help="Specific day to run (optional)"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty print output")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    all_results = []

    if args.day:
        # Run single day
        results = run_solution(args.year, args.day)
        if results:
            all_results.extend(results)
        else:
            print(f"No solution found for {args.year}-{args.day:02d}", file=sys.stderr)
            sys.exit(1)
    else:
        # Run all days for the year
        # First pass to count available days
        available_days = []
        for day in range(1, 26):
            padded = f"{day:02d}"
            base = f"{args.year}-{padded}"
            if (
                Path(f"{base}.py").exists()
                or Path(f"{base}.rs").exists()
                or Path(f"{base}.zig").exists()
            ):
                available_days.append(day)

        total = len(available_days)
        for day in available_days:
            print(
                f"\rRunning day {day} of {total}", end="", flush=True, file=sys.stderr
            )
            results = run_solution(args.year, day)
            if results:
                all_results.extend(results)

        # Clear progress line
        if all_results:
            print("\r" + " " * 30 + "\r", end="", file=sys.stderr)

    if all_results:
        print_results(all_results, pretty=args.pretty, stats=args.stats)
    else:
        print(f"No solutions found for year {args.year}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
