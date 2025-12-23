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
    language: str


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
        stdout=subprocess.DEVNULL,
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


def run_nim(filepath: Path) -> list[str] | None:
    """Compile and run a Nim solution."""
    try:
        result = subprocess.run(
            ["nim", "r", "-d:release", str(filepath)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, Exception):
        return None


def parse_output(lines: list[str], language: str) -> list[Result]:
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
                    language=language,
                )
            )
    return results


def run_solution(year: int, day: int, lang_filter: str | None = None) -> list[Result] | None:
    """Run a solution and parse its output for all available languages."""
    padded = f"{day:02d}"
    base = f"{year}-{padded}"

    # Check which files exist
    py_file = Path(f"{base}.py")
    rs_file = Path(f"{base}.rs")
    zig_file = Path(f"{base}.zig")
    nim_file = Path(f"nim_{year}_{padded}.nim")

    all_results = []

    # Run implementations based on filter
    if (lang_filter is None or lang_filter == "python") and py_file.exists():
        output = run_python(py_file)
        if output:
            all_results.extend(parse_output(output, "python"))

    if (lang_filter is None or lang_filter == "rust") and rs_file.exists():
        output = run_rust(rs_file, base)
        if output:
            all_results.extend(parse_output(output, "rust"))

    if (lang_filter is None or lang_filter == "zig") and zig_file.exists():
        output = run_zig(zig_file, base)
        if output:
            all_results.extend(parse_output(output, "zig"))

    if (lang_filter is None or lang_filter == "nim") and nim_file.exists():
        output = run_nim(nim_file)
        if output:
            all_results.extend(parse_output(output, "nim"))

    return all_results if all_results else None


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

        lang_width = max(len(r.language) for r in all_results)
        lang_width = max(lang_width, len("Lang"))

        time_width = 11  # For formatted time strings

        # Print header
        print(
            f"{'Year':<{year_width}} │ {'Day':<{day_width}} │ {'Part':<{part_width}} │ {'Answer':<{answer_width}} │ {'Lang':<{lang_width}} │ {'Time':>{time_width}}"
        )
        print(
            f"{'─' * year_width}─┼─{'─' * day_width}─┼─{'─' * part_width}─┼─{'─' * answer_width}─┼─{'─' * lang_width}─┼─{'─' * time_width}"
        )

        # Print results
        for r in all_results:
            time_str = format_time(r.time_ns, pretty=True)
            print(
                f"{r.year:<{year_width}} │ {r.day:<{day_width}} │ {r.part:<{part_width}} │ {r.answer:<{answer_width}} │ {r.language:<{lang_width}} │ {time_str}"
            )

        if stats:
            total_time = sum(r.time_ns for r in all_results)
            avg_time = total_time // len(all_results)
            min_time = min(r.time_ns for r in all_results)
            max_time = max(r.time_ns for r in all_results)

            # Calculate total width for footer (including separators)
            total_label_width = (
                year_width + 3 + day_width + 3 + part_width + 3 + answer_width + 3 + lang_width
            )

            print(
                f"{'─' * year_width}─┴─{'─' * day_width}─┴─{'─' * part_width}─┴─{'─' * answer_width}─┴─{'─' * lang_width}─┼─{'─' * time_width}"
            )
            print(
                f"{'Total':<{total_label_width}} │ {format_time(total_time, pretty=True)}"
            )
            print()
            print("Overall Statistics:")
            print(f"  Total time: {format_time(total_time, pretty=True)}")
            print(f"  Average:    {format_time(avg_time, pretty=True)}")
            print(f"  Min:        {format_time(min_time, pretty=True)}")
            print(f"  Max:        {format_time(max_time, pretty=True)}")
            print(f"  Count:      {len(all_results)} parts")

            # Stats by language
            from collections import defaultdict
            by_lang = defaultdict(list)
            for r in all_results:
                by_lang[r.language].append(r.time_ns)

            print("\nBy Language:")
            for lang in sorted(by_lang.keys()):
                times = by_lang[lang]
                lang_total = sum(times)
                lang_avg = lang_total // len(times)
                lang_min = min(times)
                lang_max = max(times)
                print(f"  {lang}:")
                print(f"    Total:   {format_time(lang_total, pretty=True)}")
                print(f"    Average: {format_time(lang_avg, pretty=True)}")
                print(f"    Min:     {format_time(lang_min, pretty=True)}")
                print(f"    Max:     {format_time(lang_max, pretty=True)}")
                print(f"    Count:   {len(times)} parts")
    else:
        # Simple tab-separated output
        print("Year\tDay\tPart\tAnswer\tLang\tTime (ns)")
        for r in all_results:
            print(f"{r.year}\t{r.day}\t{r.part}\t{r.answer}\t{r.language}\t{r.time_ns}")

        if stats:
            from collections import defaultdict

            total_time = sum(r.time_ns for r in all_results)
            print(f"\nOverall Statistics:")
            print(f"Total time: {format_time(total_time, pretty=False)} ns")
            print(f"Count: {len(all_results)} parts")

            # Stats by language
            by_lang = defaultdict(list)
            for r in all_results:
                by_lang[r.language].append(r.time_ns)

            print("\nBy Language:")
            for lang in sorted(by_lang.keys()):
                times = by_lang[lang]
                lang_total = sum(times)
                lang_avg = lang_total // len(times)
                lang_min = min(times)
                lang_max = max(times)
                print(f"  {lang}:")
                print(f"    Total:   {lang_total} ns")
                print(f"    Average: {lang_avg} ns")
                print(f"    Min:     {lang_min} ns")
                print(f"    Max:     {lang_max} ns")
                print(f"    Count:   {len(times)} parts")


def main():
    parser = argparse.ArgumentParser(description="Run Advent of Code solutions")
    parser.add_argument("year", type=int, help="Year to run")
    parser.add_argument(
        "day", type=int, nargs="?", help="Specific day to run (optional)"
    )
    parser.add_argument("--plain", action="store_true", help="Plain tab-separated output (default: pretty)")
    parser.add_argument("--no-stats", action="store_true", help="Hide statistics (default: show stats)")
    parser.add_argument("--lang", type=str, choices=["python", "rust", "zig", "nim"], help="Only run specific language")

    args = parser.parse_args()

    # Invert flags for easier logic
    pretty = not args.plain
    stats = not args.no_stats
    lang_filter = args.lang

    all_results = []

    if args.day:
        # Run single day
        results = run_solution(args.year, args.day, lang_filter)
        if results:
            all_results.extend(results)
        else:
            lang_msg = f" for {lang_filter}" if lang_filter else ""
            print(f"No solution found for {args.year}-{args.day:02d}{lang_msg}", file=sys.stderr)
            sys.exit(1)
    else:
        # Run all days for the year
        # First pass to count available days
        available_days = []
        for day in range(1, 26):
            padded = f"{day:02d}"
            base = f"{args.year}-{padded}"

            # Check if the day has the requested language or any language if no filter
            has_lang = False
            if lang_filter == "python" or lang_filter is None:
                has_lang = has_lang or Path(f"{base}.py").exists()
            if lang_filter == "rust" or lang_filter is None:
                has_lang = has_lang or Path(f"{base}.rs").exists()
            if lang_filter == "zig" or lang_filter is None:
                has_lang = has_lang or Path(f"{base}.zig").exists()
            if lang_filter == "nim" or lang_filter is None:
                has_lang = has_lang or Path(f"nim_{args.year}_{padded}.nim").exists()

            if has_lang:
                available_days.append(day)

        total = len(available_days)
        for day in available_days:
            print(
                f"\rRunning day {day} of {total}", end="", flush=True, file=sys.stderr
            )
            results = run_solution(args.year, day, lang_filter)
            if results:
                all_results.extend(results)

        # Clear progress line
        if all_results:
            print("\r" + " " * 30 + "\r", end="", file=sys.stderr)

    if all_results:
        print_results(all_results, pretty=pretty, stats=stats)
    else:
        print(f"No solutions found for year {args.year}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
