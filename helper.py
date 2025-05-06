import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

try:
    import yaml
except ImportError:
    print("Please install PyYAML (pip install pyyaml) to continue.")
    sys.exit(1)


def find_testcase_files(folder: str) -> list[str]:
    """
    Recursively find all .yaml files in the given directory (folder).
    """
    yaml_files: list[str] = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".yaml"):
                yaml_files.append(os.path.join(root, file))
    return sorted(yaml_files)


def do_distribution(folder: str):
    """
    Calculate and display how many testcases belong to each level, both overall and broken down by category.
    """
    levels_list = ["paper", "wood", "bronze", "silver", "gold"]

    level_counts = {level: 0 for level in levels_list}

    # NEW: hold distribution by category and level
    category_by_level: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {lvl: 0 for lvl in levels_list}
    )
    categories: set[str] = set()

    for filepath in find_testcase_files(folder):
        try:
            with open(filepath, "r") as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
            # Derive category from subfolder rather than file contents
            relative_path = os.path.relpath(filepath, folder)
            dir_part = os.path.dirname(relative_path)
            category = dir_part.split(os.path.sep)[0] if dir_part else "unknown"

            level = data.get("level", "unknown")

            # Track overall level count if recognized
            if level in level_counts:
                level_counts[level] += 1

            # Track category-level count
            categories.add(category)
            if level in category_by_level[category]:
                category_by_level[category][level] += 1

        except Exception as e:
            print(f"Warning: Failed to read '{filepath}' due to: {e}")

    # Create a Console object to display tables
    console = Console()
    console.print(
        f"Distribution of testcases in folder: {folder}\n", style="bold yellow"
    )

    # Print detailed distribution by category and level using another Rich Table
    table_detail = Table(
        title="Distribution by Level per Category",
        show_header=True,
        header_style="bold magenta",
    )

    # Add first column, Category
    table_detail.add_column("Category", justify="left", style="cyan")

    # Create a column for each level
    for lvl in levels_list:
        table_detail.add_column(lvl, justify="right")

    # Final column for the row total
    table_detail.add_column("Total", justify="right", style="bold")

    grand_totals = {lvl: 0 for lvl in levels_list}
    total_tests = 0

    for cat in sorted(categories):
        row_counts = category_by_level[cat]
        cat_sum = 0
        row_data = [cat]  # start with the category name

        for lvl in levels_list:
            count = row_counts.get(lvl, 0)
            cat_sum += count
            grand_totals[lvl] += count
            row_data.append(str(count))

        total_tests += cat_sum
        row_data.append(str(cat_sum))  # final column: total tests for cat

        table_detail.add_row(*row_data)

    # Add a row at the end for totals across all categories
    totals_row = ["TOTAL"]
    for lvl in levels_list:
        totals_row.append(str(grand_totals[lvl]))
    totals_row.append(str(total_tests))

    # We'll highlight this final totals row
    table_detail.add_row(*totals_row, style="bold yellow")

    console.print(table_detail)


def do_cleanup():
    """
    Clean up surfkit devices and trackers that match specific patterns.
    Python equivalent of cleanup_surfkit.sh.
    """
    device_pattern = re.compile(
        r"^surfkit_desktop_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    tracker_pattern = re.compile(
        r"^surfkit_tracker_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )

    # Helper function to run surfkit commands
    def run_surfkit_cmd(cmd: list[str]):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            print("Error: surfkit command not found")
            sys.exit(1)

    # Clean up devices
    result = run_surfkit_cmd(["surfkit", "list", "devices"])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            device_name = line.split()[0]
            if device_pattern.match(device_name):
                print(f"Killing device: {device_name}")
                run_surfkit_cmd(["surfkit", "delete", "device", device_name])

    # Clean up trackers
    result = run_surfkit_cmd(["surfkit", "list", "trackers"])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            tracker_name = line.split()[0]
            if tracker_pattern.match(tracker_name):
                print(f"Killing tracker: {tracker_name}")
                run_surfkit_cmd(["surfkit", "delete", "tracker", tracker_name])


def main():
    parser = argparse.ArgumentParser(
        prog="helper.py", description="CLI for OSUniverse helpers"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # distribution subcommand
    dist_parser = subparsers.add_parser(
        "distribution", help="Display test distribution over levels"
    )
    dist_parser.add_argument(
        "--folder", default="testcases", help="Folder to scan for .yaml files"
    )

    # cleanup subcommand
    subparsers.add_parser("cleanup", help="Clean up stuck surfkit devices and trackers")

    args = parser.parse_args()

    if args.command == "distribution":
        do_distribution(args.folder)
    elif args.command == "cleanup":
        do_cleanup()


if __name__ == "__main__":
    main()
