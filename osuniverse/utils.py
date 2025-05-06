import json
import os
from datetime import datetime
from typing import Any

from osuniverse.data.testcaserun import TestCaseRun

WEIGHTS: dict[str, float] = {
    "paper": 0.5,
    "wood": 1,
    "bronze": 2,
    "silver": 4,
    "gold": 8,
}


def find_json_files(directory: str) -> list[str]:
    """Recursively find all JSON files in the given directory."""
    json_files: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return sorted(json_files)


def load_scored_run(json_path: str) -> TestCaseRun:
    with open(json_path, "r") as f:
        data = json.load(f)
    return TestCaseRun.from_dict(data)


def save_scored_run(json_path: str, run: TestCaseRun):
    with open(json_path, "w") as f:
        json.dump(run.to_dict(), f, indent=2)


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_number(num: int | float) -> str:
    """Format large numbers with K and M suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(int(num))


def calculate_stats(json_files: list[str]) -> dict[str, int | float | dict[str, Any]]:
    """Calculate statistics for all test runs."""
    total = len(json_files)
    passed = 0
    ai_passed = 0
    ai_errors = 0
    human_passed = 0
    human_failed = 0
    human_unreviewed = 0
    loaded = 0
    disagreements = 0

    total_duration = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_validation_input_tokens = 0
    total_validation_output_tokens = 0

    by_levels: dict[str, Any] = {}
    by_categories: dict[str, Any] = {}

    by_categories_by_level: dict[str, dict[str, Any]] = {}

    for file in json_files:
        try:
            run = load_scored_run(file)
            loaded += 1
            score = run.ai_score if run.human_score < 0 else run.human_score
            if score > 0:
                passed += 1
            if run.ai_score >= 1.0:
                ai_passed += 1
            elif run.ai_score == -1.0:
                ai_errors += 1
            if run.human_score >= 1.0:
                human_passed += 1
            elif run.human_score == 0:
                human_failed += 1
            else:  # human_score == -1
                human_unreviewed += 1
            if (
                run.ai_score != run.human_score
                and run.ai_score != -1.0
                and run.human_score != -1.0
            ):
                disagreements += 1
            run_duration = run.trajectory[-1].timestamp - run.trajectory[0].timestamp

            total_duration += run_duration
            total_input_tokens += run.input_tokens
            total_output_tokens += run.output_tokens
            total_validation_input_tokens += run.validation_input_tokens
            total_validation_output_tokens += run.validation_output_tokens

            if run.category not in by_categories:
                by_categories_by_level[run.category] = {}
            if run.level not in by_categories_by_level[run.category]:
                by_categories_by_level[run.category][run.level] = {
                    "amount": 0,
                    "passed": 0,
                }
            by_categories_by_level[run.category][run.level]["amount"] += 1
            by_categories_by_level[run.category][run.level]["passed"] += (
                1 if score > 0 else 0
            )

            if run.level not in by_levels:
                by_levels[run.level] = {
                    "amount": 0,
                    "passed": 0,
                    "duration": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "validation_input_tokens": 0,
                    "validation_output_tokens": 0,
                }
            by_levels[run.level]["amount"] += 1
            by_levels[run.level]["passed"] += 1 if score > 0 else 0
            by_levels[run.level]["duration"] += run_duration
            by_levels[run.level]["input_tokens"] += run.input_tokens
            by_levels[run.level]["output_tokens"] += run.output_tokens
            by_levels[run.level]["validation_input_tokens"] += (
                run.validation_input_tokens
            )
            by_levels[run.level]["validation_output_tokens"] += (
                run.validation_output_tokens
            )

            if run.category not in by_categories:
                by_categories[run.category] = {
                    "amount": 0,
                    "passed": 0,
                    "duration": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "validation_input_tokens": 0,
                    "validation_output_tokens": 0,
                }
            by_categories[run.category]["amount"] += 1
            by_categories[run.category]["passed"] += 1 if score > 0 else 0
            by_categories[run.category]["duration"] += run_duration
            by_categories[run.category]["input_tokens"] += run.input_tokens
            by_categories[run.category]["output_tokens"] += run.output_tokens
            by_categories[run.category]["validation_input_tokens"] += (
                run.validation_input_tokens
            )
            by_categories[run.category]["validation_output_tokens"] += (
                run.validation_output_tokens
            )
        except Exception:
            continue

    reviewed = human_passed + human_failed

    weighted_score = 0
    weighted_total = 0
    for level, score in by_levels.items():
        weighted_score += score["passed"] * WEIGHTS[level]
        weighted_total += score["amount"] * WEIGHTS[level]
    weighted_success_rate = (
        (weighted_score / weighted_total * 100) if weighted_total > 0 else 0
    )

    return {
        "total": total,
        "loaded": loaded,
        "passed": passed,
        "success_rate": (passed / loaded * 100) if loaded > 0 else 0,
        "weighted_success_rate": weighted_success_rate,
        "ai_passed": ai_passed,
        "ai_success_rate": (ai_passed / loaded * 100) if loaded > 0 else 0,
        "ai_errors": ai_errors,
        "human_passed": human_passed,
        "human_failed": human_failed,
        "human_unreviewed": human_unreviewed,
        "human_success_rate": (human_passed / reviewed * 100) if reviewed > 0 else 0,
        "disagreements": disagreements,
        "disagreement_rate": (disagreements / loaded * 100) if loaded > 0 else 0,
        "total_duration": total_duration,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_validation_input_tokens": total_validation_input_tokens,
        "total_validation_output_tokens": total_validation_output_tokens,
        "by_levels": by_levels,
        "by_categories": by_categories,
        "by_categories_by_level": by_categories_by_level,
    }
