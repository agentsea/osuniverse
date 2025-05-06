import argparse
import json
import os
import time
from enum import Enum
from multiprocessing import Pool

from dotenv import load_dotenv
from rich.console import Console

from osuniverse.config import Config
from osuniverse.data.testcase import TestCase
from osuniverse.data.testcaserun import TestCaseRun
from osuniverse.runners.surfkit_agent_runner import SurfkitAgentRunner
from osuniverse.validators.cot_gemini_validator import COTGeminiValidator

load_dotenv()

console = Console()


class RunStatus(Enum):
    COMPLETED = "completed"
    RUN_FAILED = "run_failed"
    VALIDATION_FAILED = "validation_failed"


def parse_args():
    parser = argparse.ArgumentParser(description="osuniverse benchmark runner")

    parser.add_argument(
        "--agent-yaml",
        type=str,
        default="agents/react/agent.yaml",
        help="Path to the Surfkit-compatible agent yaml file",
    )

    parser.add_argument(
        "--agent-model",
        type=str,
        default="gpt-4o",
        help="Model to use for the agent. The choice of model is limited by the agent yaml file. Refer to README for more details.",
    )

    parser.add_argument(
        "--agent-model-base-url",
        type=str,
        default=None,
        help="Base url for the agent model. If not specified, the default base url will be used.",
    )

    parser.add_argument(
        "--testcases",
        type=str,
        default="testcases",
        help="Directory containing test cases",
    )

    parser.add_argument(
        "--results",
        type=str,
        default="results",
        help="Directory to store benchmark results",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Categories of test cases to run. If not specified, run all. Categories are separated by commas.",
    )

    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Levels of test cases to run (paper, wood, bronze, silver, gold). If not specified, run all. Levels are separated by commas.",
    )

    parser.add_argument(
        "--max-steps",
        type=str,
        default=None,
        help="Maximum number of steps for each test case. If there are several values separated by commas, the values are applied to levels from paper to gold, with the last value being applied to all remaining levels. For example, --max-steps 10,20,30 will set 10 steps for paper, 20 steps for wood, 30 steps for bronze, silver, and gold.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the benchmark without actually running the test cases",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="run-all",
        choices=["run-all", "rerun-failed", "validate-only"],
        help="Mode of the benchmark. run-all mode will run all the test cases that have not been run yet. rerun-failed mode will rerun the test cases that have been run and failed, and run the test cases that have not been run yet. validate-only mode will only validate the test cases that have been run.",
    )

    parser.add_argument(
        "--runners",
        type=int,
        default=1,
        help="Number of parallel runners to use",
    )

    args = parser.parse_args()

    # Create and populate config object
    config = Config()
    config.agent_yaml = args.agent_yaml
    config.agent_model = args.agent_model
    config.agent_model_base_url = args.agent_model_base_url
    config.testcases_dir = args.testcases
    config.results_dir = args.results
    config.categories = args.categories.split(",") if args.categories else []
    config.levels = args.levels.split(",") if args.levels else []
    max_steps_str: list[str] = args.max_steps.split(",") if args.max_steps else []
    if len(max_steps_str) > 0:
        max_steps = [int(step) for step in max_steps_str]
        last_step = max_steps[-1]
        for _ in range(len(max_steps), 5):
            max_steps.append(last_step)
        config.max_steps = {
            "paper": max_steps[0],
            "wood": max_steps[1],
            "bronze": max_steps[2],
            "silver": max_steps[3],
            "gold": max_steps[4],
        }
    config.dry_run = args.dry_run
    config.mode = args.mode
    config.runners = args.runners
    return config


def load_testcases_and_runs(
    testcases_dir: str,
    selected_categories: list[str] = [],
    selected_levels: list[str] = [],
    mode: str = "run-all",
) -> list[tuple[str, str, str, str, TestCase, TestCaseRun | None]]:
    testcases: list[tuple[str, str, str, str, TestCase, TestCaseRun | None]] = []

    # Get immediate subdirectories (categories)
    categories = [
        d
        for d in os.listdir(testcases_dir)
        if os.path.isdir(os.path.join(testcases_dir, d))
    ]

    # Process each category
    for category_name in categories:
        category_path = os.path.join(testcases_dir, category_name)

        # Read all yaml files in the category directory
        for file in os.listdir(category_path):
            if file.endswith(".yaml"):
                yaml_path = os.path.join(category_path, file)
                file_id = os.path.splitext(file)[0]  # Remove extension

                # Create TestCase with category and id
                testcase = TestCase.from_yaml(
                    yaml_path, category=category_name, id=file_id
                )

                result_category_dir = os.path.join(
                    config.results_dir, testcase.category
                )
                result_path = os.path.join(result_category_dir, f"{testcase.id}.json")

                # Create category directory if it doesn't exist
                if not os.path.exists(result_category_dir):
                    os.makedirs(result_category_dir, exist_ok=True)

                if os.path.exists(result_path):
                    testcaserun = TestCaseRun.from_dict(
                        json.load(open(result_path, "r"))
                    )
                else:
                    testcaserun = None

                testcases.append(
                    (
                        file_id,
                        testcase.category,
                        testcase.level,
                        result_path,
                        testcase,
                        testcaserun,
                    )
                )

    if selected_categories != []:
        testcases = [
            testcase for testcase in testcases if testcase[1] in selected_categories
        ]

    if selected_levels != []:
        testcases = [
            testcase for testcase in testcases if testcase[2] in selected_levels
        ]

    if mode == "run-all":
        testcases = [
            testcase
            for testcase in testcases
            if (testcase[5] is None or testcase[5].id != testcase[4].id)
        ]
    elif mode == "rerun-failed":
        testcases = [
            testcase
            for testcase in testcases
            if (
                (testcase[5] is not None and testcase[5].human_score == 0)
                or (
                    testcase[5] is not None
                    and testcase[5].ai_score == 0
                    and testcase[5].human_score == -1.0
                )
                or (testcase[5] is not None and testcase[5].id != testcase[4].id)
                or testcase[5] is None
            )
        ]
    elif mode == "validate-only":
        testcases = [
            testcase
            for testcase in testcases
            if (testcase[5] is not None and testcase[5].id == testcase[4].id)
        ]

    def get_level_order(level: str) -> int:
        return ["paper", "wood", "bronze", "silver", "gold"].index(level)

    # Sort by category, level, then id
    testcases = sorted(
        testcases,
        key=lambda x: (get_level_order(x[2] or ""), x[1] or "", x[0] or ""),
    )

    return testcases


def run_testcase(
    args: tuple[TestCase, TestCaseRun | None, str, Config, int, int, int],
) -> tuple[RunStatus, str, str, float]:
    testcase, testcaserun, result_path, config, index, total, sleep_time = args

    # independent instances for each test case, for parallel runs
    console = Console()
    runner = SurfkitAgentRunner()
    validator = COTGeminiValidator()

    if config.mode == "validate-only" and testcaserun is not None:
        # reset validation results for the test case

        console.print(
            f"Resetting validation for test case {index + 1}/{total}: {testcase.id} - {testcase.name}",
            style="yellow",
        )
        testcaserun.ai_score = -1.0
        testcaserun.ai_comment = None
        testcaserun.validation_input_tokens = 0
        testcaserun.validation_output_tokens = 0
        testcaserun.checks = testcase.checks
    else:
        # run the test case
        time.sleep(sleep_time)
        console.print(
            f"Running test case {index + 1}/{total}: {testcase.id} - {testcase.name}",
            style="bold blue",
        )
        try:
            testcaserun = runner.run(testcase, config)
        except Exception as e:
            console.print(f"Error running test case: {str(e)}", style="bold red")
            return (
                RunStatus.RUN_FAILED,
                result_path,
                f"Error running test case: {str(e)}",
                0,
            )

    # run validation
    console.print(
        f"Validating test case {index + 1}/{total}: {testcase.id} - {testcase.name}",
        style="bold blue",
    )
    scoredtestcaserun = testcaserun
    for i in range(3):
        scoredtestcaserun = validator.validate(testcaserun)
        if scoredtestcaserun.ai_score > -1.0:
            break
        else:
            console.print(
                f"Validator failed with error: {scoredtestcaserun.ai_comment}. Retrying...",
                style="bold yellow",
            )
            if i == 2:
                return (
                    RunStatus.VALIDATION_FAILED,
                    result_path,
                    f"Validator failed with error: {scoredtestcaserun.ai_comment}.",
                    0,
                )

    with open(result_path, "w") as f:
        json.dump(scoredtestcaserun.to_dict(), f)

    console.print(
        f"Test case {index + 1}/{total} completed",
        style="bold green",
    )

    return (
        RunStatus.COMPLETED,
        result_path,
        f"Test case {testcase.category} - {testcase.level} - {testcase.name} completed",
        scoredtestcaserun.ai_score,
    )


if __name__ == "__main__":
    config = parse_args()

    console.print(f"🔹 Using agent yaml file: {config.agent_yaml}", style="bold green")
    console.print(f"🔹 Using agent model: {config.agent_model}", style="bold green")
    console.print(
        f"🔹 Using testcases directory: {config.testcases_dir}", style="bold green"
    )
    console.print(
        f"🔹 Using results directory: {config.results_dir}", style="bold green"
    )

    if config.categories == []:
        console.print("🔹 Benchmark is running for all categories", style="bold green")
    else:
        console.print(
            f"🔹 Benchmark is running for categories {config.categories}",
            style="bold green",
        )
    if config.levels == []:
        console.print("🔹 Benchmark is running for all levels", style="bold green")
    else:
        console.print(
            f"🔹 Benchmark is running for levels {config.levels}", style="bold green"
        )
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)

    testcases = load_testcases_and_runs(
        config.testcases_dir,
        config.categories,
        config.levels,
        config.mode,
    )

    console.print(f"Ready to run {len(testcases)} test cases:", style="bold blue")

    for testcase in testcases:
        tc = testcase[4]
        console.print(
            f"    {tc.level.upper()} / {tc.category} / {tc.id} / {tc.name} / {config.max_steps[tc.level]} max steps",
            style="blue",
        )

    if config.dry_run:  # type: ignore
        console.print(
            "Dry run mode enabled. Skipping test case execution.", style="bold yellow"
        )
        exit(0)

    # Prepare arguments for parallel processing
    args_list = [
        (testcase[4], testcase[5], testcase[3], config, i, len(testcases), 0)
        for i, testcase in enumerate(testcases)
    ]
    # We start each next thread 30 seconds later to minimize the port resolution conflicts
    for i in range(config.runners):
        if len(args_list) > i:
            args_list[i] = (
                args_list[i][0],
                args_list[i][1],
                args_list[i][2],
                args_list[i][3],
                args_list[i][4],
                args_list[i][5],
                i * 30,
            )

    # Run test cases in parallel if runners > 1
    if config.runners > 1:
        console.print(
            f"Running with {config.runners} parallel processes", style="bold green"
        )
        results: list[tuple[RunStatus, str, str, float]] = []
        with Pool(config.runners) as pool:
            # Use imap instead of map to process items one at a time
            for result in pool.imap(run_testcase, args_list):
                status, result_path, message, score = result
                results.append(result)
                console.print(
                    message + " " + str(status) + " " + result_path, style="dim"
                )

    else:
        # Run sequentially if runners = 1
        results = [run_testcase(args) for args in args_list]

    console.print(
        f"Benchmark completed. Results stored in {config.results_dir}",
        style="bold green",
    )
    console.print("SUMMARY OF THE RUN:", style="bold blue")
    for result in results:
        status, result_path, message, score = result
        if status == RunStatus.RUN_FAILED or status == RunStatus.VALIDATION_FAILED:
            style = "red"
        elif score > 0:
            style = "green"
        else:
            style = "yellow"
        console.print(
            f"{status.value.capitalize()} 🔹 Score: {'✅' if score > 0 else '❌'} 🔹 {message} 🔹 Result path: {result_path}",
            style=style,
        )
