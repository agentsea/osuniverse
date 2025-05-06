from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from rich.console import Console

from osuniverse.data.testcase import Check
from osuniverse.data.testcaserun import TestCaseRun

console = Console()


@dataclass
class CheckResult:
    check: Check
    result: str
    score: int
    validation_input_tokens: int
    validation_output_tokens: int


class BaseValidator(ABC):
    def __init__(self):
        pass

    def validate(self, testcaserun: TestCaseRun) -> TestCaseRun:
        check_results: List[CheckResult] = []

        for check in testcaserun.checks:
            check_result = self.validate_check(check, testcaserun)
            if check_result.score != -1:
                check_results.append(check_result)
                console.print(
                    f"Check {check.CHECK_TYPE} | score: {check_result.score} | result: {check_result.result}"
                )

        score: int = 0 if any(cr.score == 0 for cr in check_results) else 1
        comment: str = ""
        for cr in check_results:
            comment += f" 🔹 Check {cr.check.CHECK_TYPE} | score: {cr.score} | result: {cr.result}"

        testcaserun.ai_score = score
        testcaserun.ai_comment = comment
        testcaserun.validation_input_tokens = sum(
            cr.validation_input_tokens for cr in check_results
        )
        testcaserun.validation_output_tokens = sum(
            cr.validation_output_tokens for cr in check_results
        )

        return testcaserun

    @abstractmethod
    def validate_check(self, check: Check, testcaserun: TestCaseRun) -> CheckResult:
        return CheckResult(
            check=check,
            result="",
            score=-1,
            validation_input_tokens=0,
            validation_output_tokens=0,
        )
