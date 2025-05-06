from ..config import Config
from ..data.testcase import TestCase
from ..data.testcaserun import TestCaseRun


class BaseRunner:
    def __init__(self):
        pass

    def run(self, testcase: TestCase, config: Config) -> TestCaseRun | None:
        pass
