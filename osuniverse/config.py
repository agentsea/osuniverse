from typing import Optional


class Config:
    def __init__(self):
        self.agent_yaml: str = (
            "agents/react/agent.yaml"  # path to the Surfkit-compatible agent yaml file
        )
        self.agent_model: str = "gpt-4o"  # model to use for the agent
        self.agent_model_base_url: Optional[str] = None  # base url for the agent model
        self.testcases_dir: str = "testcases"  # path to the test cases
        self.results_dir: str = "results"  # path to store the results
        self.categories: list[str] = []  # categories of the test cases
        self.levels: list[str] = []  # levels of the test cases
        self.max_steps: dict[str, int] = {
            "paper": 5,
            "wood": 25,
            "bronze": 50,
            "silver": 75,
            "gold": 100,
        }  # maximum number of steps for each test case
        self.dry_run: bool = (
            False  # dry run the benchmark without actually running the test cases
        )
        self.mode: str = (
            "run-all"  # mode of the benchmark: run-all, rerun-failed, validate-only
        )
        self.runners: int = 1  # number of parallel runners to use
