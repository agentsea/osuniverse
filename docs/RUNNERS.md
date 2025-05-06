# Writing Custom Runners

OSUniverse relies on the [Surfkit](https://github.com/agentsea/surfkit) framework, which allows creating containerized vistual desktops, trackers and agents. If you want to write a custom agent without changing the runner, please refer to [Writing Custom Surfkit-based Agents](AGENTS.md).

However, using Surfkit is not mandatory. If you want to write your own agent with your own virtual device, please read this guide.

## Writing Custom Runners

To write your own runner, you can start from any existing runner in the `runners` directory. If you take, for example, the `surfkit_agent_runner.py` file as a base, you will need to re-implement the `run` method.

The core components to pay attention to are:

* The `run` method takes two arguments: `testcase` (of type `TestCase`) and `config` (of type `Config`).
* The `run` method should return a `TestCaseRun` object, which contains the result of the test case run.
* To keep the tests fair, you should use the Docker Image from the `testcase` object or inherit from it; you should create a new container with a random name for each test case and execute the setup script (`testcase.setup_cmd`).
    * At the time of writing this guide, all test cases use the same image. You are allowed to install additional packages in the image, if your agent requires them to operate.
* Your agent should run through the test case while producing the trajectory which consists of `Step` objects:

```python
step = Step(
    timestamp=time.time(),
    action=action,
    thought=thought,
    screenshot=screenshot,
)
```

* After the execution of the test case is complete, you should run all checks from the `testcase` object. The results of the checks have the following format:

```python
check_result = CheckResult(
    check=check, # of type Check
    result=result, # of type str
    score=score, # of type int, 0 or 1
    validation_input_tokens=validation_input_tokens, # of type int
    validation_output_tokens=validation_output_tokens, # of type int
)
```

* Trajectory, check results, and other information should be stored in the `TestCaseRun` object, which expands the `TestCase` object with additional fields:

```python
class TestCaseRun(TestCase):
    agent_yaml: str
    agent_model: str
    max_steps: int = 0
    status: Optional[str] = None
    trajectory: list[Step] = field(default_factory=list)
    result: Optional[Step] = None
    command_output_check_results: list[CommandOutputCheckResult] = field(
        default_factory=list
    )
    input_tokens: int = 0
    output_tokens: int = 0
    ai_score: float = -1.0
    ai_comment: Optional[str] = None
    human_score: float = -1.0
    human_comment: Optional[str] = None
    validation_input_tokens: int = 0
    validation_output_tokens: int = 0
```

The only crucial parts here are `trajectory` and `command_output_check_results`. Everything else is nice to have (you can set `agent_yaml` and `agent_model` to any values, they are only used for logging purposes). The last six fields are modified by the validator outside of the runner scope, so you don't need to worry about them.

## Using Custom Runners

OSUniverse CLI currently doesn't support custom runners, so you'll need to modify the `benchmark.py` file to use your runner. The modification is minimal: you just need to replace the `SurfkitAgentRunner` in the `run_testcase` function:

```python
runner = SurfkitAgentRunner()
```

with your runner:

```python
runner = YourRunner()
```

You can also modify the `run_testcase` function to add additional logging or other functionality.
