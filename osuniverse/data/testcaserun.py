from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Config
from .testcase import Check, TestCase


@dataclass
class Step:
    timestamp: float
    action: str
    thought: str
    screenshot: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "thought": self.thought,
            "screenshot": self.screenshot,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return Step(
            timestamp=data["timestamp"],
            action=data["action"],
            thought=data["thought"] if "thought" in data else "",
            screenshot=data["screenshot"] if "screenshot" in data else "",
        )


@dataclass
class CommandOutputCheckResult:
    command: str
    output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "output": self.output,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return CommandOutputCheckResult(
            command=data["command"],
            output=data["output"],
        )


@dataclass
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

    @classmethod
    def from_testcase(cls, testcase: TestCase, config: Config):
        return cls(
            id=testcase.id,
            name=testcase.name,
            category=testcase.category,
            level=testcase.level,
            task=testcase.task,
            setup_cmd=testcase.setup_cmd,
            desktop_image=testcase.desktop_image,
            checks=testcase.checks,
            agent_yaml=config.agent_yaml,
            agent_model=config.agent_model,
            max_steps=config.max_steps[testcase.level],
        )

    def add_step(self, step: Step):
        self.trajectory.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "level": self.level,
            "task": self.task,
            "setup_cmd": self.setup_cmd,
            "desktop_image": self.desktop_image,
            "checks": [check.to_dict() for check in self.checks],
            "agent_yaml": self.agent_yaml,
            "agent_model": self.agent_model,
            "max_steps": self.max_steps,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None,
            "trajectory": [step.to_dict() for step in self.trajectory],
            "command_output_check_results": [
                result.to_dict() for result in self.command_output_check_results
            ],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ai_score": self.ai_score,
            "ai_comment": self.ai_comment,
            "human_score": self.human_score,
            "human_comment": self.human_comment,
            "validation_input_tokens": self.validation_input_tokens,
            "validation_output_tokens": self.validation_output_tokens,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]):
        return TestCaseRun(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            level=data["level"],
            task=data["task"],
            setup_cmd=data["setup_cmd"],
            desktop_image=data["desktop_image"],
            checks=[Check.from_dict(check) for check in data["checks"]]
            if "checks" in data
            else [],
            agent_yaml=data["agent_yaml"],
            agent_model=data["agent_model"],
            max_steps=data["max_steps"],
            status=data["status"],
            result=Step.from_dict(data["result"]) if data["result"] else None,
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            trajectory=[Step.from_dict(step) for step in data["trajectory"]],
            command_output_check_results=[
                CommandOutputCheckResult.from_dict(result)
                for result in data["command_output_check_results"]
            ],
            ai_score=data["ai_score"],
            ai_comment=data["ai_comment"],
            human_score=data["human_score"],
            human_comment=data["human_comment"],
            validation_input_tokens=data["validation_input_tokens"]
            if "validation_input_tokens" in data
            else 0,
            validation_output_tokens=data["validation_output_tokens"]
            if "validation_output_tokens" in data
            else 0,
        )
