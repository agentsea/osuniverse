import calendar
import datetime
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict, List, Type, TypeVar, cast

import yaml

T = TypeVar("T", bound="Check")


def expand_month_placeholders(text: str) -> str:
    """Replace [%MONTH+k%] with the month name offset by k from the current month."""
    pattern = re.compile(r"\[\%MONTH\+(\d+)\%\]")

    def month_replacer(match: re.Match[str]) -> str:
        offset = int(match.group(1))
        now = datetime.datetime.now()
        new_month = (now.month - 1 + offset) % 12 + 1
        return calendar.month_name[new_month]

    return pattern.sub(month_replacer, text)


class Check(ABC):
    CHECK_TYPE: ClassVar[str]  # Added class attribute to hold check type

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def expand_placeholders(self) -> None:
        """Apply placeholder expansion to relevant attributes."""
        pass

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        check_type = data.pop("type", None)
        if check_type is None:
            raise ValueError("Missing check type in data.")
        if check_type not in CHECK_REGISTRY:
            raise ValueError(f"Unknown check type: {check_type}")
        return cast(T, CHECK_REGISTRY[check_type].from_data(data))

    @classmethod
    @abstractmethod
    def from_data(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create an instance from a dictionary (excluding the 'type' key)."""
        pass


# Registry to hold mapping of check types to classes
CHECK_REGISTRY: Dict[str, Type[Check]] = {}


def register_check(check_type: str):
    def decorator(cls: Type[Check]):
        CHECK_REGISTRY[check_type] = cls
        cls.CHECK_TYPE = check_type
        return cls

    return decorator


@register_check("returned_result")
@dataclass
class ReturnedResultCheck(Check):
    returned_result: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.CHECK_TYPE, "value": self.returned_result}

    def expand_placeholders(self) -> None:
        self.returned_result = expand_month_placeholders(self.returned_result)

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "ReturnedResultCheck":
        return cls(returned_result=data["value"])


@register_check("final_screenshot")
@dataclass
class FinalScreenshotCheck(Check):
    final_screenshot: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.CHECK_TYPE, "value": self.final_screenshot}

    def expand_placeholders(self) -> None:
        self.final_screenshot = expand_month_placeholders(self.final_screenshot)

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "FinalScreenshotCheck":
        return cls(final_screenshot=data["value"])


@register_check("expected_flow")
@dataclass
class ExpectedFlowCheck(Check):
    expected_flow: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.CHECK_TYPE, "value": self.expected_flow}

    def expand_placeholders(self) -> None:
        self.expected_flow = expand_month_placeholders(self.expected_flow)

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "ExpectedFlowCheck":
        return cls(expected_flow=data["value"])


@register_check("command_output")
@dataclass
class CommandOutputCheck(Check):
    command: str
    command_output: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.CHECK_TYPE,
            "command": self.command,
            "value": self.command_output,
        }

    def expand_placeholders(self) -> None:
        self.command = expand_month_placeholders(self.command)
        self.command_output = expand_month_placeholders(self.command_output)

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "CommandOutputCheck":
        return cls(command=data["command"], command_output=data["value"])


@dataclass
class TestCase:
    id: str
    name: str
    category: str
    level: str
    task: str
    setup_cmd: str
    desktop_image: str
    checks: List[Check]

    @classmethod
    def from_yaml(cls, yaml_path: str, category: str, id: str) -> "TestCase":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        data["category"] = category
        data["id"] = id

        # Convert check dictionaries to Check objects
        checks_data = data.pop("checks", [])
        checks = [Check.from_dict(check_dict) for check_dict in checks_data]
        test = cls(checks=checks, **data)
        test.expand_placeholders()

        return test

    def expand_placeholders(self) -> None:
        self.task = expand_month_placeholders(self.task)
        for check in self.checks:
            check.expand_placeholders()

    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize the TestCase to a dict suitable for JSON output."""
        data = asdict(self)
        # Replace check objects with their dict representation
        data["checks"] = [check.to_dict() for check in self.checks]
        return data

    def __str__(self) -> str:
        checks_str = "\n".join(f"    {check}" for check in self.checks)
        return (
            f"TestCase:\n"
            f"    id: {self.id}\n"
            f"    name: {self.name}\n"
            f"    category: {self.category}\n"
            f"    level: {self.level}\n"
            f"    task: {self.task}\n"
            f"    setup_cmd: {self.setup_cmd}\n"
            f"    desktop_image: {self.desktop_image}\n"
            f"    checks:\n{checks_str}\n"
        )
