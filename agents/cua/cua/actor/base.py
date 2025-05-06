# type: ignore

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from devicebay import Device
from mllm import Prompt as SkillPrompt
from rich.console import Console
from skillpacks import EnvState, V1Action
from taskara import Task

from threadmem import RoleThread

console = Console(force_terminal=True)


@dataclass
class Step:
    """A step in an episode"""

    state: EnvState
    action: V1Action
    thought: str
    raw_response: str
    action_opts: Optional[List[V1Action]] = None
    thread: Optional[RoleThread] = None
    task: Optional[Task] = None
    model_id: Optional[str] = None
    prompt: Optional[SkillPrompt] = None
    in_tokens: int = 0
    out_tokens: int = 0


T = TypeVar("T", bound=Device)


class Actor(ABC, Generic[T]):
    """An actor that can act on a task"""

    @abstractmethod
    def act(self, task: Task, device: T, history: List[Step]) -> Step:
        pass
