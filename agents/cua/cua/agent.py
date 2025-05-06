import logging
import os
import time
import traceback
from typing import Final, List, Optional, Tuple, Type

from agentdesk import Desktop
from devicebay import Device
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks.action_opts import ActionOpt
from surfkit.agent import TaskAgent
from surfkit.skill import Skill
from taskara import Task, TaskStatus
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
)
from toolfuse.util import AgentUtils

from .actor.base import Actor, Step
from .actor.oai import OaiActor

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)


class CUAConfig(BaseModel):
    pass


class CUA(TaskAgent):
    """A desktop agent that learns"""

    def learn_skill(
        self,
        skill: Skill,
    ):
        """Learn a skill

        Args:
            skill (Skill): The skill
        """
        raise NotImplementedError("Subclasses must implement this method")

    def solve_task(
        self,
        task: Task,
        device: Optional[Device] = None,
        max_steps: int = 30,
    ) -> Task:
        """Solve a task

        Args:
            task (Task): Task to solve.
            device (Device): Device to perform the task on. Defaults to None.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        """

        if not device:
            raise ValueError("This agent expects a desktop")

        # Post a message to the default thread to let the user know the task is in progress
        task.post_message("assistant", f"Starting task '{task.description}'")

        # Create threads in the task to update the user
        console.print("creating threads...")
        task.ensure_thread("debug")
        task.post_message("assistant", "I'll post debug messages here", thread="debug")

        # Check that the device we received is one we support
        if not isinstance(device, Desktop):
            raise ValueError("Only desktop devices supported")

        # Add standard agent utils to the device
        device.merge(AgentUtils())

        # Get the json schema for the tools
        tools = device.json_schema()
        console.print("tools: ", style="purple")
        console.print(JSON.from_data(tools))

        # Get info about the desktop
        info = device.info()
        screen_size = info["screen_size"]
        console.print(f"Screen size: {screen_size}")

        history: List[Step] = []

        # actor = SwiftActor()
        actor = OaiActor()

        # Loop to run actions
        for i in range(max_steps):
            console.print(f"-------step {i + 1}", style="green")

            try:
                step, done = self.take_action(device, task, actor, history)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.save()
                task.post_message("assistant", f"❗ Error taking action: {e}")
                return task

            if step:
                history.append(step)

            if done:
                console.print("task is done", style="green")
                # TODO: remove
                time.sleep(10)
                return task

            time.sleep(2)

        task.status = TaskStatus.FAILED
        task.save()
        task.post_message("assistant", "❗ Max steps reached without solving task")
        console.print("Reached max steps without solving task", style="red")

        return task

    @retry(
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def take_action(
        self,
        device: Device,
        task: Task,
        actor: Actor,
        history: List[Step],
    ) -> Tuple[Optional[Step], bool]:
        """Take an action

        Args:
            desktop (Desktop): Desktop to use
            task (str): Task to accomplish
            actor (Actor): Actor to use
            history (List[Step]): History of steps taken

        Returns:
            Tuple[Optional[Step], bool]: A tuple containing the step taken and whether the task is complete
        """
        try:
            # Check to see if the task has been cancelled
            if task.remote:
                task.refresh()
            if (
                task.status == TaskStatus.CANCELING
                or task.status == TaskStatus.CANCELED
            ):
                console.print(f"task is {task.status}", style="red")
                if task.status == TaskStatus.CANCELING:
                    task.status = TaskStatus.CANCELED
                    task.save()
                return None, True

            console.print("taking action...", style="white")

            step = actor.act(task, device, history)

            if not step or not step.action:
                console.print("No step is found, skipping...", style="yellow")
                return None, False

            if step.task:
                task = step.task

            # The agent will return 'result' if it believes it's finished
            if step.action.name == "result":
                console.print("final result: ", style="green")
                console.print(JSON.from_data(step.action.parameters))
                task.post_message(
                    "assistant",
                    f"✅ I think the task is done, please review the result: {step.action.parameters['value']}",
                )
                task.status = TaskStatus.FINISHED
                task.save()
                task.record_action(
                    state=step.state,
                    prompt=step.prompt,
                    action=step.action,
                    tool=device.ref(),
                    result=step.action.parameters["value"],
                    agent_id=self.name(),
                    model=step.model_id,
                    action_opts=None,
                    metadata={
                        "input_tokens": step.in_tokens,
                        "output_tokens": step.out_tokens,
                        "thought": step.thought,
                    },
                )

                return step, True

            # Find the selected action in the tool
            action = device.find_action(step.action.name)
            console.print(f"found action: {action}", style="blue")
            if not action:
                console.print(f"action returned not found: {step.action.name}")
                raise SystemError("action not found")

            # Take the selected action
            try:
                action_response = device.use(action, **step.action.parameters)
            except Exception as e:
                raise ValueError(f"Trouble using action: {e}")

            console.print(f"action output: {action_response}", style="blue")
            if action_response:
                task.post_message(
                    "assistant", f"👁️ Result from taking action: {action_response}"
                )

            opts = None
            if step.action_opts:
                opts = [
                    ActionOpt(action=action, prompt=step.prompt)
                    for action in step.action_opts
                ]

            # Record the action for feedback and tuning
            task.record_action(
                state=step.state,
                prompt=step.prompt,
                action=step.action,
                tool=device.ref(),
                result=action_response,
                agent_id=self.name(),
                model=step.model_id,
                action_opts=opts,
                metadata={
                    "input_tokens": step.in_tokens,
                    "output_tokens": step.out_tokens,
                    "thought": step.raw_response,
                },
            )

            return step, False

        except Exception as e:
            print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:
        """Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        """
        return [Desktop]

    @classmethod
    def config_type(cls) -> Type[CUAConfig]:
        """Type of config

        Returns:
            Type[CUAConfig]: Config type
        """
        return CUAConfig

    @classmethod
    def from_config(cls, config: CUAConfig) -> "CUA":
        """Create an agent from a config

        Args:
            config (CUAConfig): Agent config

        Returns:
            CUA: The agent
        """
        return CUA()

    @classmethod
    def default(cls) -> "CUA":
        """Create a default agent

        Returns:
            CUA: The agent
        """
        return CUA()

    @classmethod
    def init(cls) -> None:
        """Initialize the agent class"""
        # <INITIALIZE AGENT HERE>
        return


Agent = CUA
