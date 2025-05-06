import base64
import random
import subprocess
import time
import uuid
import os
from io import BytesIO
from typing import List

from agentdesk import Desktop
from PIL import Image
from rich.console import Console
from surfkit import solve
from taskara import Task, TaskStatus

from threadmem import RoleMessage

from ..config import Config
from ..data.testcase import CommandOutputCheck, TestCase
from ..data.testcaserun import CommandOutputCheckResult, Step, TestCaseRun
from ..runners.base import BaseRunner

console = Console()

TRACKER_IMAGE = "us-central1-docker.pkg.dev/agentsea-dev/taskara/api:884e381"


class SurfkitAgentRunner(BaseRunner):
    def __init__(self):
        super().__init__()

    def run(self, testcase: TestCase, config: Config) -> TestCaseRun:
        # 1. Create a desktop with a random name from a given image and execute the setup script
        desktop_name = None
        for i_try in range(5):
            try:
                desktop_name = f"surfkit_desktop_{uuid.uuid4()}"
                console.print(
                    f"🚀 Creating a desktop with name {desktop_name}...",
                    style="dim",
                )
                desktop = Desktop.docker(
                    name=desktop_name, image=testcase.desktop_image
                )
                break
            except Exception as e:
                n = random.randint(0, 10)
                console.print(
                    f"🚀 Desktop {desktop_name} creation failed: {e}. Retrying in {n} second(s)...",
                    style="yellow",
                )
                self.delete_desktop(desktop_name)
                time.sleep(n)
                if i_try == 4:
                    console.print(
                        "Giving up to create the desktop after 5 tries",
                        style="bold red",
                    )
                    raise e
        console.print(f"🚀 Desktop {desktop_name} created", style="bold green")
        console.print(f"🚀 Running command `{testcase.setup_cmd}`...", style="dim")
        desktop.exec(testcase.setup_cmd)  # type: ignore
        console.print(f"🚀 Command `{testcase.setup_cmd}` executed", style="bold green")

        # 2. Create a new tracker
        tracker_name = None
        for i_try in range(5):
            try:
                tracker_name = f"surfkit_tracker_{uuid.uuid4()}"
                console.print(
                    f"🚀 Creating a tracker with name {tracker_name}...", style="dim"
                )
                subprocess.run(
                    [
                        "surfkit",
                        "create",
                        "tracker",
                        "--name",
                        tracker_name,
                        "--image",
                        TRACKER_IMAGE,
                    ]
                )
                time.sleep(5)
                break
            except Exception as e:
                n = random.randint(0, 10)
                console.print(
                    f"🚀 Tracker {tracker_name} creation failed: {e}. Retrying in {n} second(s)...",
                    style="yellow",
                )
                self.delete_tracker(tracker_name)
                time.sleep(n)
                if i_try == 4:
                    console.print(
                        "Giving up to create the tracker after 5 tries",
                        style="bold red",
                    )
                    self.delete_desktop(desktop_name)
                    raise e
        console.print(f"🚀 Tracker {tracker_name} created", style="bold green")

        testcaserun = TestCaseRun.from_testcase(testcase, config)

        # 2. Run `solve` using a correct agent (depending on the config)
        task_description = testcase.task
        task: Task | None = None
        for i_try in range(5):
            try:
                os.environ["SURFKIT_AGENT_MODEL"] = config.agent_model
                if config.agent_model_base_url is not None:
                    os.environ["SURFKIT_AGENT_MODEL_BASE_URL"] = config.agent_model_base_url
                task = solve(
                    task_description,
                    agent_file=config.agent_yaml,
                    device=desktop_name,
                    tracker=tracker_name,
                    max_steps=config.max_steps[testcase.level],
                    kill=True,
                    local_keys=True,
                )
                task.refresh()
                if task.status == TaskStatus.ERROR:
                    console.print(
                        f"‼️  Task status: {task.status} {task.error}",
                        style="bold red",
                    )
                    raise ValueError(f"Task failed: {task.status} {task.error}")
                else:
                    console.print(f"🚀 Task status: {task.status}", style="bold green")
                break

            except Exception as e:
                n = random.randint(0, 10)
                console.print(
                    f"🚀 Task creation failed: {e}. Retrying in {n} second(s)...",
                    style="yellow",
                )
                time.sleep(n)
                if i_try == 4:
                    console.print(
                        "Giving up to create the task after 5 tries", style="bold red"
                    )
                    self.delete_desktop(desktop_name)
                    self.delete_tracker(tracker_name)
                    raise e
        console.print("🚀 Task is created", style="bold green")

        if task is None:
            return testcaserun

        # 3. Retrieve the trajectory from the threads/episode and store it in the testcaserun
        console.print("🚀 Waiting for the task to be done", style="bold blue")
        task.wait_for_done()
        console.print(f"🚀 Task is done with status {task.status}", style="bold green")

        # Create TestCaseRun using the factory method
        testcaserun.status = task.status.value

        episode = task.episode
        actions = episode.actions  # type: ignore
        if (
            len(actions) != 0
        ):  # If the task is properly instrumented, we can use the actions
            console.print(f"Found {len(actions)} actions", style="bold blue")
            for action in actions:
                testcaserun.add_step(
                    Step(
                        action.created,
                        str(action.action),
                        action.metadata["thought"]
                        if "thought" in action.metadata
                        else "",
                        action.state.images[0]  # type: ignore
                        if len(action.state.images) > 0  # type: ignore
                        else None,
                    )
                )
                testcaserun.input_tokens += (
                    action.metadata["input_tokens"]
                    if "input_tokens" in action.metadata
                    else 0
                )
                testcaserun.output_tokens += (
                    action.metadata["output_tokens"]
                    if "output_tokens" in action.metadata
                    else 0
                )
        else:  # If the task is not properly instrumented, we use the messages,
            # but it's not guaranteed to be in order and generally has more noise
            threads = task.threads
            console.print(f"Found {len(threads)} threads", style="bold blue")
            for thread in threads:
                console.print(
                    f"Found {len(thread.messages())} messages", style="bold blue"
                )
                for message in thread.messages():
                    testcaserun.add_step(
                        Step(
                            message.created,
                            message.text,
                            "",
                            message.images[0]  # type: ignore
                            if len(message.images) > 0  # type: ignore
                            else None,
                        )
                    )

        # Grab the latest screenshot & the potential result action if any
        last_state_img: Image.Image = desktop.take_screenshots(count=1, delay=0.0)[0]  # type: ignore
        last_state_base64: str = self.pil_image_to_data_uri(last_state_img)  # type: ignore
        potential_result_actions = [
            action.action.parameters["value"]
            for action in sorted(actions, key=lambda x: x.created, reverse=True)
            if action.action.name == "result" and "value" in action.action.parameters
        ]
        if len(potential_result_actions) == 0:  # let's try to find the action in chat
            threads = task.threads
            messages: List[RoleMessage] = []
            for thread in threads:
                messages.extend(thread.messages())
            for message in sorted(messages, key=lambda x: x.created, reverse=True):
                if "result" in message.text:
                    potential_result_actions.append(message.text)
        console.print(
            f"🚀 Potential results: {potential_result_actions}",
            style="yellow",
        )

        if len(potential_result_actions) != 0:
            testcaserun.result = Step(
                time.time(),
                potential_result_actions[0],
                "",
                last_state_base64,
            )
        else:
            testcaserun.result = Step(
                time.time(),
                "No result found; run is complete.",
                "",
                last_state_base64,
            )

        first_step = testcaserun.trajectory[0]
        last_step = testcaserun.trajectory[-1]
        duration_seconds = last_step.timestamp - first_step.timestamp
        console.print(
            f"🚀 Run completed in {duration_seconds:.2f} seconds", style="bold green"
        )
        console.print(
            f"🚀 Trajectory is created with {len(testcaserun.trajectory)} steps",
            style="bold green",
        )

        # 4. Run commands and check the output (if any)
        for check in testcase.checks:
            if isinstance(check, CommandOutputCheck):
                result: str = desktop.exec(check.command)  # type: ignore
                command_output_check_result = CommandOutputCheckResult(
                    command=check.command,  # type: ignore
                    output=result,  # type: ignore
                )
                testcaserun.command_output_check_results.append(
                    command_output_check_result
                )

        # 5. Delete the desktop and tracker
        self.delete_desktop(desktop_name)
        self.delete_tracker(tracker_name)

        # 6. Return the trajectory & command output check results
        testcaserun.trajectory.sort(key=lambda x: x.timestamp)
        return testcaserun

    def pil_image_to_data_uri(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()  # the raw PNG bytes
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def delete_tracker(self, tracker_name: str | None):
        console.print(f"🗑️ Deleting the tracker {tracker_name}", style="dim")
        if tracker_name is None:
            return
        try:
            subprocess.run(["docker", "rm", "-v", "-f", tracker_name])
            console.print(f"🗑️ Tracker {tracker_name} deleted", style="dim")
            _ = subprocess.run(
                ["surfkit", "list", "trackers"], capture_output=True, text=True
            )
            console.print("🗑️ Cleaned up the trackers list", style="dim")
        except Exception as e:
            console.print(
                f"🗑️ Tracker {tracker_name} deletion failed: {e}", style="dim"
            )

    def delete_desktop(self, desktop_name: str | None):
        console.print(f"🗑️ Deleting the desktop {desktop_name}", style="dim")
        if desktop_name is None:
            return
        try:
            subprocess.run(["docker", "rm", "-v", "-f", desktop_name])
            console.print(f"🗑️ Desktop {desktop_name} deleted", style="dim")
            _ = subprocess.run(
                ["surfkit", "list", "devices"], capture_output=True, text=True
            )
            console.print("🗑️ Cleaned up the devices list", style="dim")
        except Exception as e:
            console.print(
                f"🗑️ Desktop {desktop_name} deletion failed: {e}", style="dim"
            )
