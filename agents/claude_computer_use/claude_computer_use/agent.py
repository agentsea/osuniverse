import base64
import logging
import os
import time
import traceback
from datetime import datetime
from io import BytesIO
from typing import Any, Final, List, Optional, Tuple, Type, cast

from agentdesk.device_v1 import Desktop
from anthropic import Anthropic
from anthropic.types.beta import (
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from devicebay import Device
from pydantic import BaseModel
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState, V1Action
from surfkit.agent import TaskAgent
from taskara import Task, TaskStatus
from tenacity import before_sleep_log, retry, stop_after_attempt
from toolfuse.util import AgentUtils

from .anthropic import ToolResult, make_api_tool_result, response_to_params

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", str(logging.DEBUG))))

console = Console(force_terminal=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise ValueError("Please set the ANTHROPIC_API_KEY in your environment.")
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class ClaudeComputerUseConfig(BaseModel):
    pass


class ClaudeComputerUse(TaskAgent):  # type: ignore
    """A GUI desktop agent that slices up the image"""

    def solve_task(
        self,
        task: Task,
        device: Optional[Device] = None,  # type: ignore
        max_steps: int = 30,
    ) -> Task:
        """Solve a task

        Args:
            task (Task): Task to solve.
            device (Device): Device to perform the task on.
            max_steps (int, optional): Max steps to try and solve. Defaults to 30.

        Returns:
            Task: The task
        """

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

        # Get info about the desktop
        info = device.info()
        screen_size = info["screen_size"]
        console.print(f"Desktop info: {screen_size}")

        # Define Anthropic Computer Use tool. Refer to the docs at https://docs.anthropic.com/en/docs/build-with-claude/computer-use#computer-tool
        self.tools = [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": screen_size["x"],
                "display_height_px": screen_size["y"],
                "display_number": 1,
            },
        ]

        console.print("tools: ", style="purple")
        console.print(JSON.from_data(self.tools))

        # Create our thread and start with the task description and system prompt
        messages: list[BetaMessageParam] = []
        messages.append(
            {
                "role": "user",
                "content": [
                    BetaTextBlockParam(
                        type="text", text=task.description if task.description else ""
                    )
                ],
            }
        )

        # The following prompt is a modified copy of the prompt from Anthropic's Computer Use Demo project
        # Some other code lines in this file are also copied from Anthropic's Computer Use Demo project
        # Original file: https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo/computer_use_demo/tools

        SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
        * You are utilising an Linux virtual machine of screen size {screen_size} with internet access.
        * To open firefox, please just click on the web browser (globe) icon.
        * When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
        * When using your computer function calls, they take a while to run and send back to you.
        * The ONLY tool you can use is the Computer Use tool. Therefore, whenever you are asked to do something, you should use the Computer Use tool to do it (including opening ternimal, opening and editing files, and other operations that can be done with a mouse and keyboard if necessary).
        * The current date is {datetime.today().strftime("%A, %B %-d, %Y")}.
        </SYSTEM_CAPABILITY>

        <IMPORTANT>
        * When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
        * If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
        * When you open Google and see a cookie consent popup, click on the "Accept all" button (in any language). If the button is not visible, scroll down until you see it. Before scrolling, move the mouse closer to the header of the cookie consent popup.
        * ALWAYS close the cookie consent popup on ANY website where you see it before continuing.
        </IMPORTANT>"""

        self.system = BetaTextBlockParam(
            type="text",
            text=f"{SYSTEM_PROMPT}",
        )

        self.action_mapping = {
            "key": "hot_key",
            "type": "type_text",
            "mouse_move": "move_mouse",
            "left_click": "click",
            "left_click_drag": "drag_mouse",
            "right_click": "click",
            "middle_click": "click",
            "double_click": "double_click",
            "screenshot": "take_screenshots",
            "cursor_position": "mouse_coordinates",
        }

        for i in range(max_steps):
            console.print(f"-------step {i + 1}", style="green")

            try:
                messages, done = self.take_action(device, task, messages)
            except Exception as e:
                console.print(f"Error: {e}", style="red")
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.save()
                task.post_message("assistant", f"❗ Error taking action: {e}")
                return task

            if done:
                console.print("task is done", style="green")
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
        device: Desktop,
        task: Task,
        messages: list[BetaMessageParam],
    ) -> Tuple[list[BetaMessageParam], bool]:
        """Take an action

        Args:
            device (Desktop): Desktop to use
            task (str): Task to accomplish
            messages: Messages (LLM exchange thread) for the task

        Returns:
            bool: Whether the task is complete
        """
        try:
            # Check to see if the task has been cancelled
            if task.remote:
                task.refresh()
            console.print("Task status: ", task.status.value)
            if (
                task.status == TaskStatus.CANCELING
                or task.status == TaskStatus.CANCELED
            ):
                console.print(f"Task is {task.status}", style="red")
                if task.status == TaskStatus.CANCELING:
                    task.status = TaskStatus.CANCELED
                    task.save()
                return messages, True

            console.print("Starting action...", style="yellow")

            messages = self._maybe_filter_to_n_most_recent_images(messages, 3, 2)

            model = os.getenv("SURFKIT_AGENT_MODEL", "claude-3-5-sonnet-20241022")
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=4096,
                messages=messages,
                model=model,
                system=[self.system],
                tools=self.tools,
                betas=["computer-use-2024-10-22"],
            )

            try:
                response = raw_response.parse()
                response_params = response_to_params(response)

                messages.append(
                    {
                        "role": "assistant",
                        "content": response_params,
                    }
                )

            except Exception as e:
                console.print(f"Response failed to parse: {e}", style="red")
                raise

            # The agent will return 'end_turn' if it believes it's finished
            if response.stop_reason == "end_turn":
                console.print("Final result: ", style="green")
                console.print(JSON.from_data(response_params[0]))

                task.post_message(
                    "assistant",
                    f"✅ I think the task is done, please review the result: {response_params[0]['text']}",
                )
                task.status = TaskStatus.FINISHED
                task.save()

                metadata: dict[str, Any] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                screenshot_img = device.take_screenshots()[0]
                task.record_action(
                    state=EnvState(images=[screenshot_img]),
                    action=V1Action(
                        name="result", parameters={"value": response_params[0]["text"]}
                    ),
                    tool=device.ref(),
                    metadata=metadata,
                )

                return messages, True

            tool_result_content: list[BetaToolResultBlockParam] = []

            thought = ""
            for content_block in response_params:
                if content_block["type"] == "text":
                    task.post_message("assistant", f"👁️ {content_block.get('text')}")
                    console.print(f"👁️ {content_block.get('text')}", style="blue")
                    thought += content_block.get("text") + "\n"
                elif content_block["type"] == "tool_use":
                    input_args = cast(dict[str, Any], content_block["input"])

                    action_name = self.action_mapping[input_args["action"]]
                    action_params = input_args.copy()
                    if input_args["action"] == "right_click":
                        action_params["button"] = "right"
                    if input_args["action"] == "middle_click":
                        action_params["button"] = "middle"

                    console.print(
                        f"Found action: {action_name} with params: {input_args}",
                        style="blue",
                    )

                    task.post_message(
                        "assistant",
                        f"▶️ Taking action '{action_name}' with parameters: {input_args}",
                    )

                    del action_params["action"]

                    # Find the selected action in the tool
                    action = device.find_action(action_name)
                    console.print(f"Found action: {action}", style="blue")
                    if not action:
                        console.print(f"Action returned not found: {action_name}")
                        raise SystemError("action not found")

                    # Take the selected action
                    try:
                        if (
                            action_name != "take_screenshots"
                        ):  # Do not execute if the action is screenshot, coz we take the screenshot later anyway
                            action_params = self._get_mapped_action_params(
                                action_name, action_params
                            )
                            action_response = device.use(action, **action_params)

                            console.print(
                                f"action output: {action_response}", style="blue"
                            )

                            if action_response:
                                task.post_message(
                                    "assistant",
                                    f"👁️ Result from taking action: {action_response}",
                                )
                    except Exception as e:
                        raise ValueError(f"Trouble using action: {e}")

                    # Take the screenshot after executing the action, or if the action itself is screenshot
                    screenshot_img = device.take_screenshots()[0]
                    buffer = BytesIO()
                    screenshot_img.save(buffer, format="PNG")
                    image_data = buffer.getvalue()
                    buffer.close()
                    base64_image = base64.b64encode(image_data).decode("utf-8")

                    result = ToolResult(
                        output=None, error=None, base64_image=base64_image
                    )

                    task.post_message(
                        "assistant",
                        "Current Image",
                        images=[screenshot_img],
                        thread="debug",
                    )

                    tool_result_content.append(
                        make_api_tool_result(result, content_block["id"])
                    )

                    metadata: dict[str, Any] = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "thought": thought,
                    }
                    task.record_action(
                        state=EnvState(images=[screenshot_img]),
                        action=V1Action(name=action_name, parameters=action_params),
                        tool=device.ref(),
                        metadata=metadata,
                    )

            if not tool_result_content:
                return messages, True

            messages.append({"content": tool_result_content, "role": "user"})

            return messages, False

        except Exception as e:
            console.print("Exception taking action: ", e)
            traceback.print_exc()
            task.post_message("assistant", f"⚠️ Error taking action: {e} -- retrying...")
            raise e

    def _maybe_filter_to_n_most_recent_images(
        self,
        messages: list[BetaMessageParam],
        images_to_keep: int,
        min_removal_threshold: int,
    ) -> list[BetaMessageParam]:
        """
        With the assumption that images are screenshots that are of diminishing value as
        the conversation progresses, remove all but the final `images_to_keep` tool_result
        images in place, with a chunk of min_removal_threshold to reduce the amount we
        break the implicit prompt cache.
        """
        if images_to_keep == 0:
            return messages

        tool_result_blocks = cast(
            list[BetaToolResultBlockParam],
            [
                item
                for message in messages
                for item in (
                    message["content"] if isinstance(message["content"], list) else []
                )
                if isinstance(item, dict) and item.get("type") == "tool_result"
            ],
        )

        total_images = sum(
            1
            for tool_result in tool_result_blocks
            for content in tool_result.get("content", [])
            if isinstance(content, dict) and content.get("type") == "image"
        )

        images_to_remove = total_images - images_to_keep
        # for better cache behavior, we want to remove in chunks
        images_to_remove -= images_to_remove % min_removal_threshold

        for tool_result in tool_result_blocks:
            if isinstance(tool_result.get("content"), list):
                new_content = []
                for content in tool_result.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        if images_to_remove > 0:
                            images_to_remove -= 1
                            continue
                    new_content.append(content)
                tool_result["content"] = new_content

        return messages

    def _get_mapped_action_params(
        self, action_name: str, action_params: dict[str, Any]
    ) -> dict[str, Any]:
        if action_name != "screenshot":
            if (
                action_name in ["move_mouse", "drag_mouse"]
                and "coordinate" in action_params
            ):
                action_params["x"] = action_params["coordinate"][0]
                action_params["y"] = action_params["coordinate"][1]
                del action_params["coordinate"]
            if action_name == "hot_key" and "text" in action_params:
                keys = action_params["text"].split("+")
                keys = [key.strip().lower() for key in keys]
                keys = [key.replace("_", "") for key in keys]
                action_params["keys"] = keys
                del action_params["text"]
            if action_name == "press_key":
                action_params["key"] = action_params["key"].replace("_", "").lower()
        return action_params

    @classmethod
    def supported_devices(cls) -> List[Type[Device]]:  # type: ignore
        """Devices this agent supports

        Returns:
            List[Type[Device]]: A list of supported devices
        """
        return [Desktop]  # type: ignore

    @classmethod
    def config_type(cls) -> Type[ClaudeComputerUseConfig]:
        """Type of config

        Returns:
            Type[ClaudeComputerUseConfig]: Config type
        """
        return ClaudeComputerUseConfig

    @classmethod
    def from_config(cls, config: ClaudeComputerUseConfig) -> "ClaudeComputerUse":
        """Create an agent from a config

        Args:
            config (ClaudeComputerUseConfig): Agent config

        Returns:
            ClaudeComputerUse: The agent
        """
        return ClaudeComputerUse()

    @classmethod
    def default(cls) -> "ClaudeComputerUse":
        """Create a default agent

        Returns:
            ClaudeComputerUse: The agent
        """
        return ClaudeComputerUse()

    @classmethod
    def init(cls) -> None:
        """Initialize the agent class"""
        return


Agent = ClaudeComputerUse
