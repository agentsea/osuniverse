# type: ignore

import json
import os
from typing import List, Optional

import dotenv
from agentdesk import Desktop
from PIL import Image
from rich.console import Console
from skillpacks import EnvState, V1Action
from skillpacks.img import image_to_b64
from taskara import Task

from .action_parser import parse_action
from .base import Actor, Step
from .utils import create_response

console = Console(force_terminal=True)

dotenv.load_dotenv()

BASE64_ONE_PIXEL = image_to_b64(Image.new("RGB", (1, 1), (255, 255, 255)))


class OaiActor(Actor[Desktop]):
    """An actor that uses fine tuned openai models"""

    def __init__(self, model: Optional[str] = None):
        self.model = os.getenv("SURFKIT_AGENT_MODEL", "computer-use-preview-2025-03-11")
        self.tools = [
            {
                "type": "computer-preview",
                "display_width": 1280,
                "display_height": 800,
                "environment": "linux",
            },
        ]
        self.items = []
        self.last_call_id = None
        self.last_action_type = None
        self.last_reasoning = None
        self.keep_images = 5

    def clean_up_old_screenshots(self):
        """Remove all screenshots from the items list except for the last self.keep_images screenshots."""
        n = self.keep_images
        total_items = len(self.items)
        for i in range(total_items, 0, -1):
            item = self.items[i - 1]
            if "type" in item and item["type"] == "computer_call_output":
                n -= 1
                if n < 0:
                    item["output"]["image_url"] = BASE64_ONE_PIXEL

    def record_result_of_previous_action(self, device: Desktop):
        """Record the result of the previous action."""

        screenshots = device.take_screenshots(count=1)
        s0 = screenshots[0]
        width, height = s0.size  # Get the dimensions of the screenshot
        console.print(f"Screenshot dimensions: {width} x {height}")

        if self.last_action_type == "computer_call":
            screenshot_base64 = image_to_b64(screenshots[0])
            call_output = {
                "type": "computer_call_output",
                "call_id": self.last_call_id,
                "output": {
                    "type": "input_image",
                    "image_url": f"{screenshot_base64}",
                },
            }
            self.items.append(call_output)
        elif self.last_action_type == "function_call":
            function_output = {
                "type": "function_call_output",
                "call_id": self.last_call_id,
                "output": "success",  # hard-coded output for demo
            }
            self.items.append(function_output)
        else:
            raise ValueError(f"Unknown action type: {self.last_action_type}")

        self.last_action_type = None
        self.last_call_id = None

    def handle_item(
        self, item: dict, device: Desktop, task: Task, full_response: dict
    ) -> Optional[Step]:
        """Handle each item; may cause a computer action + screenshot."""

        # Take a screenshot of the desktop and post a message with it
        screenshots = device.take_screenshots(count=1)
        s0 = screenshots[0]
        width, height = s0.size  # Get the dimensions of the screenshot
        console.print(f"Screenshot dimensions: {width} x {height}")

        action = None

        if item["type"] == "message":
            console.print("Message: ", style="green")
            console.print(item["content"][0]["text"], style="green")

            action = V1Action(
                name="result",
                parameters={"value": item["content"][0]["text"]},
            )

        if item["type"] == "reasoning":
            console.print("Reasoning: ", style="yellow")
            console.print(item["summary"], style="yellow")
            self.last_reasoning = item["summary"]
            return None

        if item["type"] == "function_call":
            action_type, action_args = item["name"], json.loads(item["arguments"])
            console.print("Function call: ", style="blue")
            console.print(f"{action_type}({action_args})", style="blue")

            self.last_action_type = "function_call"
            self.last_call_id = item["call_id"]

            action = parse_action(action_type, action_args)

        if item["type"] == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            console.print("Computer call: ", style="blue")
            console.print(f"{action_type}({action_args})", style="blue")

            self.last_action_type = "computer_call"
            self.last_call_id = item["call_id"]

            action = parse_action(action_type, action_args)

        thought = (
            self.last_reasoning[0]["text"]
            if self.last_reasoning
            and len(self.last_reasoning) > 0
            and "text" in self.last_reasoning[0]
            else ""
        )
        step = Step(
            state=EnvState(images=screenshots),
            action=action,
            thought=thought,
            raw_response=item,
            task=task,
            thread=None,
            model_id=self.model,
            in_tokens=full_response["usage"]["input_tokens"],
            out_tokens=full_response["usage"]["output_tokens"],
        )

        self.last_reasoning = None

        return step

    def act(self, task: Task, device: Desktop, history: List[Step]) -> Step:
        if len(self.items) == 0:
            # add initial messages
            self.items.append(
                {
                    "role": "system",
                    "content": """You are a helpful assistant capable of navigating complex GUIs. 

REMEMBER: 
1. You can use mouse and keyboard to complete tasks on the computer.
2. If you don't see the screen, make a screenshot before taking any action.
3. You have to act autonomously; never ask user any questions; use your best judgement to figure out what to do next.
4. Never send any messages to the user unless it's a final result of the task.
5. When you return the next action, you should also return your reasoning for choosing this action.
""",
                }
            )
            self.items.append({"role": "user", "content": task.description})

        # add the result of the previous action to the items list (including screenshot)
        if self.last_call_id:
            self.record_result_of_previous_action(device)

        # clean up the previous screenshots, except for the last self.keep_images screenshots
        self.clean_up_old_screenshots()

        # At this point we assume that we have a full history in self.items
        # We just need to get the next action and generate a step that will be executed by the device in the main loop

        console.print("Sending request to the model...", style="white")
        response = create_response(
            model=self.model,
            input=self.items,
            tools=self.tools,
            truncation="auto",
            reasoning={"generate_summary": "concise"},
        )
        console.print("Response is received.", style="white")

        if "output" not in response:
            raise ValueError("No output from model")
        else:
            self.items += response["output"]
            for item in response["output"]:
                step = self.handle_item(item, device, task, response)
                if step:
                    return step
            return None
