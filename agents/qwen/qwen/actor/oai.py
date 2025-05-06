# type: ignore

import os
from datetime import datetime
from typing import List, Optional, Tuple

import dotenv
from agentdesk import Desktop
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState, V1Action
from skillpacks.img import image_to_b64
from taskara import Task

from threadmem import RoleThread

from .action_parser import parse_action
from .base import Actor, Step

console = Console(force_terminal=True)

dotenv.load_dotenv()


class OaiActor(Actor[Desktop]):
    """An actor that uses fine tuned openai models"""

    def __init__(self, model: Optional[str] = None):
        self.model = os.getenv("SURFKIT_AGENT_MODEL", "qwen2.5-vl-72b-instruct")
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    def act(self, task: Task, device: Desktop, history: List[Step]) -> Step:
        thread = RoleThread()

        # Take a screenshot of the desktop and post a message with it
        screenshots = device.take_screenshots(count=1)
        s0 = screenshots[0]
        width, height = s0.size  # Get the dimensions of the screenshot
        console.print(f"Screenshot dimensions: {width} x {height}")

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
<SYSTEM_CAPABILITY>
* You are a highly experienced Linux user, capable of using a mouse and keyboard to interact with a computer, and take screenshots.
* You are utilising an Linux virtual machine of screen size 1024x768 with internet access.
* To open Firefox, please just click on the web browser (globe) icon.
* The current date is {datetime.today().strftime("%A, %B %d, %Y")}.
</SYSTEM_CAPABILITY>

<TASK>
{task.description}
</TASK>

<INSTRUCTIONS>
* You are given the task and the action history with screenshots. For each new screenshot, you need to describe the current state, to consider the previous actions and screenshots, and to decide the next action. Also, describe what you expect to happen after the next action.
* AVOID repeating the same action if it doesn't lead to the expected result.
* Return your thoughts as plain text and the next action in the format decribed below.
* For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{{\"name\": \"<function-name>\", \"arguments\": \"<args-json-object>\"}}
</tool_call>

* ALWAYS return the action in the format decribed above. Make sure to include the <tool_call> and </tool_call> tags.
</INSTRUCTIONS>

<IMPORTANT>
* You are given the task and the action history with the current state screenshot. For each new screenshot, you need to describe the current state, to consider the previous actions, and to decide the next action. 
* When you open Google and see a cookie consent popup, click on the "Accept all" button (in any language). If the button is not visible, scroll down until you see it. Before scrolling, move the mouse closer to the header of the cookie consent popup.
* ALWAYS close the cookie consent popup on ANY website where you see it before continuing.
</IMPORTANT>

<EXAMPLE>
I see that the cookie consent popup is visible. I need to close it by clicking on the "Accept all" button. I expect to see the cookie consent popup closed after the action.
<tool_call>
{{\"name\": \"computer_use\", \"arguments\": {{\"action\": \"left_click\", \"coordinate\": [100, 100]}}}}
</tool_call>
</EXAMPLE>
""",
                    },
                    {
                        "type": "text",
                        "text": """

<TOOLS>
{"type": "function", "function": {
    "name_for_human": "computer_use",
    "name": "computer_use", 
    "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
                   "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
                   "You must click on desktop icons to start applications.\n"
                   "* Some applications may take time to start or process actions, so you may need to wait and take "
                   "successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window "
                   "doesn't open, try wait and taking another screenshot.\n"
                   "* The screen's resolution is 1024x768.\n"
                   "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a "
                   "screenshot to determine the coordinates of the element before moving the cursor.\n"
                   "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting "
                   "your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
                   "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
                   "Don't click boxes on their edges unless asked.",
    "parameters": {
        "properties": {
            "action": {
                "description": "The action to perform. The available actions are:\n"
                             "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
                             "* `type`: Type a string of text on the keyboard.\n"
                             "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                             "* `left_click`: Click the left mouse button.\n"
                             "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                             "* `right_click`: Click the right mouse button.\n"
                             "* `middle_click`: Click the middle mouse button.\n"
                             "* `double_click`: Double-click the left mouse button.\n"
                             "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
                             "* `wait`: Wait specified seconds for the change to happen.\n"
                             "* `terminate`: Terminate the current task and report its completion status.",
                "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", 
                        "middle_click", "double_click", "scroll", "wait", "terminate"],
                "type": "string"
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array"
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string"
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. "
                             "Required only by `action=mouse_move` and `action=left_click_drag`.",
                "type": "array"
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. "
                             "Required only by `action=scroll`.",
                "type": "number"
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number"
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"]
            }
        },
        "required": ["action"],
        "type": "object"
    },
    "args_format": "Format the arguments as a JSON object."
}}
</TOOLS>
""",
                    },
                ],
            }
        ]

        for step in history:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the next action?",
                        },
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": step.raw_response,
                        },
                    ],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_b64(screenshots[0])},
                    },
                ],
            }
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
        )

        try:
            thought, actions = self._parse_response(completion)
            selection = self._select_action(actions)
            console.print("action selection: ", style="white")
            console.print(JSON.from_data(selection.model_dump()))

            task.post_message(
                "assistant",
                f"▶️ Taking action '{selection.name}' with parameters: {selection.parameters}",
            )

        except Exception as e:
            console.print(f"Response failed to parse: {e}", style="red")
            raise

        step = Step(
            state=EnvState(images=screenshots),
            action=selection,
            thought=thought,
            raw_response=completion.choices[0].message.content,
            task=task,
            thread=thread,
            model_id=self.model,
            in_tokens=completion.usage.prompt_tokens,
            out_tokens=completion.usage.completion_tokens,
        )

        return step

    def _parse_response(self, response: ChatCompletion) -> Tuple[str, List[V1Action]]:
        content = response.choices[0].message.content
        thought, output = parse_action(content)
        return thought, output

    def _select_action(self, actions: List[V1Action]) -> V1Action:
        console.print("action options: ", style="white")

        for i, act in enumerate(actions):
            console.print(f"Option {i + 1}:", style="yellow")
            console.print(JSON.from_data(act.model_dump()), style="blue")

        action = actions[0]
        return action
