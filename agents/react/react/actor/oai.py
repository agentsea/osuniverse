# type: ignore

import os
from datetime import datetime
from typing import List, Optional

import dotenv
import json_repair
from agentdesk import Desktop
from litellm import completion
from rich.console import Console
from rich.json import JSON
from skillpacks import EnvState, V1Action
from skillpacks.img import image_to_b64
from taskara import Task
from toolfuse import AgentUtils

from threadmem import RoleThread

from .base import Actor, Step

console = Console(force_terminal=True)

dotenv.load_dotenv()


class OaiActor(Actor[Desktop]):
    """An actor that uses fine tuned openai models"""

    def __init__(self, model: Optional[str] = None):
        self.model = os.getenv("SURFKIT_AGENT_MODEL", "gpt-4o")
        self.base_url = os.getenv("SURFKIT_AGENT_MODEL_BASE_URL", None)

    def act(self, task: Task, device: Desktop, history: List[Step]) -> Step:
        thread = RoleThread()

        # Take a screenshot of the desktop and post a message with it
        screenshots = device.take_screenshots(count=1)
        s0 = screenshots[0]
        width, height = s0.size  # Get the dimensions of the screenshot
        console.print(f"Screenshot dimensions: {width} x {height}")

        device.merge(AgentUtils())
        tools = device.json_schema()

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
<SYSTEM_CAPABILITY>
* You are a highly experienced Linux user, capable of using a mouse and keyboard to interact with a computer, and take screenshots.
* You are utilising an Linux virtual machine of screen size {width}x{height} with internet access.
* To open Firefox, please just click on the web browser (globe) icon.
* The current date is {datetime.today().strftime("%A, %B %d, %Y")}.
</SYSTEM_CAPABILITY>

<TASK>
{task.description}
</TASK>

<INSTRUCTIONS>
* You are given the task and the action history with screenshots. For each new screenshot, you need to describe the current state, to consider the previous actions and screenshots, and to decide the next action. Also, describe what you expect to happen after the next action.
* AVOID repeating the same action if it doesn't lead to the expected result.
* Return your thoughts and the next action as a JSON object with the following format:

{{
    "reflection": "A brief reflection on whether the previous actions have been successful or not.",
    "observation": "A brief description of the current state of the environment.",
    "plan": "Your thoughts about what has to be done next, why, and what you expect to happen after the next action.",
    "action": "The next action adhering to the format decribed below"
}}

* The tools that are available to you (that is, actions you can perform) are the following:

{JSON.from_data(tools)}

* ALWAYS return the action in the format decribed above.
* Return ONLY the content of the JSON object, not the surrounding text or any other characters, like ```json or ```. ONLY the JSON object.
</INSTRUCTIONS>

<IMPORTANT>
* You are given the task and the action history with the current state screenshot. For each new screenshot, you need to describe the current state, to consider the previous actions, and to decide the next action. 
* When you open Google and see a cookie consent popup, click on the "Accept all" button (in any language). If the button is not visible, scroll down until you see it. Before scrolling, move the mouse closer to the header of the cookie consent popup.
* ALWAYS close the cookie consent popup on ANY website where you see it before continuing.
</IMPORTANT>

<EXAMPLE>
{{
    "reflection": "I have opened the website, and now the cookie consent popup is on the screen.",
    "observation": "The cookie consent popup is visible.",
    "plan": "I need to close it by clicking on the 'Accept all' button.",
    "action": {{"name": "click", "parameters": {{"x": 137, "y": 263, "button": "left"}}}}
}}
</EXAMPLE>

<EXAMPLE>
{{
    "reflection": "I have opened the Wikipedia web page.",
    "observation": "The Wikipedia web page is visible. The search bar is visible and in focus.",
    "plan": "I need to type the search query 'Finland' into the search bar.",
    "action": {{"name": "type_text", "parameters": {{"text": "Finland"}}}}
}}
</EXAMPLE>

<EXAMPLE>
{{
    "reflection": "The date is visible in the top right corner, so I need to return it to the user.",
    "observation": "The desktop is empty; the current date is visible in the top right corner.",
    "plan": "I need to return the current date to the user.",
    "action": {{"name": "result", "parameters": {{"value": "The current date is January 1, 2025"}}}}
}}
</EXAMPLE>
""",
                    }
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

        completion_response = completion(
            model=self.model,
            messages=messages,
            max_tokens=1000,
            base_url=self.base_url,
        )

        try:
            content = completion_response.choices[0].message.content
            console.print("Response content: ", content, style="green")
            content = json_repair.loads(content)
            reflection = content["reflection"]
            observation = content["observation"]
            plan = content["plan"]
            action = V1Action(
                name=content["action"]["name"],
                parameters=content["action"]["parameters"],
            )
            console.print("action selection: ", style="white")
            console.print(JSON.from_data(action.model_dump()))

            thought = f"🤔 {reflection} 👁️ {observation} 💡 {plan}"
            task.post_message("assistant", thought)
            task.post_message(
                "assistant",
                f"▶️ Taking action '{action.name}' with parameters: {action.parameters}",
            )

        except Exception as e:
            console.print(f"Response failed to parse: {e}", style="red")
            raise

        step = Step(
            state=EnvState(images=screenshots),
            action=action,
            thought=thought,
            raw_response=completion_response.choices[0].message.content,
            task=task,
            thread=thread,
            model_id=self.model,
            in_tokens=completion_response.usage.prompt_tokens,
            out_tokens=completion_response.usage.completion_tokens,
        )

        return step
