# type: ignore

import re
from typing import Tuple

import json_repair
from rich.console import Console
from skillpacks.server.models import V1Action

console = Console()


def parse_action(content: str) -> Tuple[str, list[V1Action]]:
    """
    "action":
    * `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
    * `type`: Type a string of text on the keyboard.
    * `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
    * `left_click`: Click the left mouse button.
    * `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
    * `right_click`: Click the right mouse button.
    * `middle_click`: Click the middle mouse button.
    * `double_click`: Double-click the left mouse button.
    * `scroll`: Performs a scroll of the mouse scroll wheel.
    * `wait`: Wait specified seconds for the change to happen.
    * `terminate`: Terminate the current task and report its completion status.

    "parameters":
        "keys": {
            "description": "Required only by `action=key`.",
            "type": "array",
        },
        "text": {
            "description": "Required only by `action=type`.",
            "type": "string",
        },
        "coordinate": {
            "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
            "type": "array",
        },
        "pixels": {
            "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
            "type": "number",
        },
        "time": {
            "description": "The seconds to wait. Required only by `action=wait`.",
            "type": "number",
        },
        "status": {
            "description": "The status of the task. Required only by `action=terminate`.",
            "type": "string",
            "enum": ["success", "failure"],
        },

        Example:
        <tool_call>
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [1240, 783]}}
        </tool_call>
    """

    output = []
    console.print(f"Raw content: {content}")

    # Extract tool calls between <tool_call> and </tool_call> tags
    tool_call_pattern = r"<tool_call>\n(.*?)\n(?:</tool_call>|📐|⚗)"
    tool_call_matches = re.findall(tool_call_pattern, content, re.DOTALL)
    tools_used = []
    if tool_call_matches:
        for match in tool_call_matches:
            tools_used.append(match.strip())

    # Extract any text before the first tool call as thought
    pre_tool_pattern = r"^(.*?)(?=<tool_call>)"
    pre_tool_match = re.search(pre_tool_pattern, content, re.DOTALL)
    if pre_tool_match:
        thought = pre_tool_match.group(1).strip()

    for tool_used in tools_used:
        tool_used_json = json_repair.loads(tool_used)
        console.print(f"Found tool usage: {tool_used_json}", style="green")
        action_name = tool_used_json["arguments"]["action"]
        parameters = {}

        if action_name == "key":
            action_name = "hot_key"
            parameters["keys"] = tool_used_json["arguments"]["keys"]
        elif action_name == "type":
            action_name = "type_text"
            parameters["text"] = tool_used_json["arguments"]["text"]
        elif action_name == "mouse_move":
            action_name = "move_mouse"
            parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
            parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "left_click":
            action_name = "click"
            parameters["button"] = "left"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "left_click_drag":
            action_name = "drag_mouse"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "right_click":
            action_name = "click"
            parameters["button"] = "right"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "middle_click":
            action_name = "click"
            parameters["button"] = "middle"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "double_click":
            action_name = "double_click"
            parameters["button"] = "left"
            if "coordinate" in tool_used_json["arguments"]:
                parameters["x"] = tool_used_json["arguments"]["coordinate"][0]
                parameters["y"] = tool_used_json["arguments"]["coordinate"][1]
        elif action_name == "scroll":
            action_name = "scroll"
            parameters["clicks"] = tool_used_json["arguments"]["pixels"] // 10
        elif action_name == "wait":
            action_name = "wait"
            parameters["seconds"] = tool_used_json["arguments"]["time"]
        elif action_name == "terminate":
            action_name = "result"
            parameters["value"] = (
                "Status: " + tool_used_json["arguments"]["status"] + " " + thought
            )

        console.print(f"Parsed Action: {action_name}", style="yellow")
        console.print(f"Parsed Params: {parameters}", style="blue")

        # Create the V1Action
        action = V1Action(name=action_name, parameters=parameters)
        output.append(action)

    return thought, output


if __name__ == "__main__":
    # Example 1: Key Action
    print("=== Example 1: key action ===")
    content_key = (
        "Thought: Let's press some keys\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "key", "keys": ["ctrl", "c"]}}\n'
        "</tool_call>"
    )
    actions_key = parse_action(content_key)
    print("Actions (key):", actions_key)
    print("")

    # Example 2: Type Action
    print("=== Example 2: type action ===")
    content_type = (
        "Thought: I'll type some text\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "type", "text": "Hello World"}}\n'
        "</tool_call>"
    )
    actions_type = parse_action(content_type)
    print("Actions (type):", actions_type)
    print("")

    # Example 3: Mouse Move Action
    print("=== Example 3: mouse move action ===")
    content_move = (
        "Thought: Moving the mouse\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "mouse_move", "coordinate": [100, 200]}}\n'
        "</tool_call>"
    )
    actions_move = parse_action(content_move)
    print("Actions (move):", actions_move)
    print("")

    # Example 4: Left Click Action
    print("=== Example 4: left click action ===")
    content_click = (
        "Thought: Let's click something\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click"}}\n'
        "</tool_call>"
    )
    actions_click = parse_action(content_click)
    print("Actions (click):", actions_click)
    print("")

    # Example 5: Drag Action
    print("=== Example 5: drag action ===")
    content_drag = (
        "Thought: I'm dragging something\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "left_click_drag", "coordinate": [300, 400]}}\n'
        "</tool_call>"
    )
    actions_drag = parse_action(content_drag)
    print("Actions (drag):", actions_drag)
    print("")

    # Example 6: Right Click Action
    print("=== Example 6: right click action ===")
    content_right = (
        "Thought: Right clicking\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "right_click"}}\n'
        "</tool_call>"
    )
    actions_right = parse_action(content_right)
    print("Actions (right):", actions_right)
    print("")

    # Example 7: Middle Click Action
    print("=== Example 7: middle click action ===")
    content_middle = (
        "Thought: Middle clicking\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "middle_click"}}\n'
        "</tool_call>"
    )
    actions_middle = parse_action(content_middle)
    print("Actions (middle):", actions_middle)
    print("")

    # Example 8: Double Click Action
    print("=== Example 8: double click action ===")
    content_double = (
        "Thought: Double clicking\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "double_click"}}\n'
        "</tool_call>"
    )
    actions_double = parse_action(content_double)
    print("Actions (double):", actions_double)
    print("")

    # Example 9: Scroll Action
    print("=== Example 9: scroll action ===")
    content_scroll = (
        "Thought: Scrolling down\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "scroll", "pixels": -30}}\n'
        "</tool_call>"
    )
    actions_scroll = parse_action(content_scroll)
    print("Actions (scroll):", actions_scroll)
    print("")

    # Example 10: Wait Action
    print("=== Example 10: wait action ===")
    content_wait = (
        "Thought: Waiting for a bit\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "wait", "time": 5}}\n'
        "</tool_call>"
    )
    actions_wait = parse_action(content_wait)
    print("Actions (wait):", actions_wait)
    print("")

    # Example 11: Terminate Action
    print("=== Example 11: terminate action ===")
    content_terminate = (
        "Thought: Task completed successfully\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n'
        "</tool_call>"
    )
    actions_terminate = parse_action(content_terminate)
    print("Actions (terminate):", actions_terminate)
    print("")
