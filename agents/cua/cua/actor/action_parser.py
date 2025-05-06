# type: ignore

from typing import Optional

from rich.console import Console
from skillpacks.server.models import V1Action

console = Console()

CUA_KEY_TO_AGENTDESK_KEY = {
    "/": "/",
    "\\": "\\\\",
    "alt": "alt",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "backspace": "backspace",
    "capslock": "capslock",
    "cmd": "command",
    "ctrl": "ctrl",
    "delete": "delete",
    "end": "end",
    "enter": "enter",
    "esc": "escape",
    "home": "home",
    "insert": "insert",
    "option": "alt",
    "pagedown": "pagedown",
    "pageup": "pageup",
    "shift": "shift",
    "space": "space",
    "super": "command",
    "tab": "tab",
    "win": "win",
}

"""
AGENTDESK KEYS:

[ "\\t", "\\n", "\\r", " ", "!", '\\"', "\\#", "\\$", "\\%", "\\&", "\\'",
"\\(", "\\)", "\\*", "\\+", ",", "-", "\\.", "/", "0", "1", "2", "3",
"4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "\\?", "@",
"\\[", "\\\\", "\\]", "\\^", "\\_", "\\`", "a", "b", "c", "d", "e",
"f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
"t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "accept", "add",
"alt", "altleft", "altright", "apps", "backspace", "browserback",
"browserfavorites", "browserforward", "browserhome", "browserrefresh",
"browsersearch", "browserstop", "capslock", "clear", "convert", "ctrl",
"ctrlleft", "ctrlright", "decimal", "del", "delete", "divide", "down",
"end", "enter", "esc", "escape", "execute", "f1", "f10", "f11", "f12",
"f13", "f14", "f15", "f16", "f17", "f18", "f19", "f2", "f20", "f21",
"f22", "f23", "f24", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "final",
"fn", "help", "home", "insert", "left", "numlock", "pagedown", "pageup", "pause",
"pgdn", "pgup", "playpause", "prevtrack", "print", "printscreen",
"prntscrn", "prtsc", "prtscr", "return", "right", "scrolllock",
"select", "separator", "shift", "shiftleft", "shiftright", "sleep",
"space", "stop", "subtract", "tab", "up", "volumedown", "volumemute",
"volumeup", "win", "winleft", "winright", "yen", "command", "option",
"optionleft", "optionright" ]
"""


def parse_action(action_type: str, action_args: dict) -> Optional[V1Action]:
    """
    Parse the content into a list of V1Actions.

    Action	Example
    click(x, y, button="left")	        click(24, 150)
    double_click(x, y)	                double_click(24, 150)
    scroll(x, y, scroll_x, scroll_y)	scroll(24, 150, 0, -100)
    type(text)	                        type("Hello, World!")
    wait(ms=1000)	                    wait(2000)
    move(x, y)	                        move(24, 150)
    keypress(keys)	                    keypress(["CTRL", "C"])
    drag(path)	                        drag([[24, 150], [100, 200]])

    """
    action = None

    if action_type == "click":
        action = V1Action(
            name="click",
            parameters={
                "x": action_args["x"],
                "y": action_args["y"],
                "button": action_args["button"] if "button" in action_args else "left",
            },
        )
    elif action_type == "double_click":
        action = V1Action(
            name="double_click",
            parameters={
                "x": action_args["x"],
                "y": action_args["y"],
            },
        )
    elif action_type == "scroll":
        action = V1Action(
            name="scroll",
            parameters={
                "clicks": action_args["scroll_y"] // 10 * (-1),
            },
        )
    elif action_type == "type":
        action = V1Action(
            name="type_text",
            parameters={"text": action_args["text"]},
        )
    elif action_type == "move":
        action = V1Action(
            name="move_mouse",
            parameters={"x": action_args["x"], "y": action_args["y"]},
        )
    elif action_type == "keypress":
        keys = []
        for key in action_args["keys"]:
            low_key = key.lower()
            if low_key in CUA_KEY_TO_AGENTDESK_KEY:
                keys.append(CUA_KEY_TO_AGENTDESK_KEY[low_key])
            else:
                keys.append(low_key)
        action = V1Action(
            name="hot_key",
            parameters={"keys": keys},
        )
    elif action_type == "drag":
        coords = action_args["path"][0]
        if len(action_args["path"]) > 1:
            coords = action_args["path"][1]
        console.print(f"Dragging from {action_args['path'][0]} to {coords}")
        action = V1Action(
            name="drag_mouse",
            parameters={
                "x": coords["x"],
                "y": coords["y"],
            },
        )
    elif action_type == "wait":
        action = V1Action(
            name="wait",
            parameters={
                "seconds": action_args["ms"] // 1000 if "ms" in action_args else 1
            },
        )
    return action
