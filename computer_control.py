#!/usr/bin/env python3
import json
import subprocess
import sys
import time

import pyautogui

try:
    import Quartz
except Exception:
    Quartz = None


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


KEY_ALIASES = {
    "cmd": "command",
    "command": "command",
    "meta": "command",
    "win": "command",
    "windows": "command",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "option",
    "shift": "shift",
    "enter": "enter",
    "return": "enter",
    "esc": "esc",
    "escape": "esc",
    "space": "space",
    "tab": "tab",
    "delete": "delete",
    "backspace": "backspace",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


def get_screen_size():
    if Quartz is not None:
        display_id = Quartz.CGMainDisplayID()
        return {
            "width": int(Quartz.CGDisplayPixelsWide(display_id)),
            "height": int(Quartz.CGDisplayPixelsHigh(display_id)),
        }
    width, height = pyautogui.size()
    return {"width": int(width), "height": int(height)}


def normalize_key(key):
    normalized = str(key or "").strip().lower()
    return KEY_ALIASES.get(normalized, normalized)


def clamp_coordinate(value, maximum):
    numeric = int(round(float(value)))
    if int(maximum) <= 0:
        return max(0, numeric)
    return max(0, min(numeric, max(0, int(maximum) - 1)))


def set_clipboard_text(text):
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


def action_move_cursor(args, screen_size):
    x = clamp_coordinate(args["x"], screen_size["width"])
    y = clamp_coordinate(args["y"], screen_size["height"])
    pyautogui.moveTo(x, y, duration=0)
    return {"x": x, "y": y}


def action_click(args, screen_size):
    x = clamp_coordinate(args["x"], screen_size["width"])
    y = clamp_coordinate(args["y"], screen_size["height"])
    button = str(args.get("button") or "left").strip().lower()
    clicks = max(1, min(int(args.get("clicks", 1)), 3))
    pyautogui.click(x=x, y=y, clicks=clicks, interval=0.08 if clicks > 1 else 0, button=button)
    return {"x": x, "y": y, "button": button, "clicks": clicks}


def action_type_text(args):
    text = str(args.get("text") or "")
    submit = bool(args.get("submit"))
    if not text:
        raise ValueError("Missing text")
    set_clipboard_text(text)
    pyautogui.hotkey("command", "v")
    if submit:
        pyautogui.press("enter")
    return {"text_length": len(text), "submit": submit}


def action_press_keys(args):
    keys = args.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("Missing keys")
    normalized_keys = [normalize_key(key) for key in keys if str(key or "").strip()]
    if not normalized_keys:
        raise ValueError("Missing keys")
    if len(normalized_keys) == 1:
        pyautogui.press(normalized_keys[0])
    else:
        pyautogui.hotkey(*normalized_keys)
    return {"keys": normalized_keys}


def action_scroll(args, screen_size):
    delta_y = int(args.get("delta_y", 0))
    x = args.get("x")
    y = args.get("y")
    if x is not None and y is not None:
        move_result = action_move_cursor({"x": x, "y": y}, screen_size)
    else:
        move_result = None
    pyautogui.scroll(delta_y)
    result = {"delta_y": delta_y}
    if move_result is not None:
        result["x"] = move_result["x"]
        result["y"] = move_result["y"]
    return result


def action_wait(args):
    duration_ms = max(0, min(int(args.get("duration_ms", 0)), 30_000))
    time.sleep(duration_ms / 1000)
    return {"duration_ms": duration_ms}


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: computer_control.py <tool_name> <json_args>")

    tool_name = sys.argv[1]
    args = json.loads(sys.argv[2] or "{}")
    screen_size = get_screen_size()

    if tool_name == "computer_move_cursor":
        payload = action_move_cursor(args, screen_size)
    elif tool_name == "computer_click":
        payload = action_click(args, screen_size)
    elif tool_name == "computer_type_text":
        payload = action_type_text(args)
    elif tool_name == "computer_press_keys":
        payload = action_press_keys(args)
    elif tool_name == "computer_scroll":
        payload = action_scroll(args, screen_size)
    elif tool_name == "computer_wait":
        payload = action_wait(args)
    else:
        raise ValueError(f"Unsupported tool: {tool_name}")

    output = {
        "ok": True,
        "tool": tool_name,
        "payload": payload,
        "screen_size": screen_size,
    }
    sys.stdout.write(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.stdout.write(json.dumps({
            "ok": False,
            "error": str(error),
        }, ensure_ascii=False))
        sys.exit(1)
