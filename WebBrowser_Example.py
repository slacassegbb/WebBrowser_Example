from __future__ import annotations
import json
import time
import base64
import os
from typing import List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image, ImageDraw

import openai
from selenium import webdriver
from selenium.webdriver.common.by import By
import pyautogui

# ------------------------------------------
# (0) Environment Setup
# ------------------------------------------
load_dotenv()

# Azure OpenAI configuration for high-level tasks using environment variables.
openai.api_type = "azure"
openai.azure_endpoint = os.getenv("GPT4_AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("GPT4_AZURE_OPENAI_KEY")
openai.api_version = os.getenv("GPT4_AZURE_OPENAI_API_VERSION")

# -----------------------------------------------------------------------------
# (1) Data Structures
# -----------------------------------------------------------------------------
class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    error = "error"

class Subtask(BaseModel):
    id: str
    description: str
    agent_type: str  # Expected values: "Medium" or "Low"
    status: TaskStatus = TaskStatus.pending
    result: Optional[List[dict]] = None
    error_message: Optional[str] = None
    action: Optional[str] = None

    # For tasks that involve interacting with a webpage element:
    element_name: Optional[str] = None
    x_coordinate: Optional[int] = None
    y_coordinate: Optional[int] = None

    # Additional fields for low-level tasks:
    text_to_enter: Optional[str] = None
    key: Optional[str] = None
    end_x: Optional[int] = None
    end_y: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.id}: {self.description} ({self.agent_type})"


class SubtaskPair(BaseModel):
    medium: Subtask
    low: Subtask


class PairPlan(BaseModel):
    task: str
    pairs: List[SubtaskPair]


class RecommendationPair(BaseModel):
    success: bool
    goal_complete: bool
    Observation: str
    medium: Optional[Subtask] = None
    low: Optional[Subtask] = None


# -----------------------------------------------------------------------------
# (2) High-Level Agent
# -----------------------------------------------------------------------------
class HighLevelAgent:
    """
    High-Level Agent that decomposes a user goal into task pairs.
    """
    system_prompt = "You are the High-Level Agent. Decompose the goal into Medium and Low task pairs."

    def generate_task_breakdown(self, user_goal: str) -> PairPlan:
        """
        Generate a task breakdown plan given a user goal.
        """
        # Get the JSON schema for PairPlan to guide the LLM's output.
        plan_schema_str = json.dumps(PairPlan.model_json_schema(), indent=2)
        user_prompt = f"""
            You are the High-Level Agent. The user's goal is: '{user_goal}'.

            IMPORTANT:
            1) Follow this Pydantic JSON schema:
            {plan_schema_str}
            2) Use only these agent types: "Medium" and "Low".
            3) Each subtask pair should have:
            - A Medium task that inspects the page to extract all actionable elements (inputs, buttons, links) and returns the chosen element's name and its x,y coordinates.
            - A Low task that performs a specific UI action at that location.
            4) For Medium tasks, set the "action" field to "inspect".
            5) For Low tasks, set the "action" field to one of the following valid PyAutoGUI actions:
            "move_mouse", "single_click_at_location", "double_click_at_location", "enter_text_at_location",
            "press_enter", "select_text", "copy_text", "paste_text", "right_click_at_location", "press_key",
            "key_down", "key_up", "click_and_hold", "scroll_up", "scroll_down", "scroll_left", "scroll_right".
            6) All tasks start with "status": "pending".
            7) The browser will always be opened by default with the correct URL, so that should never be a task in the plan.
            8) Return ONLY valid JSON.

            Now produce the final JSON plan.
            """
        # Call the LLM for task breakdown.
        completion = openai.beta.chat.completions.parse(
            model="GPT4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=PairPlan
        )

        # Debug output: raw response (optional)
        raw = getattr(completion.choices[0].message, 'raw', "<no raw output>")
        print("DEBUG: High-Level Agent raw response:", raw)

        return completion.choices[0].message.parsed


# -----------------------------------------------------------------------------
# (3) Medium-Level Agent
# -----------------------------------------------------------------------------
class MediumLevelAgent:
    """
    Medium-Level Agent responsible for analyzing the page and selecting actionable elements.
    """
    system_prompt = (
        "You are the Medium-Level Agent. Analyze all extracted actionable elements and decide "
        "which element best fits the requested action."
    )

    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver

    def extract_actionable_elements(self) -> List[dict]:
        """
        Extracts all visible actionable elements (inputs, buttons, links, textareas) from the page,
        computing their absolute positions.
        """
        elements = self.driver.find_elements(By.XPATH, "//input | //button | //a | //textarea")
        actionable = []

        # Get the browser window's absolute position.
        win_pos = self.driver.get_window_position()
        # Calculate the browser chrome/title bar offset.
        title_bar_offset = self.driver.execute_script("return window.outerHeight - window.innerHeight;")

        for el in elements:
            if el.is_displayed():
                rect = self.driver.execute_script("""
                    let rect = arguments[0].getBoundingClientRect();
                    return {
                        left: rect.left, 
                        top: rect.top, 
                        width: rect.width, 
                        height: rect.height
                    };
                """, el)

                abs_x = int(rect["left"] + win_pos["x"])
                abs_y = int(rect["top"] + win_pos["y"] + title_bar_offset)
                center_x = abs_x + int(rect["width"] / 2)
                center_y = abs_y + int(rect["height"] / 2)

                actionable.append({
                    "tag": el.tag_name.lower(),
                    "id": el.get_attribute("id"),
                    "class": el.get_attribute("class"),
                    "name": el.get_attribute("name"),
                    "type": el.get_attribute("type"),
                    "placeholder": el.get_attribute("placeholder"),
                    "aria_label": el.get_attribute("aria-label"),
                    "x": center_x,
                    "y": center_y,
                    "width": int(rect["width"]),
                    "height": int(rect["height"])
                })

        return actionable

    def interpret_task(
        self, medium_subtask: Subtask, low_subtask: Subtask, actionable_elements: List[dict]
    ) -> None:
        """
        Uses the LLM to decide which element best matches the intended action.
        Updates the medium_subtask with the chosen element's details.
        """
        formatted_elements = json.dumps(actionable_elements, indent=2)
        user_prompt = f"""
            You are an assistant. Given the following JSON list of actionable elements extracted from a webpage,
            select the element that best matches the intent for the actions described below.

            Medium Task: {medium_subtask.description}
            Low Task: {low_subtask.description}

            The JSON list is:
            {formatted_elements}

            Criteria: Consider properties such as "placeholder", "aria_label", "name", and "innerText". For instance,
            if the intent is to interact with a search bar, choose an input element whose placeholder contains the word "search".

            Return ONLY a JSON object that fits the following model, updating only the fields "action", "element_name", "x_coordinate", and "y_coordinate":
            {{
            "action": "{medium_subtask.description}",
            "element_name": "<the chosen element's name (or class if name is null)>",
            "x_coordinate": <the element's x coordinate>,
            "y_coordinate": <the element's y coordinate>
            }}
            """

        print("DEBUG: Medium Agent prompt to LLM:")
        try:
            completion = openai.beta.chat.completions.parse(
                model="GPT4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=Subtask
            )
        except Exception as e:
            print("DEBUG: Exception during LLM call:", e)
            completion = None

        # Use fallback if LLM call fails or returns incomplete output.
        if completion is None:
            print("DEBUG: LLM call failed; using fallback element.")
            fallback = actionable_elements[0]
            parsed = Subtask(
                id=medium_subtask.id,
                description=medium_subtask.description,
                agent_type=medium_subtask.agent_type,
                action="move_mouse",  # default action fallback
                element_name=fallback.get("name") or fallback.get("class") or "",
                x_coordinate=fallback.get("x", 0),
                y_coordinate=fallback.get("y", 0)
            )
        else:
            raw = getattr(completion.choices[0].message, 'raw', "<no raw output>")
            parsed = completion.choices[0].message.parsed
            if parsed.x_coordinate is None or parsed.y_coordinate is None or not parsed.element_name:
                print("DEBUG: Parsed output incomplete; using fallback element.")
                fallback = actionable_elements[0]
                parsed = Subtask(
                    id=medium_subtask.id,
                    description=medium_subtask.description,
                    agent_type=medium_subtask.agent_type,
                    action="move_mouse",
                    element_name=fallback.get("name") or fallback.get("class") or "",
                    x_coordinate=fallback.get("x", 0),
                    y_coordinate=fallback.get("y", 0)
                )

        # Update medium task with the parsed values.
        medium_subtask.action = parsed.action
        medium_subtask.element_name = parsed.element_name
        medium_subtask.x_coordinate = parsed.x_coordinate
        medium_subtask.y_coordinate = parsed.y_coordinate

        print(f"DEBUG: Updated subtask {medium_subtask.id} => action={medium_subtask.action}, "
              f"element_name={medium_subtask.element_name}, x={medium_subtask.x_coordinate}, "
              f"y={medium_subtask.y_coordinate}")


# -----------------------------------------------------------------------------
# (4) Low-Level Agent using PyAutoGUI
# -----------------------------------------------------------------------------
class LowLevelAgent:
    """
    Low-Level Agent that executes UI interactions using PyAutoGUI.
    """
    system_prompt = "You are the Low-Level Agent. Execute UI interactions precisely using PyAutoGUI."

    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver  # Although provided, it's not used directly in low-level operations.

    @staticmethod
    def overlay_cursor_on_screenshot(screenshot_path: str, x: int, y: int, dot_radius: int = 10) -> None:
        """
        Overlays a red dot on the screenshot at the given (x, y) coordinates.
        """
        try:
            im = Image.open(screenshot_path)
            draw = ImageDraw.Draw(im)
            left_up_point = (x - dot_radius, y - dot_radius)
            right_down_point = (x + dot_radius, y + dot_radius)
            draw.ellipse([left_up_point, right_down_point], fill="red")
            im.save(screenshot_path)
            print(f"DEBUG: Red dot overlay added at ({x}, {y}) on screenshot {screenshot_path}")
        except Exception as e:
            print(f"DEBUG: Failed to overlay red dot: {e}")

    # ----------------------------
    # Low-Level Action Methods
    # ----------------------------
    def move_mouse(self, x: int, y: int, **kwargs) -> None:
        pyautogui.moveTo(x, y)
        pyautogui.click()
        print(f"Moved to {x}, {y}")
        time.sleep(2)

    def single_click_at_location(self, x: int, y: int, **kwargs) -> None:
        print("EXECUTING SINGLE CLICK")
        pyautogui.moveTo(x, y)
        pyautogui.click()
        time.sleep(2)

    def double_click_at_location(self, x: int, y: int, **kwargs) -> None:
        pyautogui.moveTo(x, y)
        pyautogui.doubleClick()
        print(f"Double-clicked at {x}, {y}")
        time.sleep(2)

    def enter_text_at_location(self, x: int, y: int, text_to_enter: Optional[str] = None, **kwargs) -> None:
        pyautogui.click(x, y)
        pyautogui.typewrite(text_to_enter)
        print(f"Entered text '{text_to_enter}' at {x}, {y}")
        time.sleep(2)

    def press_enter(self, x: int, y: int, **kwargs) -> None:
        pyautogui.press('enter')
        print("Pressed Enter key")
        time.sleep(2)

    def select_text(self, x: int, y: int, end_x: int, end_y: int, **kwargs) -> None:
        pyautogui.moveTo(x, y)
        pyautogui.dragTo(end_x, end_y, duration=0.2, button='left')
        print(f"Selected text from ({x}, {y}) to ({end_x}, {end_y})")
        time.sleep(2)

    def copy_text(self, x: int, y: int, **kwargs) -> None:
        pyautogui.hotkey('ctrl', 'c')
        print("Copied selected text to clipboard")
        time.sleep(2)

    def paste_text(self, x: int, y: int, **kwargs) -> None:
        pyautogui.hotkey('ctrl', 'v')
        print("Pasted text from clipboard")
        time.sleep(2)

    def right_click_at_location(self, x: int, y: int, **kwargs) -> None:
        pyautogui.moveTo(x, y)
        pyautogui.click(button='right')
        print(f"Right-clicked at {x}, {y}")
        time.sleep(2)

    def press_key(self, x: int, y: int, key: Optional[str] = None, **kwargs) -> None:
        pyautogui.press(key)
        print(f"Pressed key '{key}'")
        time.sleep(2)

    def key_down(self, x: int, y: int, key: Optional[str] = None, **kwargs) -> None:
        pyautogui.keyDown(key)
        print(f"Held down key '{key}'")
        time.sleep(2)

    def key_up(self, x: int, y: int, key: Optional[str] = None, **kwargs) -> None:
        pyautogui.keyUp(key)
        print(f"Released key '{key}'")
        time.sleep(2)

    def click_and_hold(self, x: int, y: int, **kwargs) -> None:
        pyautogui.moveTo(x, y)
        pyautogui.mouseDown()
        print(f"Clicked and holding at ({x}, {y})")
        time.sleep(1)
        pyautogui.mouseUp()
        print(f"Released mouse button at ({x}, {y})")
        time.sleep(2)

    def scroll_up(self, x: int, y: int, **kwargs) -> None:
        pyautogui.scroll(300)
        print("Scrolled up")
        time.sleep(2)

    def scroll_down(self, x: int, y: int, **kwargs) -> None:
        pyautogui.scroll(-300)
        print("Scrolled down")
        time.sleep(2)

    def scroll_left(self, x: int, y: int, **kwargs) -> None:
        pyautogui.hscroll(-300)
        print("Scrolled left")
        time.sleep(2)

    def scroll_right(self, x: int, y: int, **kwargs) -> None:
        pyautogui.hscroll(300)
        print("Scrolled right")
        time.sleep(2)

    def execute_instructions(self, instructions: List[dict]) -> dict:
        """
        Execute a series of low-level UI instructions. Each instruction is a dictionary
        containing the action and any required parameters.
        """
        print("DEBUG: Low-Level Agent received instructions:", instructions)
        for instr in instructions:
            action = instr.get("action")
            x = instr.get("x_coordinate")
            y = instr.get("y_coordinate")
            text_to_enter = instr.get("text_to_enter")
            key = instr.get("key")
            end_x = instr.get("end_x")
            end_y = instr.get("end_y")
            print(f"DEBUG: -> {action}, x={x}, y={y}")
            try:
                if action == "move_mouse":
                    self.move_mouse(x, y)
                elif action == "single_click_at_location":
                    self.single_click_at_location(x, y)
                elif action == "double_click_at_location":
                    self.double_click_at_location(x, y)
                elif action == "enter_text_at_location":
                    if text_to_enter is None:
                        raise Exception("Missing text_to_enter for enter_text_at_location")
                    self.enter_text_at_location(x, y, text_to_enter=text_to_enter)
                elif action == "press_enter":
                    self.press_enter(x, y)
                elif action == "select_text":
                    if None in (x, y, end_x, end_y):
                        raise Exception("Missing coordinates for select_text")
                    self.select_text(x, y, end_x, end_y)
                elif action == "copy_text":
                    self.copy_text(x, y)
                elif action == "paste_text":
                    self.paste_text(x, y)
                elif action == "right_click_at_location":
                    self.right_click_at_location(x, y)
                elif action == "press_key":
                    if key is None:
                        raise Exception("Missing key for press_key")
                    self.press_key(x, y, key=key)
                elif action == "key_down":
                    if key is None:
                        raise Exception("Missing key for key_down")
                    self.key_down(x, y, key=key)
                elif action == "key_up":
                    if key is None:
                        raise Exception("Missing key for key_up")
                    self.key_up(x, y, key=key)
                elif action == "click_and_hold":
                    self.click_and_hold(x, y)
                elif action == "scroll_up":
                    self.scroll_up(x, y)
                elif action == "scroll_down":
                    self.scroll_down(x, y)
                elif action == "scroll_left":
                    self.scroll_left(x, y)
                elif action == "scroll_right":
                    self.scroll_right(x, y)
                else:
                    print("DEBUG: Unknown or unhandled action:", action)
                    return {"success": False, "error_message": f"Unknown action: {action}"}
            except Exception as e:
                print("DEBUG: Low-level action failed:", str(e))
                return {"success": False, "error_message": str(e)}

        return {"success": True, "error_message": None}

    def evaluate(
        self, subtask: Subtask, screenshot_path: str, user_goal: str
    ) -> Tuple[str, bool, bool, Optional[RecommendationPair]]:
        """
        Uses the vision model to analyze a screenshot and determine if the low-level UI action
        was executed correctly and whether the overall user goal is met.
        """
        prompt_text = f"""
            Analyze the screenshot to determine if the low-level UI action '{subtask.action}' for the task 
            '{subtask.description}' was performed correctly. A small red dot in the image indicates the mouse 
            cursor's position. The intended target element is identified as '{subtask.element_name}'.

            Additionally, considering the overall user goal: '{user_goal}', determine whether the goal has been accomplished.

            If the low-level action was executed correctly and the overall goal is met, return the following JSON:
            {{
                "success": true,
                "goal_complete": true,
                "Observation": "A brief confirmation message."
            }}

            If the low-level action was executed correctly but the overall goal is not yet met, return:
            {{
                "success": true,
                "goal_complete": false,
                "Observation": "A brief confirmation message indicating that further actions are needed."
            }}

            If the action was not executed correctly, return a JSON object with recommendations:
            {{
                "success": false,
                "goal_complete": false,
                "Observation": "A brief explanation of the failure.",
                "medium": {{
                    "id": "<new_medium_id>",
                    "description": "The description of a new mid-level task to try based on the observation",
                    "agent_type": "Medium",
                    "action": null,
                    "element_name": "{subtask.element_name or 'the intended element based on observation'}"
                }},
                "low": {{
                    "id": "<new_low_id>",
                    "description": "The description of the new low-level task to try based on observation",
                    "agent_type": "Low",
                    "action": "One of: move_mouse, single_click_at_location, double_click_at_location, enter_text_at_location, press_enter, select_text, copy_text, paste_text, right_click_at_location, press_key, key_down, key_up, click_and_hold, scroll_up, scroll_down, scroll_left, scroll_right",
                    "element_name": "{subtask.element_name or 'the intended element based on observation'}"
                }}
            }}
            """
        print("DEBUG: Evaluating screenshot at path:", screenshot_path)

        # Read and encode the screenshot.
        with open(screenshot_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        content_images = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]

        try:
            response = openai.beta.chat.completions.parse(
                model="GPT4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            *content_images,
                        ],
                    }
                ],
                response_format=RecommendationPair
            )

            rec_pair: RecommendationPair = response.choices[0].message.parsed

            if rec_pair.success:
                message = f"Vision analysis indicates success: {rec_pair.Observation}"
                return message, True, rec_pair.goal_complete, None
            else:
                message = f"Vision analysis indicates failure: {rec_pair.Observation}"
                return message, False, False, rec_pair

        except Exception as e:
            print("Error processing the vision response:", str(e))
            return "An error occurred during evaluation.", False, False, None


# -----------------------------------------------------------------------------
# (5) Orchestrator for Subtask Pairs
# -----------------------------------------------------------------------------
class Orchestrator:
    """
    Orchestrates the execution of subtask pairs by coordinating the medium- and low-level agents.
    """
    def __init__(
        self, plan: PairPlan, medium_agent: MediumLevelAgent, low_agent: LowLevelAgent, user_goal: str
    ):
        self.plan = plan
        self.medium_agent = medium_agent
        self.low_agent = low_agent
        self.user_goal = user_goal

        # Ensure that the screenshots directory exists.
        os.makedirs("screenshots", exist_ok=True)

    def replan_pair(self, failed_pair: SubtaskPair, rec_pair: RecommendationPair) -> SubtaskPair:
        """
        Creates a new SubtaskPair using recommendations from the vision evaluation.
        """
        new_medium = rec_pair.medium
        new_low = rec_pair.low

        # Append a suffix to indicate a replanned subtask.
        new_medium.id = (new_medium.id + "_replan") if new_medium.id else failed_pair.medium.id + "_replan"
        new_low.id = (new_low.id + "_replan") if new_low.id else failed_pair.low.id + "_replan"
        new_medium.status = TaskStatus.pending
        new_low.status = TaskStatus.pending

        print(f"DEBUG: Replanned pair: {new_medium} and {new_low}")
        return SubtaskPair(medium=new_medium, low=new_low)

    def execute_pair(self, pair: SubtaskPair) -> None:
        """
        Executes a subtask pair: first running the medium-level task to determine target coordinates,
        then executing the low-level action and evaluating its outcome.
        """
        print("\nDEBUG: Executing subtask pair:")
        print(f"  Medium task: {pair.medium.description}")
        print(f"  Low task: {pair.low.description}")

        # --- Execute Medium-Level Task ---
        pair.medium.status = TaskStatus.in_progress
        actionable_elements = self.medium_agent.extract_actionable_elements()
        self.medium_agent.interpret_task(pair.medium, pair.low, actionable_elements)
        pair.medium.status = TaskStatus.completed

        # Propagate details from medium to low task.
        pair.low.x_coordinate = pair.medium.x_coordinate
        pair.low.y_coordinate = pair.medium.y_coordinate
        if pair.low.text_to_enter is None:
            pair.low.text_to_enter = pair.medium.text_to_enter

        # --- Execute Low-Level Task ---
        pair.low.status = TaskStatus.in_progress
        instructions = [{
            "action": pair.low.action,
            "x_coordinate": pair.low.x_coordinate,
            "y_coordinate": pair.low.y_coordinate,
            "text_to_enter": pair.low.text_to_enter,
            "key": pair.low.key,
            "end_x": pair.low.end_x,
            "end_y": pair.low.end_y
        }]
        result = self.low_agent.execute_instructions(instructions)

        if result.get("success"):
            time.sleep(1)
            screenshot_path = f"screenshots/{pair.low.id}.png"
            self.medium_agent.driver.save_screenshot(screenshot_path)
            print(f"DEBUG: Screenshot saved to {screenshot_path}")

            # Calculate relative coordinates for the overlay.
            win_pos = self.medium_agent.driver.get_window_position()
            title_bar_offset = self.medium_agent.driver.execute_script(
                "return window.outerHeight - window.innerHeight;"
            )
            viewport_origin_x = win_pos["x"]
            viewport_origin_y = win_pos["y"] + title_bar_offset
            relative_x = pair.low.x_coordinate - viewport_origin_x
            relative_y = pair.low.y_coordinate - viewport_origin_y

            # Overlay a red dot indicating the cursor position.
            LowLevelAgent.overlay_cursor_on_screenshot(screenshot_path, relative_x, relative_y)

            eval_message, action_success, goal_complete, rec_pair = self.low_agent.evaluate(
                pair.low, screenshot_path, self.user_goal
            )
            print(f"DEBUG: Vision response: {eval_message}")

            if action_success:
                pair.low.status = TaskStatus.completed
                pair.low.result = [result]
                print(f"DEBUG: Completed LOW task {pair.low.id}.")
            else:
                pair.low.status = TaskStatus.error
                pair.low.error_message = "Vision evaluation indicated failure."
                print(f"DEBUG: LOW task {pair.low.id} did not pass vision check. Recommendations: {rec_pair}")
                # Replan and execute the new subtask pair.
                new_pair = self.replan_pair(pair, rec_pair)
                print("DEBUG: Re-executing with replanned subtask pair.")
                self.execute_pair(new_pair)

            # Exit execution if overall goal is met.
            if goal_complete:
                print("DEBUG: Overall user goal accomplished. Exiting further execution.")
                exit(0)
        else:
            pair.low.status = TaskStatus.error
            pair.low.error_message = result.get("error_message")
            print(f"DEBUG: LOW task {pair.low.id} error: {pair.low.error_message}")

    def run(self) -> None:
        """
        Iterates over all subtask pairs in the plan and executes them sequentially.
        """
        print(f"DEBUG: Starting plan: {self.plan.task}")
        for pair in self.plan.pairs:
            self.execute_pair(pair)
        print("DEBUG: Finished executing plan.")
        print(self.plan.model_dump_json(indent=2))


# -----------------------------------------------------------------------------
# (6) Main Execution
# -----------------------------------------------------------------------------
def determine_starting_url(user_goal: str) -> str:
    """
    Uses the LLM to determine the best starting URL based on the user's goal.
    """
    prompt = f"""
        Given the following user goal:
        "{user_goal}"

        Select the most appropriate starting URL for this goal. 
        For example:
        - For search-related tasks, return "https://www.google.com".
        - For shopping tasks, return "https://www.amazon.ca".
        - For travel tasks, return "https://www.expedia.com".

        Return ONLY the URL.
        """
    try:
        response = openai.chat.completions.create(
            model="GPT4o",  # or your chosen model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that returns a URL based on a user goal."
                },
                {"role": "user", "content": prompt}
            ]
        )
        print("DEBUG: URL determination raw response:", response)
        url = response.choices[0].message.content.strip()
        if not url.startswith("http"):
            raise ValueError("Returned URL does not look valid.")
        print(f"DEBUG: Starting URL determined as: {url}")
        return url
    except Exception as e:
        print(f"DEBUG: Failed to determine URL, falling back to Google. Error: {e}")
        return "https://www.google.com"


if __name__ == "__main__":
    # Example user goal.
    user_goal = "shop for a fly fishing rod under 100$"
    starting_url = determine_starting_url(user_goal)

    # Set up Chrome driver options.
    from selenium.webdriver.chrome.options import Options
    chrome_options = Options()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--no-first-run")
    chrome_options.add_argument("--no-default-browser-check")
    chrome_options.add_argument("--disable-component-update")
    chrome_options.add_argument("--disable-background-networking")
    chrome_options.add_argument("--disable-sync")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "disable-infobars"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    # Initialize the Chrome driver.
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(starting_url)
    time.sleep(2)

    # Instantiate agents.
    high_agent = HighLevelAgent()
    medium_agent = MediumLevelAgent(driver)
    low_agent = LowLevelAgent(driver)

    # Generate the task breakdown plan.
    plan = high_agent.generate_task_breakdown(user_goal)
    print("\nDEBUG: Initial plan (pretty JSON):")
    print(plan.model_dump_json(indent=2))

    # Run the orchestrator.
    orchestrator = Orchestrator(plan, medium_agent, low_agent, user_goal)
    orchestrator.run()

    print("Final Plan State (pretty JSON):")
    print(plan.model_dump_json(indent=2))
    driver.quit()
