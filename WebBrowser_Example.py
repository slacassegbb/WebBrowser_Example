#!/usr/bin/env python3
"""
Web Browser Multi-Agent System

This implementation demonstrates:
- HighLevelAgent: Orchestrates planning, execution, and evaluation.
- MediumLevelAgent: Translates planning tasks into concrete UI actions.
- LowLevelAgent: Executes UI actions and evaluates outcomes using before and after screenshots.
A shared memory (a list of MemoryEvent objects) is used to persist events.
Python: 3.12.4
"""

from __future__ import annotations
import json
import time
import base64
import os
from typing import List, Optional, Dict, Any
from enum import Enum

import openai
from pydantic import BaseModel
from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.common.by import By
import pyautogui

# -------------------------
# Environment Configuration
# -------------------------
load_dotenv()
openai.api_type = "azure"
openai.azure_endpoint = os.getenv("GPT4_AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("GPT4_AZURE_OPENAI_KEY")
openai.api_version = os.getenv("GPT4_AZURE_OPENAI_API_VERSION")

# -------------------------
# Data Models and Enums
# -------------------------
class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    error = "error"

class PlanningTask(BaseModel):
    """
    A high-level planning task representing an abstract step toward the user goal.
    """
    id: str
    description: str
    agent_type: str  # e.g., "planner" or "user"
    status: TaskStatus = TaskStatus.pending
    result: Optional[List[dict]] = None
    error_message: Optional[str] = None

class UIAction(BaseModel):
    """
    A concrete UI action generated from a planning task.
    """
    action: str
    x_coordinate: int
    y_coordinate: int
    text_to_enter: Optional[str] = None
    key: Optional[str] = None
    end_x: Optional[int] = None
    end_y: Optional[int] = None

class Recommendation(BaseModel):
    """
    The evaluation result after executing a UI action.
    """
    success: bool
    goal_complete: bool
    observation: str
    new_task: Optional[PlanningTask] = None

# -------------------------
# Memory Event Structure
# -------------------------
class MemoryEvent(BaseModel):
    timestamp: float
    agent: str
    event: str
    task_id: Optional[str] = None
    ui_action: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    path: Optional[str] = None

# -------------------------
# HighLevelAgent
# -------------------------
class HighLevelAgent:
    """
    Orchestrates the overall workflow:
    - Generates planning tasks.
    - Uses MediumLevelAgent to convert tasks to UI actions.
    - Invokes LowLevelAgent to execute actions and evaluate outcomes.
    If MediumLevelAgent returns an action with 'change_subtask', a new planning task is generated.
    """
    INITIAL_PROMPT_TEMPLATE = (
        "User goal: '{user_goal}'.\n"
        "Based on the current webpage screenshot and context, generate the initial planning task to progress toward the goal. "
        "Return a JSON object matching the PlanningTask model."
    )
    NEXT_PROMPT_TEMPLATE = (
        "User goal: '{user_goal}'.\n"
        "Previous observation: '{observation}'.\n"
        "Memory (events): {memory}.\n"
        "Generate the next planning task to overcome any issues and progress toward the goal. "
        "Return a JSON object matching the PlanningTask model."
    )

    def __init__(self, medium_agent: MediumLevelAgent, low_agent: LowLevelAgent, user_goal: str, memory: List[MemoryEvent]):
        self.medium_agent = medium_agent
        self.low_agent = low_agent
        self.user_goal = user_goal
        self.memory = memory

    def generate_task(self, prompt_template: str, extra_info: Optional[str] = None) -> PlanningTask:
        """
        Generate a planning task via an LLM call.
        """
        memory_str = json.dumps([e.dict() for e in self.memory])
        prompt = prompt_template.format(
            user_goal=self.user_goal,
            observation=extra_info or "",
            memory=memory_str
        )
        messages = [
            {"role": "system", "content": "You are a planner agent that creates a structured plan to achieve the user’s goal by determining tasks and their order. It assigns tasks to the right agents to ensure effective goal completion"},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = openai.beta.chat.completions.parse(
                model="GPT4o",
                messages=messages,
                response_format=PlanningTask
            )
            task: PlanningTask = completion.choices[0].message.parsed
            self.memory.append(MemoryEvent(
                timestamp=time.time(),
                agent="HighLevelAgent",
                event="Generated planning task",
                task_id=task.id,
                ui_action=task.dict()
            ))
            print("HighLevelAgent: Generated task:", task)
            return task
        except Exception as e:
            print("HighLevelAgent: Error generating task:", e)
            self.memory.append(MemoryEvent(
                timestamp=time.time(),
                agent="HighLevelAgent",
                event="Task generation error",
                error=str(e)
            ))
            return PlanningTask(
                id="fallback",
                description="Perform a default UI action to progress toward the goal.",
                agent_type="planner"
            )

    def run(self) -> None:
        """
        Main orchestration loop.
        """
        # Capture an initial screenshot for context.
        init_screenshot = "screenshots/initial.png"
        self.low_agent.capture_screenshot(init_screenshot)

        # Generate the initial planning task.
        current_task = self.generate_task(self.INITIAL_PROMPT_TEMPLATE)

        while True:
            print(f"HighLevelAgent: Executing task: {current_task.description}")
            current_task.status = TaskStatus.in_progress

            # Capture a "before" screenshot.
            pre_screenshot_path = f"screenshots/{current_task.id}_before.png"
            self.low_agent.capture_screenshot(pre_screenshot_path)

            # Translate the planning task to a concrete UI action.
            ui_action = self.medium_agent.run(current_task)
            if ui_action is None or ui_action.action == "change_subtask":
                error_info = "No appropriate UI element found; changing subtask."
                self.memory.append(MemoryEvent(
                    timestamp=time.time(),
                    agent="HighLevelAgent",
                    event="Translation failure/change_subtask",
                    task_id=current_task.id,
                    error=error_info
                ))
                current_task = self.generate_task(self.NEXT_PROMPT_TEMPLATE, error_info)
                continue

            # Execute the UI action.
            execution_result = self.low_agent.run(ui_action)
            if not execution_result.get("success"):
                error_info = execution_result.get("error_message", "Unknown execution error.")
                self.memory.append(MemoryEvent(
                    timestamp=time.time(),
                    agent="HighLevelAgent",
                    event="Execution error",
                    task_id=current_task.id,
                    error=error_info
                ))
                current_task = self.generate_task(self.NEXT_PROMPT_TEMPLATE, error_info)
                continue

            # Capture an "after" screenshot.
            post_screenshot_path = f"screenshots/{current_task.id}_after.png"
            self.low_agent.capture_screenshot(post_screenshot_path)

            # Evaluate the execution using both screenshots.
            recommendation = self.low_agent.evaluate_execution(current_task, pre_screenshot_path, post_screenshot_path, self.user_goal)
            print("HighLevelAgent Observation:", recommendation.observation)
            self.memory.append(MemoryEvent(
                timestamp=time.time(),
                agent="HighLevelAgent",
                event="Evaluation completed",
                task_id=current_task.id,
                observation=recommendation.observation
            ))

            # Check if the goal is complete.
            if recommendation.goal_complete:
                self.memory.append(MemoryEvent(
                    timestamp=time.time(),
                    agent="HighLevelAgent",
                    event="Goal achieved",
                    task_id=current_task.id
                ))
                print("HighLevelAgent: Goal achieved!")
                break

            # Use any new planning task from the recommendation; otherwise, generate a new task.
            if recommendation.new_task:
                current_task = recommendation.new_task
            else:
                current_task = self.generate_task(self.NEXT_PROMPT_TEMPLATE, recommendation.observation)

        print("HighLevelAgent: Execution complete.")
# -------------------------
# MediumLevelAgent
# -------------------------
class MediumLevelAgent:
    """
    Translates a high-level planning task into a concrete UI action by examining the current UI.
    The prompt instructs the LLM to select an appropriate UI element from the provided list.
    If no suitable element is found, the LLM is instructed to return an action with type 'change_subtask'.
    """
    SYSTEM_PROMPT = (
        "You are a Web Browser agent. Given a planning task and a list of actionable UI elements, "
        "determine the best UI action to perform. Valid actions include 'enter_text', 'press_enter', 'click', "
        "'double_click', 'undo_text', 'select_text', 'copy', 'paste', 'right_click', 'press_key', 'scroll', and 'move_mouse'.\n"
        "If the appropriate UI element is not found in the provided list, return an action with 'change_subtask' "
        "and include a message explaining the situation. Return ONLY a JSON object matching the UIAction model."
    )

    def __init__(self, driver: webdriver.Chrome, memory: List[MemoryEvent]):
        self.driver = driver
        self.memory = memory

    def extract_ui_elements(self) -> List[dict]:
        """
        Extract actionable UI elements from the current webpage.
        This version includes various XPath selectors to capture clickable elements.
        """
        xpath_query = (
            "//input | //button | //a | //textarea | "  # traditional interactive elements
            "//*[@role='button'] | "                        # elements explicitly marked as buttons
            "//*[contains(@class, 'btn')] | "               # elements with class names suggesting a button
            "//*[ (self::div or self::span or self::i) and @onclick ] | "  # clickable non-standard elements
            "//*[ (self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::p) "  # headers and paragraphs that might be clickable
            "and contains(@style, 'cursor: pointer') "      # with a pointer cursor style
            "and string-length(normalize-space(text())) > 0 ]"
        )
        elements = self.driver.find_elements(By.XPATH, xpath_query)
        actionable = []
        win_pos = self.driver.get_window_position()
        title_bar_offset = self.driver.execute_script("return window.outerHeight - window.innerHeight;")
        
        seen = set()  # to avoid duplicates
        for el in elements:
            if el.is_displayed():
                rect = self.driver.execute_script(
                    """
                    let rect = arguments[0].getBoundingClientRect();
                    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
                    """,
                    el
                )
                # Skip elements that are essentially invisible
                if rect["width"] == 0 or rect["height"] == 0:
                    continue

                abs_x = int(rect["left"] + win_pos["x"])
                abs_y = int(rect["top"] + win_pos["y"] + title_bar_offset)
                center_x = abs_x + int(rect["width"] / 2)
                center_y = abs_y + int(rect["height"] / 2)
                
                key = (center_x, center_y, rect["width"], rect["height"])
                if key in seen:
                    continue
                seen.add(key)
                
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
                    "height": int(rect["height"]),
                    "text": el.text.strip()
                })
        return actionable

    def translate_task_to_action(self, task: PlanningTask) -> Optional[UIAction]:
        """
        Convert a planning task into a concrete UI action via an LLM call.
        """
        ui_elements = self.extract_ui_elements()
        if not ui_elements:
            print("MediumLevelAgent: No UI elements found.")
            return None

        prompt = (
            f"Task: {task.description}\n"
            f"UI Elements: {json.dumps(ui_elements, indent=2)}\n"
            f"Memory (events): {json.dumps([e.dict() for e in self.memory])}\n\n"
            "Determine the best UI action to execute this task. "
            "If no appropriate element is available, return an action with 'change_subtask' and an explanation. "
            "Return ONLY a JSON object matching the UIAction model with fields: "
            "action, x_coordinate, y_coordinate, text_to_enter, key, end_x, end_y."
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = openai.beta.chat.completions.parse(
                model="GPT4o",
                messages=messages,
                response_format=UIAction
            )
            action: UIAction = completion.choices[0].message.parsed
            self.memory.append(MemoryEvent(
                timestamp=time.time(),
                agent="MediumLevelAgent",
                event="Translated task to UIAction",
                task_id=task.id,
                ui_action=action.dict()
            ))
            print(f"MediumLevelAgent: Generated UIAction: {action}")
            return action
        except Exception as e:
            print("MediumLevelAgent Error:", e)
            self.memory.append(MemoryEvent(
                timestamp=time.time(),
                agent="MediumLevelAgent",
                event="Translation error",
                task_id=task.id,
                error=str(e)
            ))
            return None

    def run(self, task: PlanningTask) -> Optional[UIAction]:
        """
        Entry point: translate a given planning task into a UI action.
        """
        return self.translate_task_to_action(task)

# -------------------------
# LowLevelAgent
# -------------------------
class LowLevelAgent:
    """
    Executes concrete UI actions and then evaluates the outcome.
    Executes UI actions and captures before and after screenshots for evaluation.
    """
    SYSTEM_PROMPT_CONTROLLER = (
        "You are the Low-Level Web Controller. Your responsibilities are twofold: "
        "first, to ensure that the UI action is executed precisely; and second, to analyze the resulting UI state. "
        "After performing an action, review the provided before and after screenshots to determine whether the intended outcome has been reached. "
        "If the outcome is not achieved, suggest a generic corrective task to overcome the issue, avoiding repetition of previous failures. "
        "Return a JSON object matching the Recommendation model."
    )

    def __init__(self, driver: webdriver.Chrome, memory: List[MemoryEvent]):
        self.driver = driver
        self.memory = memory

    def execute(self, action: UIAction) -> dict:
        """
        Execute the UI action using PyAutoGUI.
        """
        print("LowLevelAgent: Executing action:", action)
        try:
            if action.action == "move_mouse":
                pyautogui.moveTo(action.x_coordinate, action.y_coordinate)
                pyautogui.click()
            elif action.action in ["click", "single_click"]:
                pyautogui.moveTo(action.x_coordinate, action.y_coordinate)
                pyautogui.click()
            elif action.action == "double_click":
                pyautogui.moveTo(action.x_coordinate, action.y_coordinate)
                pyautogui.doubleClick()
            elif action.action == "enter_text":
                pyautogui.click(action.x_coordinate, action.y_coordinate)
                time.sleep(0.5)
                for _ in range(50):
                    pyautogui.press('backspace')
                    time.sleep(0.02)
                time.sleep(0.5)
                if action.text_to_enter:
                    pyautogui.typewrite(action.text_to_enter)
                pyautogui.press('enter')
            elif action.action == "press_enter":
                pyautogui.press('enter')
            elif action.action == "undo_text":
                pyautogui.click(action.x_coordinate, action.y_coordinate)
                time.sleep(0.5)
                for _ in range(50):
                    pyautogui.press('backspace')
                    time.sleep(0.02)
            elif action.action == "select_text":
                if None in (action.x_coordinate, action.y_coordinate, action.end_x, action.end_y):
                    raise Exception("Missing coordinates for select_text")
                pyautogui.moveTo(action.x_coordinate, action.y_coordinate)
                pyautogui.dragTo(action.end_x, action.end_y, duration=0.2, button='left')
            elif action.action == "copy":
                pyautogui.hotkey('ctrl', 'c')
            elif action.action == "paste":
                pyautogui.hotkey('ctrl', 'v')
            elif action.action == "right_click":
                pyautogui.moveTo(action.x_coordinate, action.y_coordinate)
                pyautogui.click(button='right')
            elif action.action == "press_key":
                if action.key is None:
                    raise Exception("Missing key for press_key")
                pyautogui.press(action.key)
            elif action.action == "scroll":
                if action.text_to_enter and action.text_to_enter.lower() == "up":
                    pyautogui.scroll(300)
                else:
                    pyautogui.scroll(-300)
            else:
                print("LowLevelAgent: Unknown action:", action.action)
                return {"success": False, "error_message": f"Unknown action: {action.action}"}
        except Exception as e:
            print("LowLevelAgent: Action execution failed:", str(e))
            return {"success": False, "error_message": str(e)}
        self.memory.append(MemoryEvent(
            timestamp=time.time(),
            agent="LowLevelAgent",
            event="Executed UIAction",
            ui_action=action.dict()
        ))
        return {"success": True, "error_message": None}

    def capture_screenshot(self, path: str) -> None:
        """
        Capture and save a screenshot.
        """
        self.driver.save_screenshot(path)
        print(f"LowLevelAgent: Screenshot saved at {path}")
        self.memory.append(MemoryEvent(
            timestamp=time.time(),
            agent="LowLevelAgent",
            event="Captured screenshot",
            path=path
        ))

    def evaluate_execution(self, task: PlanningTask, pre_screenshot_path: str, post_screenshot_path: str, user_goal: str) -> Recommendation:
        """
        Evaluate the outcome of an executed action using both 'before' and 'after' screenshots.
        """
        with open(pre_screenshot_path, "rb") as pre_file:
            pre_base64 = base64.b64encode(pre_file.read()).decode("utf-8")
        with open(post_screenshot_path, "rb") as post_file:
            post_base64 = base64.b64encode(post_file.read()).decode("utf-8")
        prompt = (
            f"Evaluate the execution of the task '{task.description}'.\n"
            f"User goal: '{user_goal}'.\n"
            "Compare the before and after screenshots to determine whether the intended outcome has been achieved. "
            "If not, suggest a generic corrective next step. "
            "Return a JSON object matching the Recommendation model."
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT_CONTROLLER},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pre_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{post_base64}"}}
            ]}
        ]
        try:
            completion = openai.beta.chat.completions.parse(
                model="GPT4o",
                messages=messages,
                response_format=Recommendation
            )
            recommendation: Recommendation = completion.choices[0].message.parsed
            print("LowLevelAgent: Evaluation result:", recommendation)
            self.memory.append(MemoryEvent(
                timestamp=time.time(),
                agent="LowLevelAgent",
                event="Evaluated execution",
                task_id=task.id,
                observation=recommendation.observation
            ))
            return recommendation
        except Exception as e:
            print("LowLevelAgent: Evaluation error:", e)
            self.memory.append(MemoryEvent(
                timestamp=time.time(),
                agent="LowLevelAgent",
                event="Evaluation error",
                task_id=task.id,
                error=str(e)
            ))
            return Recommendation(
                success=False,
                goal_complete=False,
                observation="Evaluation failed.",
                new_task=None
            )

    def run(self, action: UIAction) -> dict:
        """
        Execute the given UI action.
        """
        return self.execute(action)

# -------------------------
# URL Determination Helper
# -------------------------
def determine_starting_url(user_goal: str) -> str:
    prompt = (
        f"Given the user goal: '{user_goal}', select the most appropriate starting URL. "
        "For search tasks return 'https://www.google.com', for shopping return 'https://www.amazon.com', "
        "for travel return 'https://www.expedia.com'. Return ONLY the URL."
    )
    try:
        response = openai.chat.completions.create(
            model="GPT4o",
            messages=[
                {"role": "system", "content": "You are an assistant that selects a starting URL based on the user goal."},
                {"role": "user", "content": prompt}
            ]
        )
        url = response.choices[0].message.content.strip()
        if not url.startswith("http"):
            raise ValueError("Invalid URL returned.")
        print(f"Determined starting URL: {url}")
        return url
    except Exception as e:
        print("URL determination error:", e)
        return "https://www.google.com"

# -------------------------
# Main Execution
# -------------------------
def main():
    # Example user goals:
    # user_goal = "Find a fly fishing rod under 100$ and add it to the cart"
    # user_goal = "Search for art museum on the Map"
    # user_goal = "Find a github repository related to multi-agents"
    user_goal = "Find flights from montreal to vancouver"
    starting_url = determine_starting_url(user_goal)
    
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
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "disable-infobars"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(starting_url)
    time.sleep(2)
    os.makedirs("screenshots", exist_ok=True)
    
    # Create shared memory as a simple list of MemoryEvent objects.
    shared_memory: List[MemoryEvent] = []
    
    medium_agent = MediumLevelAgent(driver, shared_memory)
    low_agent = LowLevelAgent(driver, shared_memory)
    high_agent = HighLevelAgent(medium_agent, low_agent, user_goal, shared_memory)
    
    high_agent.run()
    driver.quit()

if __name__ == "__main__":
    main()



