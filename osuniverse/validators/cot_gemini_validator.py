import base64
import io
import os
from typing import List

import google.generativeai as genai
import json_repair
from PIL import Image
from rich.console import Console

from ..data.testcase import (
    Check,
    CommandOutputCheck,
    ExpectedFlowCheck,
    FinalScreenshotCheck,
    ReturnedResultCheck,
)
from ..data.testcaserun import Step, TestCaseRun
from .base import BaseValidator, CheckResult

console = Console()


class COTGeminiValidator(BaseValidator):
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")  # type: ignore

        # NOTE: if you want to use `gemini-2.5-pro-exp-03-25` don't forget that:
        # 1) It doesn't support `presence_penalty` and `frequency_penalty` (remove from the request params below)
        # 2) It has rate limits and works poorly with multi-process test runs
        # self.model = genai.GenerativeModel(model_name="gemini-2.5-pro-preview-03-25")  # type: ignore
        # console.print("Waiting for Gemini API to be ready...")
        # time.sleep(10)

        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=key)  # type: ignore
        self.max_image_size = (
            768,
            768,
        )  # Maximum width/height for resized images; Gemini resizes to this size anyway

    def _resize_base64_image(self, base64_string: str) -> str:
        """Resize a base64 image to reduce its size while maintaining aspect ratio"""
        try:
            # Decode base64 to binary
            img_data = base64.b64decode(base64_string)

            # Open image with Pillow
            img = Image.open(io.BytesIO(img_data))

            # Calculate new size maintaining aspect ratio
            img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

            # Save resized image to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or "PNG", quality=85, optimize=True)

            # Convert back to base64
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            console.print(f"Warning: Failed to resize image: {e}", style="yellow")
            return base64_string

    def _generate_content_part_from_step(self, step: Step) -> dict[str, str] | None:
        if step.screenshot and step.screenshot.startswith("data:image"):
            # Extract base64 part after the comma
            base64_part = step.screenshot.split(",", 1)[1]
            resized_base64 = self._resize_base64_image(base64_part)
            image_url = f"data:image/png;base64,{resized_base64}"
        else:
            image_url = step.screenshot if step else None
        if image_url and image_url.startswith("data:image"):
            mime_type = image_url.split(";")[0].split(":")[1]
            base64_data = image_url.split(",")[1]
            return {"mime_type": mime_type, "data": base64_data}
        return None

    def validate_check(self, check: Check, testcaserun: TestCaseRun) -> CheckResult:
        system_prompt = ""
        prompt = ""
        content_parts = []

        if isinstance(check, ReturnedResultCheck):
            system_prompt = self._returned_result_system_prompt()
            prompt = f"Agent Task: {testcaserun.task}\n"
            prompt += f"Agent Returned Result: {testcaserun.result.action if testcaserun.result else ''}\n"
            prompt += f"Expected Returned Result: {check.returned_result if isinstance(check, ReturnedResultCheck) else ''}\n"  # type: ignore
            prompt += "The final screenshot is attached below."
            content_parts: List[str | dict[str, str]] = [system_prompt, prompt]
            step = testcaserun.result
            if step:
                content_part = self._generate_content_part_from_step(step)
                if content_part:
                    content_parts.append(content_part)
        elif isinstance(check, FinalScreenshotCheck):
            system_prompt = self._final_screenshot_system_prompt()
            prompt = f"Agent Task: {testcaserun.task}\n"
            prompt += f"Expected Final Screenshot Description: {check.final_screenshot if isinstance(check, FinalScreenshotCheck) else ''}\n"  # type: ignore
            prompt += "The final screenshot is attached below."
            content_parts: List[str | dict[str, str]] = [system_prompt, prompt]
            step = testcaserun.result
            if step:
                content_part = self._generate_content_part_from_step(step)
                if content_part:
                    content_parts.append(content_part)
        elif isinstance(check, ExpectedFlowCheck):
            system_prompt = self._expected_flow_system_prompt()
            prompt = f"Agent Task: {testcaserun.task}\n"
            prompt += f"Expected Flow: {check.expected_flow if isinstance(check, ExpectedFlowCheck) else ''}\n"  # type: ignore
            content_parts = [system_prompt, prompt]
            for index, step in enumerate(testcaserun.trajectory):
                step_description = f"Step: {index}. Timestamp: {step.timestamp}. Agent action: {step.action}"
                content_parts.append(step_description)
                if step.screenshot:
                    content_part = self._generate_content_part_from_step(step)
                    if content_part:
                        content_parts.append(content_part)
        elif isinstance(check, CommandOutputCheck):
            system_prompt = self._expected_command_output_system_prompt()
            expected_output = check.command_output  # type: ignore
            command_outputs = [
                result.output
                for result in testcaserun.command_output_check_results
                if result.command == check.command  # type: ignore
            ]
            if len(command_outputs) == 0:
                return CheckResult(
                    check=check,
                    result="Command output not found",
                    score=0,
                    validation_input_tokens=0,
                    validation_output_tokens=0,
                )
            prompt = f"Agent Task: {testcaserun.task}\n"
            prompt += f"Agent Command: {check.command}\n"  # type: ignore
            prompt += f"Expected Command Output: {expected_output}\n"
            prompt += f"Command Output: {command_outputs[0]}\n"
            content_parts = [system_prompt, prompt]
        else:
            console.print(
                f"Warning: Unsupported check type: {type(check)}", style="yellow"
            )
            return CheckResult(
                check=check,
                result="Unsupported check type: " + str(type(check)),
                score=0,
                validation_input_tokens=0,
                validation_output_tokens=0,
            )

        try:
            # Make request to Gemini
            response = self.model.generate_content(  # type: ignore
                content_parts,  # type: ignore
                generation_config=genai.GenerationConfig(  # type: ignore
                    temperature=0.0,
                    top_p=0.95,
                    top_k=20,
                    presence_penalty=0.2,
                    frequency_penalty=0.2,
                ),
            )

            # Parse response
            result = json_repair.loads(response.text)

            metadata = response.usage_metadata
            validation_input_tokens = metadata.prompt_token_count
            validation_output_tokens = metadata.candidates_token_count

            return CheckResult(
                check=check,
                result=result["comment"],  # type: ignore
                score=result["score"],  # type: ignore
                validation_input_tokens=validation_input_tokens,
                validation_output_tokens=validation_output_tokens,
            )
        except Exception as e:
            console.print(
                f"Warning: Failed to validate returned result: {e}", style="yellow"
            )
            return CheckResult(
                check=check,
                result=f"Error: {e}",
                score=0,
                validation_input_tokens=0,
                validation_output_tokens=0,
            )

    def _returned_result_system_prompt(self) -> str:
        prompt = """You are a Gemini Validator for a GUI-navigation test case. Your task is to evaluate a test run provided by a GUI agent. Analyze the agent's task description, the returned result, the associated final screenshot, and the expected outcome reference. Your evaluation must strictly rely on the observable data presented, without inferring details beyond what is provided. Then, output a numeric score (score: 0 or 1) along with a detailed comment.

INSTRUCTIONS:
- Respond with a valid JSON object containing exactly two keys: "score" (an integer 0 or 1) and "comment" (a descriptive string).
- Evaluate only the elements visible in the provided text and screenshot. Do not infer any information that is not explicitly shown.
- Scrutinize details such as file names, dates, numerical values, and other relevant specifics.
- If the expected result appears clearly—even in an alternative textual representation—assign a score of 1 provided all critical details are met.
- Use precise language and directly reference the observable evidence in your commentary.
- In your evaluation, include a clear checklist of questions to ask yourself (e.g., "Did the agent choose the correct dates?", "Is the file name correct?", "Are numerical values consistent?") and explicitly answer each question as part of your comment.
- Return only valid JSON, for example:
{"score": 1, "comment": "The agent's output matches the expected result with all details accurately represented. Checklist: [Correct dates: Yes, File name: Yes, Numerical values: Yes]"}
- Do not output any additional text or commentary."""
        return prompt

    def _final_screenshot_system_prompt(self) -> str:
        prompt = """You are a Gemini Validator for GUI-navigation test runs. Your role is to evaluate the final screenshot against the provided expected state description. Analyze the task and confirm that the screenshot's visual output aligns with the stated requirements. Your evaluation must be based solely on the details that can be observed from the screenshot.

INSTRUCTIONS:
- Respond with a valid JSON object containing exactly two keys: "score" (an integer 0 or 1) and "comment" (a descriptive explanation).
- Clearly define the expected visual criteria (e.g., specific UI elements, correct dates, accurate file names) if applicable.
- Verify that every visual criterion is satisfied; if any expected element is missing or incorrect, assign a score of 0.
- Base your evaluation strictly on the observable evidence from the screenshot.
- In your evaluation, include a clear checklist of questions (e.g., "Are all required UI elements present?", "Are the dates correct?", "Is the file name accurate?") and provide explicit answers for each in your comment.
- Return only valid JSON, for example:
{"score": 1, "comment": "Criteria verified: all required UI elements, dates, and file names match the expected state. Checklist: [UI elements: Yes, Dates: Yes, File name: Yes]"}
- Do not include any extraneous text."""
        return prompt

    def _expected_flow_system_prompt(self) -> str:
        prompt = """You are a Gemini Validator tasked with evaluating the complete execution flow of a GUI-navigation test run. Your goal is to compare the agent's full action trajectory—including both textual descriptions and associated screenshots—with the expected workflow. Ensure your analysis is strictly based on the provided data.

INSTRUCTIONS:
- Respond with a valid JSON object containing exactly two keys: "score" (an integer 0 or 1) and "comment" (a detailed explanation).
- Begin by outlining the key criteria derived from the expected workflow (for example, file creation, content verification, and proper file saving).
- Evaluate each step based on the evidence in the corresponding screenshots and text details.
- Assign a score of 1 only if every defined criterion is completely met; note that extra steps (for example, encountering an error but successfully recovering) should be acknowledged separately and do not affect the score as long as all required parts of the flow are completed.
- Pay attention to all observable details, such as file names, timestamps, and UI appearances.
- In your evaluation, include a clear checklist of questions you must answer (e.g., "Was the file created?", "Does the file contain the requested content?", "Was the file saved in the correct directory?", "Were any extra steps performed, and if so, did they impact the core flow?").
- Return only valid JSON, for example:
{"score": 1, "comment": "Criteria met: file was created correctly with accurate content and saved in the proper directory. Extra steps were noted but did not compromise the required workflow. Checklist: [File created: Yes, Content correct: Yes, Correct directory: Yes, Extra steps non-penalizing: Yes]"}
- Do not output any additional text or commentary."""
        return prompt

    def _expected_command_output_system_prompt(self) -> str:
        prompt = """You are a Gemini Validator tasked with evaluating the output of a command executed after a GUI-navigation agent finished its task. Your goal is to compare this command output with the expected output, considering the agent's task. Ensure your analysis is strictly based on the provided data.

INSTRUCTIONS:
- Respond with a valid JSON object containing exactly two keys: "score" (an integer 0 or 1) and "comment" (a detailed explanation).
- Begin by outlining the key criteria derived from the expected output (for example, file creation, content verification, and proper file saving).
- Evaluate the command output based on the evidence in text details.
- Assign a score of 1 only if every defined criterion is completely met.
- Pay attention to all observable details, such as file names, timestamps, and UI appearances.
- In your evaluation, include a clear checklist of questions you must answer (e.g., "Is the output formatted correctly?", "Is the output contains expected data?", "Is the output complete?").
- Return only valid JSON, for example:
{"score": 1, "comment": "Criteria met: the output is formatted correctly and contains expected data. Checklist: [Output formatted: Yes, Output contains expected data: Yes]"}
- Do not output any additional text or commentary."""
        return prompt
