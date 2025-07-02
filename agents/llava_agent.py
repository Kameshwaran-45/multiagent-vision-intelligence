import base64
import json
import os
from typing import Dict, Any
import requests
from autogen import AssistantAgent, UserProxyAgent


class LLavaAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="LLaVADescriber",
            llm_config={
                "config_list": [{"model": "local-llava", "api_key": "none"}],
                "temperature": 0,
            }
        )

    def call_llava_tool(self, text: str, image_path: str) -> Dict[str, Any]:
        """
        Calls the LLaVA model via Ollama's API to analyze an image with a given prompt.
        Returns a JSON-compatible dictionary.
        """
        # Normalize and validate image path
        image_path = os.path.abspath(image_path)
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        try:
            # Convert the image to base64
            with open(image_path, "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

            # Prepare the payload for Ollama's /api/generate endpoint
            payload = {
                "model": "llava:latest",  # Ensure this matches the model name in your Ollama setup
                "prompt": text,
                "images": [image_b64]
            }

            # Send the request to Ollama server
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response (Ollama's /api/generate streams JSON lines)
            result = ""
            for line in response.text.splitlines():
                try:
                    json_line = json.loads(line)
                    if "response" in json_line:
                        result += json_line["response"]
                except json.JSONDecodeError:
                    continue

            return {"result": result.strip()} if result else {"error": "No valid response from model"}
        except FileNotFoundError:
            return {"error": f"Image file not found: {image_path}"}
        except requests.RequestException as e:
            return {"error": f"Failed to connect to Ollama server: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to analyze image: {str(e)}"}

    def generate_reply(self, messages, sender, config=None):
        """
        Processes a JSON message with prompt and image_path, calls call_llava_tool, and returns a JSON string.
        """
        try:
            message = json.loads(messages[-1]["content"])
            prompt = message.get("prompt", "Describe the contents of this image in detail.")
            image_path = message.get("image_path")
            if not image_path:
                return json.dumps({"error": "No image path provided"})
            result = self.call_llava_tool(prompt, image_path)
            return json.dumps(result)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid message format: {str(e)}"})


# Example usage (for testing purposes)
if __name__ == "__main__":
    llava_agent = LLavaAgent()
    # Example call to test the LLaVA tool
    result = llava_agent.call_llava_tool(
        text="Describe the contents of this image in detail.",
        image_path=r'C:\Users\ktb3kor\Downloads\Projects\multiagent-vision-intelligence\00000000.jpg'
    )
    print(json.dumps(result, indent=2))