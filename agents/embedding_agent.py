import json
import os
from typing import Dict, Any
from autogen import AssistantAgent, UserProxyAgent
from sentence_transformers import SentenceTransformer


class EmbeddingAgent(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="EmbeddingGenerator",
            llm_config={
                "config_list": [{"model": "sentence-transformer", "api_key": "none"}],
                "temperature": 0,
            }
        )
        # Load the SentenceTransformer model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight, popular model
        except Exception as e:
            raise Exception(f"Failed to load SentenceTransformer model: {str(e)}")

    def generate_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generates an embedding for the given text using a SentenceTransformer model.
        Returns a JSON-compatible dictionary with the embedding or an error message.
        """
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: Text must be a non-empty string"}

        try:
            # Generate embedding using SentenceTransformer
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            return {"result": embedding}
        except Exception as e:
            return {"error": f"Failed to generate embedding: {str(e)}"}

    def generate_reply(self, messages, sender, config=None):
        try:
            message = json.loads(messages[-1]["content"])
            text = message.get("text")
            if not text:
                return json.dumps({"error": "No text provided"})
            result = self.generate_embedding(text)
            return json.dumps(result)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid message format: {str(e)}"})


# Example usage (for testing purposes)
if __name__ == "__main__":
    embedding_agent = EmbeddingAgent()
    # Example call to test the embedding generation
    result = embedding_agent.generate_embedding(
        text="This is a sample sentence for generating an embedding."
    )
    print(json.dumps(result, indent=2))