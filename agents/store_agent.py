import json
import os
from typing import Dict, Any, List
from autogen import AssistantAgent, UserProxyAgent
import chromadb
import uuid
import warnings
warnings.filterwarnings("ignore", message=".*telemetry.*")

class ChromaDBAgent(AssistantAgent):
    def __init__(self, collection_name: str = "text_embeddings"):
        super().__init__(
            name="ChromaDBStorer",
            llm_config={
                "config_list": [{"model": "chromadb", "api_key": "none"}],
                "temperature": 0,
            }
        )
        # Disable ChromaDB telemetry
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

        # Initialize ChromaDB client with persistent storage
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB client: {str(e)}")

    def store_embedding(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Stores a precomputed embedding, associated text, and metadata in ChromaDB.
        Returns a JSON-compatible dictionary with the result or an error message.
        """
        if not text or not isinstance(text, str):
            return {"error": "Invalid input: Text must be a non-empty string"}
        if not embedding or not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
            return {"error": "Invalid input: Embedding must be a non-empty list of numbers"}

        try:
            # Generate a unique ID for the entry
            entry_id = str(uuid.uuid4())

            # Prepare metadata (default to empty dict if None)
            metadata = metadata or {}

            # Store the embedding, text, and metadata in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[entry_id]
            )

            return {
                "result": {
                    "id": entry_id,
                    "text": text,
                    "metadata": metadata,
                    "message": f"Embedding stored successfully with ID {entry_id}"
                }
            }
        except Exception as e:
            return {"error": f"Failed to store embedding: {str(e)}"}

    def generate_reply(self, messages, sender, config=None):
        try:
            message = json.loads(messages[-1]["content"])
            text = message.get("text")
            embedding = message.get("embedding")
            metadata = message.get("metadata", {})
            if not text or not embedding:
                return json.dumps({"error": "Missing text or embedding"})
            result = self.store_embedding(text, embedding, metadata)
            return json.dumps(result)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid message format: {str(e)}"})


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Initialize agents
    chromadb_agent = ChromaDBAgent

    # Example embedding (mock data for testing; replace with actual embedding from EmbeddingAgent)
    sample_embedding = [0.1, 0.2, -0.3, 0.4] * 96  # Mock 384-dimensional embedding

    # # Example call to store an embedding
    store_result = chromadb_agent.store_embedding(
        text="This is a sample sentence for storing an embedding.",
        embedding=sample_embedding,
        metadata={"source": "example", "category": "test"}
    )
    print("Store Result:")
    print(json.dumps(store_result, indent=2))