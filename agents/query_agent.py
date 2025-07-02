import json
import os
from typing import Dict, Any, List
from autogen import AssistantAgent, UserProxyAgent
import chromadb


class ChromaDBQueryAgent(AssistantAgent):
    def __init__(self, collection_name: str = "text_embeddings"):
        super().__init__(
            name="ChromaDBQuerier",
            llm_config={
                "config_list": [{"model": "chromadb", "api_key": "none"}],
                "temperature": 0,
            }
        )
        # Disable ChromaDB telemetry to avoid errors
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

        # Connect to existing ChromaDB client and collection
        try:
            self.client = chromadb.PersistentClient(path="./../chroma_db")
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise Exception(f"Failed to connect to existing ChromaDB collection '{collection_name}': {str(e)}")

    def query_embedding(self, query_embedding: List[float], n_results: int = 5, distance_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Queries the existing ChromaDB collection with a precomputed embedding using cosine similarity.
        Filters results to include only those with distance <= distance_threshold.
        Returns a JSON-compatible dictionary with the top matching results or an error message.
        """
        if not query_embedding or not isinstance(query_embedding, list) or not all(isinstance(x, (int, float)) for x in query_embedding):
            return {"error": "Invalid input: Query embedding must be a non-empty list of numbers"}

        try:
            # Query the ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Filter results by distance threshold
            formatted_results = []
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                if distance <= distance_threshold:  # Only include similar results
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance  # Cosine distance (0 = identical, 1 = dissimilar)
                    })

            return {"result": formatted_results}
        except Exception as e:
            return {"error": f"Failed to query collection: {str(e)}"}


    def generate_reply(self, messages, sender, config=None):
        """
        Processes a JSON message with query_embedding, n_results, and distance_threshold, calls query_embedding, and returns a JSON string.
        """
        try:
            message = json.loads(messages[-1]["content"])
            query_embedding = message.get("query_embedding")
            n_results = message.get("n_results", 5)
            distance_threshold = message.get("distance_threshold", 0.3)  # Default threshold
            if not query_embedding:
                return json.dumps({"error": "No query embedding provided"})
            result = self.query_embedding(query_embedding, n_results, distance_threshold)
            return json.dumps(result)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid message format: {str(e)}"})


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Initialize agents
    chromadb_query_agent = ChromaDBQueryAgent()

    # Example embedding (mock data for testing; replace with actual embedding from EmbeddingAgent)
    sample_embedding = [0.1, 0.2, -0.3, 0.4] * 96  # Mock 384-dimensional embedding

    # Example call to query the collection
    query_result = chromadb_query_agent.query_embedding(
        query_embedding=sample_embedding,
        n_results=3
    )
    print("Query Result:")
    print(json.dumps(query_result, indent=2))