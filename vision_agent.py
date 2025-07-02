import json
import os
from typing import Dict, Any
from autogen import AssistantAgent, UserProxyAgent

from agents.llava_agent import LLavaAgent
from agents.embedding_agent import EmbeddingAgent
from agents.store_agent import ChromaDBAgent
from agents.query_agent import ChromaDBQueryAgent


def get_agents(image_path: str = None):
    """
    Initializes and returns the agents and user proxy agent.
    If image_path is provided, it is passed via messages to LLavaAgent.
    """
    llava_agent = LLavaAgent()
    embedding_agent = EmbeddingAgent()
    store_agent = ChromaDBAgent()
    query_agent = ChromaDBQueryAgent()
    user_agent = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        system_message="Based on the user prompt: If it contains 'describe' or 'image', ask LLavaAgent to describe the image at the provided path, then ask EmbeddingAgent to create embeddings for the description, and store them in ChromaDB using ChromaDBAgent. If the prompt contains 'search' or 'query', ask EmbeddingAgent to create an embedding for the query text and fetch results using ChromaDBQueryAgent.",
        code_execution_config=False,
    )
    return user_agent, llava_agent, embedding_agent, store_agent, query_agent


def run_workflow(image_path: str = None, prompt: str = "describe", distance_threshold: float = 0.3):
    """
    Runs the workflow based on the user prompt, using generate_reply for agent interactions.
    """
    user_agent, llava_agent, embedding_agent, store_agent, query_agent = get_agents(image_path)

    if "describe" in prompt.lower():
        if not image_path or not os.path.exists(image_path):
            return {"error": f"Invalid or missing image path: {image_path}"}

        # Step 1: Ask LLavaAgent to describe the image
        llava_result = llava_agent.generate_reply(
            messages=[{
                "content": json.dumps({
                    "prompt": "Describe the contents of this image in detail.",
                    "image_path": image_path
                })
            }],
            sender=user_agent
        )
        try:
            llava_data = json.loads(llava_result) if isinstance(llava_result, str) else llava_result
            if "error" in llava_data:
                return llava_data
            description = llava_data["result"]
        except json.JSONDecodeError:
            return {"error": f"Failed to parse LLavaAgent response: {llava_result}"}

        # Step 2: Ask EmbeddingAgent to generate embedding
        embed_result = embedding_agent.generate_reply(
            messages=[{
                "content": json.dumps({
                    "text": description
                })
            }],
            sender=user_agent
        )
        try:
            embed_data = json.loads(embed_result) if isinstance(embed_result, str) else embed_result
            if "error" in embed_data:
                return embed_data
            embedding = embed_data["result"]
        except json.JSONDecodeError:
            return {"error": f"Failed to parse EmbeddingAgent response: {embed_result}"}

        # Step 3: Ask ChromaDBAgent to store embedding with image_path in metadata
        store_result = store_agent.generate_reply(
            messages=[{
                "content": json.dumps({
                    "text": description,
                    "embedding": embedding,
                    "metadata": {"source": "image_description", "image_path": image_path}
                })
            }],
            sender=user_agent
        )
        try:
            store_data = json.loads(store_result) if isinstance(store_result, str) else store_result
            return store_data
        except json.JSONDecodeError:
            return {"error": f"Failed to parse ChromaDBAgent response: {store_result}"}

    elif "search" in prompt.lower() or "query" in prompt.lower():
        # Step 1: Ask EmbeddingAgent to generate embedding for query text
        embed_result = embedding_agent.generate_reply(
            messages=[{
                "content": json.dumps({
                    "text": prompt
                })
            }],
            sender=user_agent
        )
        try:
            embed_data = json.loads(embed_result) if isinstance(embed_result, str) else embed_result
            if "error" in embed_data:
                return embed_data
            embedding = embed_data["result"]
        except json.JSONDecodeError:
            return {"error": f"Failed to parse EmbeddingAgent response: {embed_result}"}

        # Step 2: Ask ChromaDBQueryAgent to query the collection with distance threshold
        query_result = query_agent.generate_reply(
            messages=[{
                "content": json.dumps({
                    "query_embedding": embedding,
                    "n_results": 3,
                    "distance_threshold": distance_threshold
                })
            }],
            sender=user_agent
        )
        try:
            query_data = json.loads(query_result) if isinstance(query_result, str) else query_result
            return query_data
        except json.JSONDecodeError:
            return {"error": f"Failed to parse ChromaDBQueryAgent response: {query_result}"}

    return {"error": "Invalid prompt. Use 'describe' or 'image' for image processing, or 'search'/'query' for semantic search."}


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Example 1: Process and store an image description
    image_path = r"C:\Users\ktb3kor\Downloads\Projects\multiagent-vision-intelligence\Koala.jpg"
    store_result = run_workflow(
        image_path=image_path,
        prompt="describe"
    )
    print("Store Result:")
    print(json.dumps(store_result, indent=2))

    # Example 2: Perform semantic search
    search_result = run_workflow(
        prompt="search for an image with koala in the forest with fur and sitting on tree",
        distance_threshold=0.9
)
    print("\nSearch Result:")
    print(json.dumps(search_result, indent=2))