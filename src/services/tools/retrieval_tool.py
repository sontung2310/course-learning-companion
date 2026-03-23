from src.services.retrieval import RetrievalService
from crewai.tools import BaseTool


retrieval_service = RetrievalService()


class RetrievalTool(BaseTool):
    """Tool for retrieving information using Retrieval service."""

    name: str = "retrieval"
    description: str = "Retrieve relevant documents and information using vector search through Retrieval service"

    def _run(self, question: str) -> str:
        """Run the tool with the given query."""
        try:
            return retrieval_service.retrieve_vector(question)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
