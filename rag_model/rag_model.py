import os

class RAG:
    """
    Full RAG model
    """
    def __init__(self, args) -> None:
        """
        Init
        """
        print("Initializing RAG model....")
        self.args = args
        self.pdf_file = self.args.pdf

        # Check if PDF file exits6

        return
    
    def start_rag(self) -> None:
        """
        Main function for the RAG Process
        """
        # Initialize Pinecone Pipeline
        # Read the PDF
        # Preprocessing the data
        # Indexing the data
        pass
    
    def __str__(self) -> str:
        """
        Class string
        """
        return "RAG123: pdf="+self.pdf_file