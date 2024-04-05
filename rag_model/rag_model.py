import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from rag_model.data_processing import DataProcessor
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.legacy.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

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
        self.pdf = self.args.pdf

        # Check if PDF file exits6

        return

    def init_pipeline(self) -> None:
        """
        Initializes Pinecone Pipeline
        """
        print("Initializing Pinecone")
        pass

    def start_rag(self) -> None:
        """
        Main function for the RAG Process
        """
        # Initialize Pinecone Pipeline
        self.init_pipeline()
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
        llm = OpenAI(model="gpt-3.5-turbo-0125")

        Settings.llm = llm
        Settings.embed_model = embed_model
        # Read and preprocess the PDF
        self.documents = DataProcessor.process_data(os.path.join(os.getcwd(), self.pdf))
        print(len(self.documents))

        # Indexing the data and creating VectorDB
        node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8)

        nodes = node_parser.get_nodes_from_documents(self.documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        recursive_index = VectorStoreIndex(nodes=base_nodes+objects)
        raw_index = VectorStoreIndex.from_documents(self.documents)

        reranker = FlagEmbeddingReranker(
            top_n=5,
            model="BAAI/bge-reranker-large",
        )

        recursive_query_engine = recursive_index.as_query_engine(
            similarity_top_k=15, 
            node_postprocessors=[reranker], 
            verbose=True
        )

        raw_query_engine = raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker])
        print(len(nodes))

        query = "What are the applications of quantum computings?"

        response_1 = raw_query_engine.query(query)
        print("\n***********New LlamaParse+ Basic Query Engine***********")
        print(response_1)

        pass
    
    def __str__(self) -> str:
        """
        Class string
        """
        return "RAG: pdf="+self.pdf