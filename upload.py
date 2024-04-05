import os
import sys
from pinecone import Pinecone, PodSpec
import re
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.node_parser import SentenceSplitter
from llama_index.ingestion import IngestionPipeline
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SemanticSplitterNodeParser

from tqdm import tqdm

# use llama_index version 0.9.29
# Load Pinecone API key from environment variable
api_key = os.environ.get("PINECONE_API_KEY")
if api_key is None:
    print("PINECONE_API_KEY is not set")
    sys.exit(1)

print("Pinecone API key loaded")


#https://docs.pinecone.io/guides/indexes/create-an-index
pc = Pinecone(api_key=api_key)
index_name = "llama-index-rag-nlp-coms579"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension=384,  # since BAAI/bge-small-en-v1.5 has a dimension size of 384
        #https://huggingface.co/BAAI/bge-small-en-v1.5
        metric="cosine",
        spec=PodSpec(environment="gcp-starter")
    )
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    



# read pdf file function using llamaindex

def read_pdf_file(file_name):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'Docs', file_name)
    document = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return document

def clean_up_text(document):
    # Remove hyphenated word breaks
    #document = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', document)
    # use this link for reference:
    #https://github.com/pinecone-io/examples/blob/master/learn/generation/llama-index/using-llamaindex-with-pinecone.ipynb
    # Remove unwanted patterns
    pass

def splitter_function(document, chunk_size = 128, overlap = 0.25):
    chunk_overlap = int(chunk_size * overlap)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(document)
    print(nodes[0])
    return nodes

def create_embeddings(nodes):
    #https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
    embeddings = []
    for i in tqdm(range(0, len(nodes), 1)):
        embed = embed_model.get_text_embedding(nodes[i])
        embeddings.append(embed)
    print(len(nodes), len(embeddings))
    return embeddings

def upload_to_pinecone(embeddings, nodes):
    index = pc.Index(index_name)
    ids = [str(i) for i in range(len(nodes))]
    index.upsert(vectors=list(zip(ids, embeddings)))
    # batch_size = 1
    # for i in tqdm(range(0, len(nodes), 1)):
    #     # Adjust how you access the node ID and convert embeddings to list format
        
    #     node_id = str(nodes[i].get_id())  # Adjust .get_id() as necessary
        
    #     # Ensure embeddings[i] is in the correct format for Pinecone (usually a list of floats)
    #     embedding = embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else embeddings[i]
        
    #     # Upsert the current node's ID and embedding to Pinecone
    #     index.upsert(vectors=[(node_id, embedding)])
    
    print("Upload complete")






read_pdf_file("Attention_is_all_you_need.pdf")
nodes = splitter_function(read_pdf_file("Attention_is_all_you_need.pdf"))
embeddings = create_embeddings(nodes)
upload_to_pinecone(embeddings, nodes)



    



    