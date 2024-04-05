import os
import sys
import pinecone
import re
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.node_parser import SimpleNodeParser

# use llama_index version 0.9.29
# Load Pinecone API key from environment variable
 
api_key = os.environ.get("PINECONE_API_KEY")
if api_key is None:
    print("PINECONE_API_KEY is not set")
    sys.exit(1)

print("Pinecone API key loaded")

# read pdf file function using llamaindex

def read_pdf_file(file_name):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'Docs', file_name)
    document = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return document

def clean_up_text(document):
    # Remove hyphenated word breaks
    document = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', document)
    # Remove unwanted patterns
    pass



read_pdf_file("Attention_is_all_you_need.pdf")



    



    