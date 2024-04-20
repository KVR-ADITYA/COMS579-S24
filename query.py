import os
import sys
from pinecone import Pinecone, PodSpec
import re
from llama_index.core.node_parser import SentenceSplitter


from ctransformers import AutoModelForCausalLM
from llama_index.core import  VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.schema import TextNode, BaseNode, IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.legacy.vector_stores import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from tqdm import tqdm
import argparse
# use llama_index version 0.9.29
# Load Pinecone API key from environment variable
# api_key = os.environ.get("PINECONE_API_KEY")
# if api_key is None:
#     print("PINECONE_API_KEY is not set")
#     sys.exit(1)

# print("Pinecone API key loaded")

# Function to load the Pinecone API key
# https://docs.pinecone.io/integrations/llamaindex
# def load_pinecone_api_key():
#     api_key = os.environ.get("PINECONE_API_KEY")
#     if api_key is None:
#         print("PINECONE_API_KEY is not set")
#         sys.exit(1)
#     print("Pinecone API key loaded")
#     return api_key


# #https://docs.pinecone.io/guides/indexes/create-an-index

# pc = Pinecone(api_key=api_key)
# index_name = "llama-index-rag-nlp-coms579"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name = index_name,
#         dimension=768,
#         metric="cosine",
#         spec=PodSpec(environment="gcp-starter")
#     )

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index_name = "llama-index-rag-nlp-coms579"

# Function to create a Pinecone index
def create_pinecone_index(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Adjust as necessary
            metric="cosine",
            spec=PodSpec(environment="gcp-starter")
        )
    return pc    

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
    print("Splitting complete")
    print("Number of chunks: ", len(nodes))
    print("First Chunk\n", nodes[0])
    return nodes

def create_embeddings(nodes):
    #https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
    embeddings = []
    for i in tqdm(range(0, len(nodes), 1)):
        embed = embed_model.get_text_embedding(nodes[i])
        embeddings.append(embed)
    print(len(nodes), len(embeddings))
    return embeddings

def upload_to_pinecone(pc, index_name, embeddings, nodes):
    index = pc.Index(index_name)
    ids = [str(i) for i in range(len(nodes))]
    index.upsert(vectors=list(zip(ids, embeddings)))
    print("Upload complete")

# Custom type function for overlap
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x

def main():
    # Load Pinecone API key
    parser = argparse.ArgumentParser(description="queries on the uploaded document")
    parser.add_argument("--api_key", type=str, required=True, help="Pinecone API key")
    parser.add_argument("--query", type=str, required=True, help="query regarding the pdf")
    parser.add_argument("--top_k", type=int, default=5, help="The number of top K documents to be considered")
    args = parser.parse_args()
    query = args.query
    top_k = args.top_k
    print("Question: " + query + " top k:" + str(top_k))


    pc = Pinecone(api_key=args.api_key)

    pc_index = pc.Index(index_name)


    # Create embeddings for the query
    Settings.embed_model = embed_model
    query_embedding = embed_model.get_text_embedding(query)
    print(query_embedding)

    # Get uery result from Pinecone
    model_url= "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
    llm = LlamaCPP(
        model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        #model_path="./llm/llama-2-7b-chat.Q2_K.gguf",
        temperature=0.1,
        max_new_tokens=384,
        context_window=3000,
        generate_kwargs={},
        verbose=False,) 
    
    # vector_store = PineconeVectorStore(pinecone_index=pc_index)
    

    # vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


    # # Grab 5 search results
    # retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

    # # # Query vector DB
    # answer = retriever.retrieve(query)

    # # # Inspect results
    # print([i.get_content() for i in answer])

    # # Pass in your retriever from above, which is configured to return the top 5 results
    # query_engine = RetrieverQueryEngine(retriever=retriever)

    # # Now you query:
    # llm_query = query_engine.query(query)
    # print(llm_query.response)
    
    query_result = pc_index.query(vector = query_embedding, top_k=top_k, include_values=True, include_metadata=True)



    # print(query_result.to_str())

    # print("Query Result: " + str(query_result))
    _nodes= []
    print("Query: ", query)
    print("Retrieval")
    print("top-k :", top_k)
    for i, _t in enumerate(query_result['matches']):
        print(i, end=':  ')
        # print(_t)
        _node =IndexNode.from_text_node(TextNode(text=_t['metadata']['text']), _t['id'])
        _nodes.append(_node)
        print("Similarity: ", _t['score'])
        print("---------------------------")
        
    print(_nodes)

    if _nodes is not None and len(_nodes) >= 1 and isinstance(_nodes[0], BaseNode):
        print("----------------------------ENTER --------------------------")
    # create vector store index
    _index = VectorStoreIndex(_nodes)

    #Re-rank
    query_engine = _index.as_query_engine(similarity_top_k=top_k, llm=llm)
    response = query_engine.query(query)
    print("Answer: ")
    print(str(response))

if __name__ == "__main__":
    main()

