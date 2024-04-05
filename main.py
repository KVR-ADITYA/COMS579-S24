import os
import sys
import argparse
from rag_model.rag_model import RAG


def get_args():
    parse_string = argparse.ArgumentParser("Path to pdf file")
    parse_string.add_argument("--pdf", help="path to the pdf file", type=str)
    return parse_string.parse_args()


def main():
    print("hello")
    # Parse the command line args
    args = get_args()
    # print(args.pdf)


    # Instantiate the RAG Model
    rag = RAG(args)
    rag.start_rag()

    print("Initialized RAG")
    print(rag)


    # TBD


if __name__ == "__main__":
    os.environ["PINECONE_API_KEY"] = "29e930bd-afd3-436a-8782-40d774866b10"
    # API access to llama-cloud
    os.environ["LLAMA_CLOUD_API_KEY"] = "llx-

    # Using OpenAI API for embeddings/llms
    os.environ["OPENAI_API_KEY"] = "sk-"
    main()