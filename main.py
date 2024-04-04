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

    print("Initialized RAG")
    print(rag)


    # TBD


if __name__ == "__main__":
    main()