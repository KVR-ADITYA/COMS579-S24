import os
from llama_parse import LlamaParse


class DataProcessor:
    """
    This is class that handles everything related to data preprocessing
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def process_data(pdf):
        print("Processing Data.....")
        # Read the PDF
        parser = LlamaParse(
            result_type="markdown",  # "markdown" and "text" are available
            # verbose=True,
        )
        print(pdf)
        documents = parser.load_data(pdf)

        # print(documents)
        return documents

        pass