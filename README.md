# COMS579-S24
RAG - NLP Project for spring 2024 - COMS-579 - TEAM Members Aditya Kota and Ankush Kumar Mishra

## Directory Structure
'''
your_project_directory/
│
├── upload.py
│
└── Docs/
    └── your_pdf_file.pdf
'''

In the current working directory there should be a "Docs" folder. The pdf file should be inside it.

## Installation Instruction


'pip install -r requirements.txt'




## How to run from Terminal

### Required Arguments

**--api_key**
Your Pinecone API key.

**--file_name**
The name of the PDF file to process.

### Optional Arguments

**--chunk_size**
Chunk size for splitting. Default = 128

**--overlap**
Overlap ratio between chunk. Should be between 0 and 1. Default 0.25




'
python3 upload.py --api_key <your_pinecone_api_key> --file_name <pdf_filename>
'

'
python3 upload.py --api_key <your_pinecone_api_key> --file_name <pdf_filename> --chunk_size <chunk_size_(default_128)> --overlap <overlap_ratio(default_0.25)>
'

## For help

'
python3 upload.py --help
'



