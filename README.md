# Vietnamese keyword extraction using Gensim

[![Python 3.10.7](https://img.shields.io/badge/python-3.10.7-blue)](https://www.python.org/downloads/release/python-3107/)[![Gensim 4.2.0](https://img.shields.io/badge/Gensim-4.2.0-purple)](https://pypi.org/project/gensim/)[![underthesea 1.3.3](https://img.shields.io/badge/underthesea-1.3.3-blue)](https://pypi.org/project/underthesea/)[![numpy 1.23.3](https://img.shields.io/badge/numpy-1.23.3-blue)](https://pypi.org/project/numpy/)

Vietnamese keyword extraction using LSA, LDA, NMF techniques with Gensim package

## Approach
Assuming that number of topics is the number of documents in the corpus, we use the topic-term matrix to extract the existing top keywords of the documents.  

## Installation
Create virtual environment then install required packages:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
The dataset includes 18 VNExpress articles in 6 different fields and one article for testing.

## Usage
```bash
keyword_extraction.py [-h] --input_file INPUT_FPATH --result OUTPUT_FPATH [--model_type {lsa,lda,nmf}]

Vietnamese keyword extraction

options:
  -h, --help            show this help message and exit
  --input_file INPUT_FPATH
                        Input file path
  --result OUTPUT_FPATH
                        Output file path
  --model_type {lsa,lda,nmf}
                        keyword extraction model type
```