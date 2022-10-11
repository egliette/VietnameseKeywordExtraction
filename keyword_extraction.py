import argparse

from gensim.models import LsiModel, LdaModel, Nmf

from utils import load_corpus, create_doc_term_matrix, get_keywords


def main(input_fpath, output_fpath, model_type):
    corpus = load_corpus("data")
    with open(input_fpath, "r", encoding="utf-8") as input_file:
        input_doc = input_file.read()
        corpus.append(input_doc)
    
    dictionary, doc_term_matrix = create_doc_term_matrix(corpus)

    if model_type == "lsa":
        model = LsiModel(doc_term_matrix, 
                         id2word=dictionary, 
                         num_topics=len(doc_term_matrix))
    elif model_type == "lda":
        model = LdaModel(doc_term_matrix, 
                         id2word=dictionary, 
                         num_topics=len(doc_term_matrix))
    elif model_type == "nmf":
        model = Nmf(doc_term_matrix, 
                    id2word=dictionary, 
                    num_topics=len(doc_term_matrix))
    
    keywords = get_keywords(dictionary, doc_term_matrix, model)
    
    with open(output_fpath, "w", encoding="utf-8") as output_file:
        for keyword in keywords:
            output_file.write(keyword + "\n")
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese keyword extraction")

    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Input file path",
                        dest="input_fpath")
    parser.add_argument("--result",
                        type=str,
                        required=True,
                        help="Output file path",
                        dest="output_fpath")
    parser.add_argument("--model_type",
                        type=str,
                        default="lsa",
                        choices=["lsa", "lda", "nmf"],
                        help="keyword extraction model type",
                        dest="model_type")
                    

    args = parser.parse_args()

    main(**vars(args))
