from tf_idf_retrieval import TF_IDF_LegalDocumentRetriever
from tf_idf_BM25search_retrieval import TF_IDF_BM25SEARCH
from tf_idf_W2V import TF_IDF_W2V
from phoBert_sentenceTransformer import PhoBERT_SENTENCE_TRANSFORMER
from tf_idf_Glove import TF_IDF_GLOVE
import os

current_dir = os.getcwd()
PROJECT_DIR = os.path.dirname(current_dir)

MODEL_DIR=os.path.join(PROJECT_DIR, "VnCoreNLP")
DATASET_DIR=os.path.join(PROJECT_DIR, "dataset")
VECTORIZER_PATH=os.path.join(PROJECT_DIR, "vectorizer")

def tfidf_retrieval(query):
    try:
        retriever = TF_IDF_LegalDocumentRetriever(MODEL_DIR, DATASET_DIR, VECTORIZER_PATH)
        hits = retriever.search(query, limit=1)
        retriever.print_results(hits)
        hit = hits.points[0]
        return f"Score: {hit.score:.4f} | law_id: {hit.payload['law_id']} | article_id: {hit.payload['article_id']}"
    except Exception as e:
        return f"Error: {str(e)}"

def tf_idf_bm25(query):
    try:
        retriever = TF_IDF_BM25SEARCH(MODEL_DIR, DATASET_DIR, VECTORIZER_PATH)
        hits = retriever.search(query, limit=1)
        retriever.print_results(hits)
        hit = hits[0]
        return f"Score: {hit['score']} | law_id : {hit['payload']['law_id']} | article_id : {hit['payload']['article_id']}"
    except Exception as e:
        return f"Error: {str(e)}"


def tf_idf_w2v(query):
    try:
        retriever = TF_IDF_LegalDocumentRetriever(MODEL_DIR, DATASET_DIR, VECTORIZER_PATH)
        hits = retriever.search(query, limit=1)
        retriever.print_results(hits)
        hit = hits.points[0]
        return f"Score: {hit.score:.4f} | law_id: {hit.payload['law_id']} | article_id: {hit.payload['article_id']}"
    except Exception as e:
        return f"Error: {str(e)}"
def phoBert_sentence_transformers(query):
    try:
        retriever = PhoBERT_SENTENCE_TRANSFORMER(MODEL_DIR, VECTORIZER_PATH)
        hits = retriever.search(query, limit=1)
        retriever.print_results(hits)
        hit = hits.points[0]
        return f"Score: {hit.score:.4f} | law_id: {hit.payload['law_id']} | article_id: {hit.payload['article_id']}"
    except Exception as e:
        return f"Error: {str(e)}"

def tf_idf_glove(query):
    try:
        retriever = TF_IDF_GLOVE(MODEL_DIR, DATASET_DIR, VECTORIZER_PATH)
        hits = retriever.search(query, limit=1)
        retriever.print_results(hits)
        hit = hits.points[0]
        return f"Score: {hit.score:.4f} | law_id: {hit.payload['law_id']} | article_id: {hit.payload['article_id']}"
    except Exception as e:
        return f"Error: {str(e)}"

def predict(query: str, model_name: str) -> str:
    model_map = {
        "tf-idf_retrival": tfidf_retrieval,
        "tf_idf_bm25": tf_idf_bm25,
        "tf_idf_w2v": tf_idf_w2v,
        "phoBert_sentences-transformers": phoBert_sentence_transformers,
        "tf_idf_glove": tf_idf_glove
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    print(f"Using model: {model_name}")
    return model_map[model_name](query)
