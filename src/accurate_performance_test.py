import os
import json

from phoBert_sentenceTransformer import PhoBERT_SENTENCE_TRANSFORMER
from tf_idf_retrieval import TF_IDF_LegalDocumentRetriever
from tf_idf_BM25search_retrieval import TF_IDF_BM25SEARCH
from tf_idf_W2V import TF_IDF_W2V

CUR_DIR = os.getcwd()
PROJECT_DIR = os.path.join(CUR_DIR, "..")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

data_path = os.path.join(DATASET_DIR, "train_question_answer.json")

with open(data_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
data_items = data["items"]
test_items = data_items[2556:]
n = len(test_items)

def get_accurate_score(instance):
    score: float = 0.0
    for item in test_items:
        query = item["question"]
        relevant_articles = item["relevant_articles"]
        answer = instance.top_n_answer(query, 10)
        m = len(relevant_articles)
        for article in relevant_articles:
            if article in answer:
                score += (1/m)
    return (score/n)

model_dir="D:/VnCoreNLP"
vectorizer_path=os.path.join(PROJECT_DIR, "vectorizer")
dataset_dir=os.path.join(PROJECT_DIR, "dataset")

phobert = PhoBERT_SENTENCE_TRANSFORMER(model_dir=model_dir, vectorizer_path=vectorizer_path)
tfidf_base = TF_IDF_LegalDocumentRetriever(model_dir=model_dir,dataset_dir=dataset_dir, vectorizer_path=vectorizer_path)
bm25 = TF_IDF_BM25SEARCH(model_dir=model_dir,dataset_dir=dataset_dir, vectorizer_path=vectorizer_path)
w2v = TF_IDF_W2V(model_dir=model_dir,dataset_dir=dataset_dir, vectorizer_path=vectorizer_path)

phobert_score = get_accurate_score(phobert)
tfidf_base_score = get_accurate_score(tfidf_base)
bm25_score = get_accurate_score(bm25)
w2v_score = get_accurate_score(w2v)

scores = {"phobert_score": phobert_score, "tfidf_base_score": tfidf_base_score, "bm25_score": bm25_score, "w2v_score": w2v_score}
print(scores)