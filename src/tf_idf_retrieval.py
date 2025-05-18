import os
import joblib
import re
import string
import jpype
from qdrant_client import QdrantClient
from sklearn.preprocessing import normalize
from vncorenlp_singleton import get_vncore_model

class TF_IDF_LegalDocumentRetriever:
    def __init__(self,
                 model_dir: str,
                 dataset_dir: str,
                 vectorizer_path: str,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "tfidf_search_Law_document_retrivial"):
        self.vncore_model = get_vncore_model(model_dir=model_dir)
        ## Initial the segmentation model and the pattern to remove the stop word
        with open(os.path.join(dataset_dir, "stopwords_processed.txt"), "r", encoding="utf-8") as f:
            stopwords_list = list(map(str.strip, f))
        self.pattern = r"\b(" + "|".join(map(re.escape, stopwords_list)) + r")\b"

        ## load vectorizer
        self.vectorizer = joblib.load(os.path.join(vectorizer_path, 'tfidf_vectorizer.pkl'))

        # Connect to Qdrant
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=60.0)
        self.collection_name = collection_name
    
    def clean_query(self, query: str) -> str:
        segmented = self.vncore_model.word_segment(query)
        cleaned = " ".join(segmented)
        cleaned = re.sub(self.pattern, "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        query_word = [token for token in cleaned.split() if token not in string.punctuation]
        cleaned_query = ' '.join(query_word)
        return cleaned_query
    
    def vectorize_process_query(self, query: str):
        cleaned_query = self.clean_query(query)
        query_vec = self.vectorizer.transform([cleaned_query])
        vectored_query = normalize(query_vec).toarray()[0]
        return vectored_query
    
    def search(self, query: str, limit: int = 10):
        query_vector = self.vectorize_process_query(query)
        
        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False
        ) 
        return hits
    
    def print_results(self, hits):
        for hit in hits.points:
            print(f"Score: {hit.score:.4f} | law_id: {hit.payload['law_id']} | article_id: {hit.payload['article_id']}")

    def top_n_answer(self, query, n):
        answers = []
        hits = self.search(query, n)
        for hit in hits.points:
            answers.append({"law_id": hit.payload['law_id'], "article_id": hit.payload['article_id']})
        
        return answers

if __name__ == "__main__":
    current_dir = os.getcwd()
    PROJECT_DIR = os.path.dirname(current_dir)

    retriever = TF_IDF_LegalDocumentRetriever(
        model_dir="D:/VnCoreNLP",
        dataset_dir=os.path.join(PROJECT_DIR, "dataset"),
        vectorizer_path=os.path.join(PROJECT_DIR, "vectorizer")
    )

    query = "Đập phá biển báo “khu vực biên giới” bị phạt thế nào?"
    results = retriever.search(query, limit=3)
    retriever.print_results(results)
