import os
import json
import joblib
import py_vncorenlp
import re
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class TF_IDF_LegalDocumentRetriever:
    def __init__(self,
                 model_dir: str,
                 dataset_dir: str,
                 vectorizer_path: str,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "tfidf_search_Law_document_retrivial"):
        # Load VnCoreNLP
        self.vncore_model = py_vncorenlp.VnCoreNLP(save_dir=model_dir)

        ## Initial the segmentation model and the pattern to remove the stop word
        with open(os.path.join(dataset_dir, "stopwords_processed.txt"), "r", encoding="utf-8") as f:
            stopwords_list = list(map(str.strip, f))
        self.pattern = r"\b(" + "|".join(map(re.escape, stopwords_list)) + r")\b"

        ## load vectorizer
        self.vectorizer = joblib.load(os.path.join(vectorizer_path, 'tfidf_vectorizer.pkl'))

        # Connect to Qdrant
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
    
    def _preprocess_query(self, query: str) -> str:
        segmented = self.vncore_model.word_segment(query)
        cleaned = " ".join(segmented)
        cleaned = re.sub(self.pattern, "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    
    def _encode_query(self, cleaned_query: str):
        query_vec = self.vectorizer.transform([cleaned_query])
        return normalize(query_vec).toarray()[0]
    
    def process_query(self, query: str):
        cleaned_query = self._preprocess_query(query)
        vectored_query = self._encode_query(cleaned_query)
        return vectored_query
    
    def search(self, query: str, limit: int = 3):
        query_vector = self.process_query(query)
        
        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=3,
            with_payload=True,
            with_vectors=False
        ) 
        return hits
    
    def print_results(self, hits):
        for hit in hits.points:
            print(f"Score: {hit.score:.4f} | law_id: {hit.payload['law_id']} | article_id: {hit.payload['article_id']}")


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
