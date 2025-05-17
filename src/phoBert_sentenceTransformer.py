import os
import json
import torch
import py_vncorenlp
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sklearn.preprocessing import normalize

class PhoBERT_SENTENCE_TRANSFORMER:
    def __init__(self,
                 model_dir: str,
                 vectorizer_path: str,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "PhoBERT_Embedded_Law_Retrieval"):
        self.vncore_model = py_vncorenlp.VnCoreNLP(save_dir=model_dir)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.sentence_transformer = sentence_embedded_model = SentenceTransformer(os.path.join(vectorizer_path, "PhoBert"), device='cuda' if torch.cuda.is_available() else 'cpu' )

    def clean_query(self, query):
        query_list = self.vncore_model.word_segment(query) # segment
        cleaned_query = " ".join(query_list)
        return cleaned_query

    def vectorize_query(self, query):
        cleaned_query = self.clean_query(query)
        query_vector = self.sentence_transformer.encode(cleaned_query)
        query_vector = normalize(query_vector.reshape(1, -1))[0]
        return query_vector
    
    def search(self, query, limit: int = 10):
        query_vector = self.vectorize_query(query)
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

if __name__ == "__main__":
    current_dir = os.getcwd()
    PROJECT_DIR = os.path.dirname(current_dir)

    retriever = PhoBERT_SENTENCE_TRANSFORMER(
        model_dir="D:/VnCoreNLP",
        vectorizer_path=os.path.join(PROJECT_DIR, "vectorizer")
    )

    query = "Đập phá biển báo “khu vực biên giới” bị phạt thế nào?"
    results = retriever.search(query, limit=3)
    retriever.print_results(results)