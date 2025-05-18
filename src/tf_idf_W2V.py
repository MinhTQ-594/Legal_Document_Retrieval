import os
import numpy as np
from gensim.models import Word2Vec

from sklearn.preprocessing import normalize
from tf_idf_retrieval import TF_IDF_LegalDocumentRetriever

class TF_IDF_W2V(TF_IDF_LegalDocumentRetriever):
    def __init__(self,
                 model_dir: str,
                 dataset_dir: str,
                 vectorizer_path: str,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "Word2Vec_Law_document_retrivial"):
        super().__init__(model_dir=model_dir, dataset_dir=dataset_dir, vectorizer_path=vectorizer_path, qdrant_host=qdrant_host, qdrant_port=qdrant_port, collection_name=collection_name)
        self.w2v_model = Word2Vec.load(os.path.join(vectorizer_path,"w2v", "w2v.model"))

    def get_weighted_sentence_vector(self, sentence: str):
        tfidf_scores = self.vectorizer.transform([sentence])
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_dict = {
            feature_names[col]: tfidf_scores[0, col]
            for col in tfidf_scores.nonzero()[1]
        }

        words = sentence.split()
        word_vecs = []
        weights = []
        for word in words:
            if word in self.w2v_model.wv and word in tfidf_dict:
                vec = self.w2v_model.wv[word]
                weight = tfidf_dict[word]
                word_vecs.append(vec * weight)
                weights.append(weight)

        if word_vecs:
            return np.sum(word_vecs, axis=0) / np.sum(weights)
        else:
            return np.zeros(self.w2v_model.vector_size)
    
    def vectorize_process_query(self, query: str):
        cleaned_query = self.clean_query(query)
        query_vec = self.get_weighted_sentence_vector(cleaned_query)
        query_vector = normalize(query_vec.reshape(1, -1))[0]
        return query_vector
    
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

    retriever = TF_IDF_W2V(
        model_dir="D:/VnCoreNLP",
        dataset_dir=os.path.join(PROJECT_DIR, "dataset"),
        vectorizer_path=os.path.join(PROJECT_DIR, "vectorizer")
    )

    query = "Thừa phát lại được tống đạt những giấy tờ, hồ sơ, tài liệu nào?"
    results = retriever.search(query, limit=3)
    retriever.print_results(results)