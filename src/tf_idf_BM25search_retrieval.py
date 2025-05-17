import os
import pickle
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from tf_idf_retrieval import TF_IDF_LegalDocumentRetriever

class TF_IDF_BM25SEARCH(TF_IDF_LegalDocumentRetriever):
    def __init__(self,
                 model_dir: str,
                 dataset_dir: str,
                 vectorizer_path: str,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "tfidf_search_Law_document_retrivial"):
        super().__init__(model_dir=model_dir, dataset_dir=dataset_dir, vectorizer_path=vectorizer_path, qdrant_host=qdrant_host, qdrant_port=qdrant_port, collection_name=collection_name)
        with open(os.path.join(vectorizer_path, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
    
    def bm25_topN(self, query, N = 50):
        cleaned_query = self.clean_query(query)
        bm25_scores = self.bm25.get_scores(cleaned_query.split())
        top_n_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:N]

        return bm25_scores, top_n_idx
    
    def search(self, query, limit):
        query_vector = self.vectorize_process_query(query)
        bm25_scores, top_n_idx = self.bm25_topN(query, N = 50)

        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=50,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=top_n_idx)  # danh sách ID (int hoặc str)
                    )
                ]
            )
        )

        final_results = []
        for hit in hits.points:
            qdrant_score = hit.score               # cosine similarity
            doc_id = hit.id                        # id của tài liệu
            bm25_score = bm25_scores[int(doc_id)]  # phải đảm bảo doc_id khớp thứ tự BM25

            # Normalize BM25 score
            max_bm25 = max(bm25_scores) or 1
            bm25_score_norm = bm25_score / max_bm25

            # Kết hợp
            alpha = 0.7
            combined_score = alpha * qdrant_score + (1 - alpha) * bm25_score_norm

            final_results.append({
                "id": doc_id,
                "score": combined_score,
                "qdrant_score": qdrant_score,
                "bm25_score": bm25_score_norm,
                "payload": hit.payload
            })

        # Sắp xếp kết quả
        final_results.sort(key=lambda x: x["score"], reverse=True)

        return final_results[:limit]
    
    def print_results(self, final_results):
        for result in final_results:
            print(f"Score: {result['score']} | law_id : {result['payload']['law_id']} | article_id : {result['payload']['article_id']}")
        
    
if __name__ == "__main__":
    current_dir = os.getcwd()
    PROJECT_DIR = os.path.dirname(current_dir)

    retriever = TF_IDF_BM25SEARCH(
        model_dir="D:/VnCoreNLP",
        dataset_dir=os.path.join(PROJECT_DIR, "dataset"),
        vectorizer_path=os.path.join(PROJECT_DIR, "vectorizer")
    )

    query = "Mục tiêu phát triển khu công nghiệp hỗ trợ được quy định như thế nào?"
    results = retriever.search(query, limit=3)
    retriever.print_results(results)
    

