import os
import json
import torch
import pymongo
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Import Retriever Classes ---
from phoBert_sentenceTransformer import PhoBERT_SENTENCE_TRANSFORMER
from tf_idf_retrieval import TF_IDF_LegalDocumentRetriever
from tf_idf_BM25search_retrieval import TF_IDF_BM25SEARCH
from tf_idf_W2V import TF_IDF_W2V
# vncorenlp_singleton will be used by the imported retrievers

# --- Global Configurations ---
# LLM Configuration
LLM_MODEL_ID = "google/gemma-3-1b-it"
HF_TOKEN = os.environ.get("HF_TOKEN", "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") # Recommended: login via huggingface-cli

# Atlas pretrained_phobert Retriever Configuration
ORIGINAL_EMBEDDING_MODEL_NAME = 'dangvantuan/vietnamese-embedding'
ATLAS_SRV_STRING = os.environ.get("ATLAS_SRV_STRING", "mongodb+srv://username:02122004@rag-legal-cluster.gydcd6m.mongodb.net/?retryWrites=true&w=majority&appName=rag-legal-cluster") 
ATLAS_DATABASE_NAME = "zalo_ai_legal_db"
ATLAS_COLLECTION_NAME = "legal_articles_full"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Paths
# /src/
CUR_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_GUESS = os.path.join(CUR_SCRIPT_DIR, "..")

DATASET_DIR = os.path.join(PROJECT_ROOT_GUESS, "dataset")
VECTORIZER_DIR = os.path.join(PROJECT_ROOT_GUESS, "vectorizer") # For models
#VNCORENLP_MODEL_DIR = os.environ.get("VNCORENLP_MODEL_DIR", os.path.join(PROJECT_ROOT_GUESS, "models", "vncorenlp")) 
VNCORENLP_MODEL_DIR = os.path.join(PROJECT_ROOT_GUESS, "VnCoreNLP")

# Qdrant Configurations 
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
PHOBERT_COLLECTION = "PhoBERT_Embedded_Law_Retrieval"
TFIDF_COLLECTION = "tfidf_search_Law_document_retrivial"
W2V_COLLECTION = "Word2Vec_Law_document_retrivial"
GLOVE_COLLECTION = "Glove_Law_document_retrivial"

# --- Global Dictionary for Retriever Instances ---
RETRIEVERS = {}

# -------- data for back retrieval ---------------
corpus_data_path = os.path.join(DATASET_DIR, "processed_legal_corpus.json")

with open(corpus_data_path, "r", encoding="utf-8") as file:
    corpus_data = json.load(file)

# Build the index
index = {
    law["law_id"]: {
        article["article_id"]: article
        for article in law["articles"]
    }
    for law in corpus_data
}

# --- 1. Model Loading ---
def load_llm_model():
    """Loads and configures the LLM (Gemma)."""
    gemma_tokenizer = None
    gemma_model = None
    selected_device = None

    if torch.cuda.is_available():
        selected_device = torch.device("cuda")
        print(f"CUDA device found. Using device: {selected_device}")
    else:
        selected_device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    try:
        print(f"\nLoading tokenizer for {LLM_MODEL_ID}...")
        gemma_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=HF_TOKEN)
        print("LLM Tokenizer loaded successfully.")

        print(f"\nLoading model {LLM_MODEL_ID}...")
        gemma_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            token=HF_TOKEN,
            # torch_dtype=torch.bfloat16
        )
        gemma_model.to(selected_device)
        print(f"LLM Model {LLM_MODEL_ID} loaded successfully on {gemma_model.device}.")
        if gemma_tokenizer.pad_token_id is None:
            gemma_tokenizer.pad_token_id = gemma_tokenizer.eos_token_id
            print("Set LLM tokenizer pad_token_id to eos_token_id.")

    except Exception as e:
        print(f"An error occurred loading LLM {LLM_MODEL_ID}: {e}")
        print(f"Ensure you have accepted terms for {LLM_MODEL_ID} on Hugging Face and are logged in or provided a token.")
        gemma_tokenizer, gemma_model = None, None
    
    return gemma_tokenizer, gemma_model, selected_device

def load_original_pretrained_phobert_model():
    """Loads the original pretrained_phobert model for Atlas-based retrieval."""
    pretrained_phobert_model = None
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pretrained_phobert_model = SentenceTransformer(ORIGINAL_EMBEDDING_MODEL_NAME, device=selected_device)
        print(f"Successfully loaded original pretrained_phobert model '{ORIGINAL_EMBEDDING_MODEL_NAME}' on {selected_device}.")

        hf_model_config_max_pos_embed = pretrained_phobert_model[0].auto_model.config.max_position_embeddings
        compatible_max_seq_length = hf_model_config_max_pos_embed - 2 # Account for [CLS] and [SEP]
        current_st_max_seq_length = pretrained_phobert_model[0].max_seq_length
        tokenizer_max_len = pretrained_phobert_model.tokenizer.model_max_length
        final_max_seq_len = min(compatible_max_seq_length, tokenizer_max_len, current_st_max_seq_length)

        if current_st_max_seq_length != final_max_seq_len:
            print(f"Adjusting SentenceTransformer's max_seq_length from {current_st_max_seq_length} to {final_max_seq_len}.")
            pretrained_phobert_model[0].max_seq_length = final_max_seq_len
        print(f"Effective max_seq_length for pretrained_phobert model: {pretrained_phobert_model[0].max_seq_length}")

    except Exception as e:
        print(f"Error loading original pretrained_phobert model '{ORIGINAL_EMBEDDING_MODEL_NAME}': {e}")
    return pretrained_phobert_model

def initialize_all_retrievers(original_pretrained_phobert_instance=None):
    """Initializes all available retriever models and stores them in RETRIEVERS dict."""
    global RETRIEVERS
    print("\nInitializing retrievers...")

    # 1. Original pretrained_phobert + Atlas
    if original_pretrained_phobert_instance:
        RETRIEVERS['original_pretrained_phobert_atlas'] = {
            "instance": original_pretrained_phobert_instance,
            "type": "ATLAS_pretrained_phobert",
            "description": "pretrained_phobert ('dangvantuan/vietnamese-embedding') with MongoDB Atlas Vector Search"
        }
        print(f"SUCCESS: Initialized 'original_pretrained_phobert_atlas' retriever.")
    else:
        print(f"INFO: Original pretrained_phobert model not loaded, 'original_pretrained_phobert_atlas' retriever skipped.")

    # 2. PhoBERT Sentence Transformer 
    try:
        phobert_retriever = PhoBERT_SENTENCE_TRANSFORMER(
            model_dir=VNCORENLP_MODEL_DIR, 
            vectorizer_path=VECTORIZER_DIR, 
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=PHOBERT_COLLECTION
        )
        RETRIEVERS['phobert_qdrant'] = {
            "instance": phobert_retriever,
            "type": "QDRANT_PHOBERT",
            "description": "PhoBERT Sentence Transformer with Qdrant"
        }
        print(f"SUCCESS: Initialized 'phobert_qdrant' retriever.")
    except Exception as e:
        print(f"FAILED to initialize PhoBERT retriever: {e}")

    # 3. TF-IDF
    try:
        tfidf_retriever = TF_IDF_LegalDocumentRetriever(
            model_dir=VNCORENLP_MODEL_DIR, 
            dataset_dir=DATASET_DIR,      
            vectorizer_path=VECTORIZER_DIR,  
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=TFIDF_COLLECTION
        )
        RETRIEVERS['tfidf_qdrant'] = {
            "instance": tfidf_retriever,
            "type": "QDRANT_TFIDF",
            "description": "TF-IDF with Qdrant"
        }
        print(f"SUCCESS: Initialized 'tfidf_qdrant' retriever.")
    except Exception as e:
        print(f"FAILED to initialize TF-IDF (Qdrant) retriever: {e}")

    # 4. TF-IDF + BM25
    try:
        bm25_retriever = TF_IDF_BM25SEARCH(
            model_dir=VNCORENLP_MODEL_DIR,
            dataset_dir=DATASET_DIR,
            vectorizer_path=VECTORIZER_DIR, 
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=TFIDF_COLLECTION 
        )
        RETRIEVERS['bm25_qdrant'] = {
            "instance": bm25_retriever,
            "type": "QDRANT_BM25",
            "description": "TF-IDF + BM25 with Qdrant"
        }
        print(f"SUCCESS: Initialized 'bm25_qdrant' retriever.")
    except Exception as e:
        print(f"FAILED to initialize TF-IDF+BM25 retriever: {e}")

    # 5. TF-IDF + Word2Vec
    try:
        w2v_retriever = TF_IDF_W2V(
            model_dir=VNCORENLP_MODEL_DIR,
            dataset_dir=DATASET_DIR,
            vectorizer_path=VECTORIZER_DIR, 
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=W2V_COLLECTION
        )
        RETRIEVERS['w2v_qdrant'] = {
            "instance": w2v_retriever,
            "type": "QDRANT_W2V",
            "description": "TF-IDF + Word2Vec with Qdrant"
        }
        print(f"SUCCESS: Initialized 'w2v_qdrant' retriever.")
    except Exception as e:
        print(f"FAILED to initialize TF-IDF+Word2Vec retriever: {e}")

    # 6. TF-IDF + GloVe
    try:
        glove_retriever = TF_IDF_W2V(
            model_dir=VNCORENLP_MODEL_DIR,
            dataset_dir=DATASET_DIR,
            vectorizer_path=VECTORIZER_DIR, 
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=GLOVE_COLLECTION
        )
        RETRIEVERS['glove_qdrant'] = {
            "instance": w2v_retriever,
            "type": "QDRANT_W2V",
            "description": "TF-IDF + Glove with Qdrant"
        }
        print(f"SUCCESS: Initialized 'glove_qdrant' retriever.")
    except Exception as e:
        print(f"FAILED to initialize TF-IDF+Word2Vec retriever: {e}")

    if not RETRIEVERS:
        print("\nCRITICAL WARNING: No retrievers were successfully initialized! Chatbot cannot function.")
    else:
        print(f"\nInitialized {len(RETRIEVERS)} retriever(s).")

# --- 2. Core RAG Functions ---
def embed_query_pretrained_phobert(query_text: str, model: SentenceTransformer):
    """Embeds query using the provided pretrained_phobert model (used by original Atlas retriever)."""
    if not model:
        print("Error: pretrained_phobert Embedding model is not loaded for embed_query_pretrained_phobert.")
        return None
    if not isinstance(query_text, str) or not query_text.strip():
        print("Error: Query text must be a non-empty string.")
        return None
    try:
        query_embedding = model.encode([query_text], convert_to_tensor=False, show_progress_bar=False)
        return query_embedding[0]
    except Exception as e:
        print(f"Error embedding query with pretrained_phobert: {e}")
        return None

def search_atlas_pretrained_phobert(query_embedding: np.ndarray, top_n: int = 3):
    """
    Searches MongoDB Atlas using $vectorSearch.
    """
    if query_embedding is None:
        print("Error: Query embedding is None. Cannot search Atlas.")
        return []
    
    query_vector_list = query_embedding.tolist()
    retrieved_docs = []
    mongo_client = None
    try:
        mongo_client = pymongo.MongoClient(ATLAS_SRV_STRING, serverSelectionTimeoutMS=10000)
        mongo_client.admin.command('ping') 
        db = mongo_client[ATLAS_DATABASE_NAME]
        collection = db[ATLAS_COLLECTION_NAME]

        pipeline = [
            {
                "$vectorSearch": {
                    "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                    "queryVector": query_vector_list,
                    "path": "embedding", 
                    "numCandidates": top_n * 15,
                    "limit": top_n
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "title": 1,
                    "law_id": 1,
                    "article_id": 1,
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        results_cursor = collection.aggregate(pipeline)
        retrieved_docs = list(results_cursor)
        if not retrieved_docs:
            print("No documents found in Atlas via $vectorSearch.")
            
    except pymongo.errors.ConfigurationError as e:
        print(f"MongoDB Configuration Error (check SRV string, credentials, Atlas reachability): {e}")
    except pymongo.errors.OperationFailure as e:
        print(f"MongoDB Operation Failure (check index name '{ATLAS_VECTOR_SEARCH_INDEX_NAME}', query syntax, index build status): {e}")
    except pymongo.errors.ConnectionFailure as err:
        print(f"MongoDB Atlas connection failed during search: {err}.")
    except Exception as e:
        print(f"An error occurred during Atlas document search: {e}")
    finally:
        if mongo_client:
            mongo_client.close()
    return retrieved_docs

def retrieve_documents_unified(query_text: str, retriever_key: str, top_n: int = 3):
    """
    Unified function to retrieve documents using the selected retriever.
    Returns documents in a common format: list of dicts with
    {'text', 'title', 'law_id', 'article_id', 'score'}.
    """
    if retriever_key not in RETRIEVERS:
        print(f"Error: Retriever '{retriever_key}' not found.")
        return []

    retriever_config = RETRIEVERS[retriever_key]
    retriever_instance = retriever_config["instance"]
    retriever_type = retriever_config["type"]
    formatted_docs = []

    print(f"\nUsing retriever: {retriever_config['description']} for query: \"{query_text[:50]}...\"")

    # helper function to get article by IDs
    def get_article_by_ids(law_id, article_id):
        global index
        article = index.get(law_id, {}).get(article_id)
        if article:
            return {
                "title": article["title"],
                "text": article["text"]
            }
        return None
    
    try:
        if retriever_type == "ATLAS_pretrained_phobert":
            pretrained_phobert_model = retriever_instance
            query_embedding = embed_query_pretrained_phobert(query_text, pretrained_phobert_model)
            if query_embedding is None:
                return []
            atlas_raw_docs = search_atlas_pretrained_phobert(query_embedding, top_n)
            for doc in atlas_raw_docs:
                law_id = doc.get("law_id", "N/A")
                articale_id = doc.get("article_id", "N/A")
                article = get_article_by_ids(law_id, articale_id)
                formatted_docs.append({
                    "text": article.get("text", "N/A") if article else doc.get("text", "N/A"),
                    "title": article.get("title", "N/A") if article else doc.get("title", "N/A"),
                    "law_id": law_id,
                    "article_id": articale_id,
                    "score": doc.get("similarity_score", 0.0)
                })

        elif retriever_type.startswith("QDRANT_"):
            
            raw_hits = retriever_instance.search(query_text, limit=top_n)

            if retriever_type == "QDRANT_BM25":
                for hit in raw_hits: 
                    payload = hit.get('payload', {})
                    law_id = payload.get("law_id", "N/A")
                    article_id = payload.get("article_id", "N/A")
                    article = get_article_by_ids(law_id, article_id)
                    formatted_docs.append({
                        "text": article.get("text", "N/A") if article else payload.get("text", "N/A"),
                        "title": article.get("title", "N/A") if article else payload.get("title", "N/A"),
                        "law_id": payload.get("law_id", "N/A"),
                        "article_id": payload.get("article_id", "N/A"),
                        "score": hit.get("score", 0.0)
                    })
            else:
                if hasattr(raw_hits, 'points'):
                    for hit_point in raw_hits.points:
                        payload = hit_point.payload if hit_point.payload else {}
                        law_id = payload.get("law_id", "N/A")
                        article_id = payload.get("article_id", "N/A")
                        article = get_article_by_ids(law_id, article_id)
                        formatted_docs.append({
                            "text": article.get("text", "N/A") if article else payload.get("text", "N/A"),
                            "title": article.get("title", "N/A") if article else payload.get("title", "N/A"),
                            "law_id": payload.get("law_id", "N/A"),
                            "article_id": payload.get("article_id", "N/A"),
                            "score": hit_point.score
                        })
                else:
                    print(f"Warning: Unexpected result format from {retriever_key}. Expected Qdrant QueryResponse.")

        else:
            print(f"Error: Unknown retriever type '{retriever_type}' for key '{retriever_key}'.")
            return []

    except Exception as e:
        print(f"Error during document retrieval with '{retriever_key}': {e}")
        import traceback
        traceback.print_exc()
        return []
    
    if not formatted_docs:
        print("No documents retrieved.")
    return formatted_docs


def generate_answer(user_query: str, retrieved_docs: list, model, tokenizer, device):
    """Generates an answer using the LLM based on the query and retrieved documents."""
    if not model or not tokenizer:
        return "Lỗi: Mô hình LLM hoặc tokenizer chưa được tải."
    if not user_query:
        return "Vui lòng cung cấp câu hỏi."

    context_str = ""
    if retrieved_docs:
        context_str += "Dựa vào các thông tin pháp lý sau đây:\n"
        for i, doc in enumerate(retrieved_docs[:10]): # Use top 10
            context_str += f"\n--- Trích đoạn {i+1} (Từ Điều: {doc.get('title', 'N/A')}, Luật/Văn bản: {doc.get('law_id', 'N/A')}, Độ liên quan: {doc.get('score', 0.0):.4f}) ---\n"
            context_str += str(doc.get('text', '')).strip()
            context_str += f"\n--- Hết trích đoạn {i+1} ---\n"
        context_str += "\nHãy trả lời câu hỏi sau đây. Chỉ sử dụng thông tin được cung cấp trong các trích đoạn trên. Trả lời một cách ngắn gọn và chính xác.\n"
    else:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi này."

    full_prompt_for_chat = f"{context_str}\nCâu hỏi: {user_query}\nTrả lời:"
    
    chat_messages = [{"role": "user", "content": full_prompt_for_chat}]
    
    try:
        # Format: <start_of_turn>user\n{YOUR_PROMPT_HERE}<end_of_turn>\n<start_of_turn>model\n

        templated_input_string = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(templated_input_string, return_tensors="pt", padding=True).to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,  # Slightly increased for potentially more complex answers
            temperature=0.5,    # Slightly lower for more factual responses
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        # Decode only the generated part
        input_token_len = inputs.input_ids.shape[-1]
        generated_tokens = outputs[0][input_token_len:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Fallback if the above decoding fails or is empty
        if not answer:
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            # Attempt to remove the prompt part manually if it's included
            if full_prompt_for_chat in answer: # This is a crude way
                 answer = answer.split(full_prompt_for_chat)[-1].strip().split("Trả lời:")[-1].strip()


        return answer
    except Exception as e:
        print(f"Error during answer generation: {e}")
        import traceback
        traceback.print_exc()
        return "Xin lỗi, tôi gặp lỗi khi tạo câu trả lời."

# --- 3. Main Chat Loop ---
def main_chat_loop(llm_tok, llm_gen_model, current_llm_device):
    print("\nChào mừng bạn đến với Chatbot Tư vấn Pháp lý ZaloAI (Đa Truy xuất)!")
    print("Nhập 'quit' hoặc 'exit' để thoát.")

    if not RETRIEVERS:
        print("Lỗi nghiêm trọng: Không có retriever nào được tải. Chatbot không thể hoạt động.")
        return

    print("\nVui lòng chọn một mô hình truy xuất thông tin:")
    retriever_options_keys = list(RETRIEVERS.keys())
    for i, key in enumerate(retriever_options_keys):
        print(f"  {i+1}. {RETRIEVERS[key]['description']}")

    selected_retriever_key = None
    while True:
        try:
            choice_idx = int(input(f"Nhập lựa chọn của bạn (1-{len(retriever_options_keys)}): ")) - 1
            if 0 <= choice_idx < len(retriever_options_keys):
                selected_retriever_key = retriever_options_keys[choice_idx]
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
        except ValueError:
            print("Vui lòng nhập một số.")
    
    print(f"\nBạn đã chọn retriever: {RETRIEVERS[selected_retriever_key]['description']}")
    print("-------------------------------------------------------------")

    while True:
        user_query = input("\nBạn hỏi: ").strip()
        if user_query.lower() in ['quit', 'exit']:
            print("Cảm ơn bạn đã sử dụng chatbot! Tạm biệt.")
            break
        if not user_query:
            print("Vui lòng nhập câu hỏi.")
            continue

        relevant_docs = retrieve_documents_unified(user_query, selected_retriever_key, top_n=3)

        if relevant_docs:
            print("\n--- Thông tin tham khảo được tìm thấy: ---")
            for i, doc in enumerate(relevant_docs):
                score_val = doc.get('score', doc.get('similarity_score', 0.0))
                print(f"  {i+1}. Điều: {doc.get('title', 'N/A')} (VB: {doc.get('law_id', 'N/A')}, Art: {doc.get('article_id', 'N/A')}) | Độ liên quan: {score_val:.4f}")
            print("-----------------------------------------")
        else:
            print("--- Không tìm thấy thông tin tham khảo. ---")

        final_answer = generate_answer(user_query, relevant_docs, llm_gen_model, llm_tok, current_llm_device)
        
        print(f"\nBot Trả lời:\n{final_answer}")
        print("-------------------------------------------------------------")

# --- Script Execution ---
if __name__ == '__main__':
    print("="*50)
    print("Khởi tạo Chatbot Tư vấn Pháp lý ZaloAI (Đa Truy xuất)")
    print("="*50)
    
    print("\n--- Bước 1: Tải mô hình Ngôn ngữ Lớn (LLM) ---")
    gemma_tokenizer, gemma_model, llm_device = load_llm_model()

    original_pretrained_phobert_model = None
    if ATLAS_SRV_STRING and "username:password" not in ATLAS_SRV_STRING :
        print("\n--- Bước 2: Tải mô hình pretrained_phobert (cho Atlas Retriever) ---")
        original_pretrained_phobert_model = load_original_pretrained_phobert_model()
    else:
        print("\n--- Bước 2: Bỏ qua tải mô hình pretrained_phobert (Atlas SRV không được cấu hình đúng) ---")


    print("\n--- Bước 3: Khởi tạo các mô hình Truy xuất (Retrievers) ---")
    initialize_all_retrievers(original_pretrained_phobert_instance=original_pretrained_phobert_model)

    if gemma_tokenizer and gemma_model and RETRIEVERS:
        print("\n--- Kiểm tra kết nối ban đầu (nếu có) ---")
        if 'original_pretrained_phobert_atlas' in RETRIEVERS:
            print("Kiểm tra kết nối MongoDB Atlas cho retriever pretrained_phobert...")
            try:
                temp_client = pymongo.MongoClient(ATLAS_SRV_STRING, serverSelectionTimeoutMS=5000)
                temp_client.admin.command('ping')
                print("Kết nối MongoDB Atlas thành công.")
                # db_check = temp_client[ATLAS_DATABASE_NAME]
                # collection_check = db_check[ATLAS_COLLECTION_NAME]
                # doc_count = collection_check.count_documents({})
                # print(f"Tìm thấy {doc_count} tài liệu trong bộ sưu tập '{ATLAS_COLLECTION_NAME}'.")
            except Exception as e:
                print(f"Lỗi kết nối MongoDB Atlas: {e}. Retriever pretrained_phobert gốc có thể không hoạt động.")
            finally:
                if 'temp_client' in locals() and temp_client:
                    temp_client.close()
        
        print(f"\nChatbot sẵn sàng! LLM đang chạy trên thiết bị: {llm_device}.")
        main_chat_loop(gemma_tokenizer, gemma_model, llm_device)
    else:
        print("\nLỗi nghiêm trọng: Không thể tải LLM hoặc không có retriever nào được khởi tạo. Chatbot không thể khởi động.")
        if not gemma_tokenizer or not gemma_model:
            print(" - Vấn đề với LLM.")
        if not RETRIEVERS:
            print(" - Không có retriever nào khả dụng.")

    print("\nChatbot đã kết thúc.")